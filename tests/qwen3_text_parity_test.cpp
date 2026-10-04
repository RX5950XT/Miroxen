// Qwen3-VL 文字編碼器與 transformers 的數值比對（tools/ref/qwen3_text_ref.py 產生參考）。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "core/dequant.h"
#include "text/qwen3_text.h"
#include "text/tokenizer.h"

using namespace mx;

int main() {
  const char* gguf = "models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q8_0.gguf";
  const char* ref = "models/ref/qwen3_text_ref.safetensors";
  const char* tokp = "models/qwen-image-2.1/hf/processor/tokenizer.json";
  if (!std::filesystem::exists(gguf) || !std::filesystem::exists(ref)) {
    std::printf("SKIP：缺少 %s 或 %s\n", gguf, ref);
    return 0;
  }
  try {
    TensorStore st, rs;
    st.add_file(gguf, "\x01");
    rs.add_file(ref, "\x01");
    const WeightRef& ir = rs.get("ids");
    std::vector<int32_t> ids(static_cast<const int32_t*>(ir.data),
                             static_cast<const int32_t*>(ir.data) + ir.numel());
    int fails = 0;
    if (std::filesystem::exists(tokp)) {  // tokenizer 與 Python tokenizers 一致
      auto tok = Tokenizer::load(tokp);
      bool same = tok->encode(rs.meta_str("text")) == ids;
      std::printf("tokenizer 與參考一致：%s\n", same ? "PASS" : "FAIL");
      fails += !same;
    }
    Qwen3TextEncoder te(st, 0);
    Ctx c(0, size_t(512) << 20, te.scratch_bytes() * 2);
    const int64_t n = int64_t(ids.size()), dim = te.dim();
    DevBuf out(size_t(n) * dim * 2, 0);
    te.encode(c, ids, out.as<bf16>());
    std::vector<uint16_t> raw(n * dim);
    MX_CUDA(cudaMemcpy(raw.data(), out.get(), raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> got(n * dim), expect(n * dim);
    dequant_cpu(raw.data(), GType::BF16, n * dim, got.data());
    const WeightRef& hr = rs.get("hidden");
    dequant_cpu(hr.data, hr.type, hr.numel(), expect.data());
    // 第 0 個 token（<|im_start|>）在 Qwen 的激活值極大，分開統計。
    for (int part = 0; part < 2; part++) {
      double num = 0, den = 0;
      for (int64_t t = part ? 1 : 0; t < (part ? n : 1); t++)
        for (int64_t i = 0; i < dim; i++) {
          double d = got[t * dim + i] - expect[t * dim + i];
          num += d * d;
          den += double(expect[t * dim + i]) * expect[t * dim + i];
        }
      double re = std::sqrt(num / (den + 1e-30));
      bool ok = re < 0.03;
      fails += !ok;
      std::printf("%s：相對誤差 %.4f → %s\n", part ? "token 1..n" : "token 0", re, ok ? "PASS" : "FAIL");
    }
    return fails ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
