// Qwen3-VL 視覺塔與圖文編碼和 transformers 的數值比對（需要 models/ 底下的權重與 tools/ref/qwen3_vl_ref.py 的輸出）。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "core/dequant.h"
#include "text/qwen3_text.h"
#include "text/qwen3_vision.h"

using namespace mx;

static std::vector<float> host_f32(const WeightRef& r) {
  std::vector<float> v(r.numel());
  dequant_cpu(r.data, r.type, r.numel(), v.data());
  return v;
}

static std::vector<float> bf16_vec(const std::vector<bf16>& b) {
  std::vector<float> v(b.size());
  for (size_t i = 0; i < b.size(); i++) v[i] = __bfloat162float(b[i]);
  return v;
}

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); i++) {
    double d = double(a[i]) - b[i];
    num += d * d;
    den += double(b[i]) * b[i];
  }
  return a.size() == b.size() ? std::sqrt(num / (den + 1e-30)) : 1e9;
}

int main() {
  const char* mmproj = "models/image/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf";
  const char* te = "models/image/qwen-image-2.1/Qwen3VL-8B-Instruct-Q8_0.gguf";
  const char* ref = "models/ref/qwen3_vl_ref.safetensors";
  for (const char* p : {mmproj, te, ref})
    if (!std::filesystem::exists(p)) {
      std::printf("SKIP：缺少 %s\n", p);
      return 0;
    }
  try {
    TensorStore vs, ts, rs;
    vs.add_file(mmproj, "\x01");
    ts.add_file(te, "\x01");
    rs.add_file(ref, "\x01");
    const int h = std::atoi(rs.meta_str("h").c_str()), w = std::atoi(rs.meta_str("w").c_str());
    const int64_t drop = std::atoll(rs.meta_str("drop_idx").c_str());
    const WeightRef& rgb = rs.get("rgb");
    MX_CHECK(rgb.numel() == int64_t(h) * w * 3, "rgb 大小不符");

    Qwen3VisionEncoder venc(vs, 0);
    Ctx vc(0, venc.act_bytes(h, w), venc.scratch_bytes());
    VisionFeatures f = venc.encode(vc, static_cast<const uint8_t*>(rgb.data), h, w);
    int fails = 0;
    auto check = [&](const char* name, const std::vector<float>& got, double tol) {
      double e = rel_err(got, host_f32(rs.get(name)));
      bool ok = e < tol;
      fails += !ok;
      std::printf("%-8s 相對誤差 %.4f → %s\n", name, e, ok ? "PASS" : "FAIL");
    };
    // 官方 bf16 實作對 fp32 參考本身就差：embeds 5.5%、ds 1.4/2.4/3.8%（27 層 ViT 的 bf16 累積），門檻依此設定。
    check("embeds", bf16_vec(f.embeds), 0.08);
    for (size_t i = 0; i < f.deepstack.size(); i++) check(strf("ds%zu", i).c_str(), bf16_vec(f.deepstack[i]), 0.06);

    const WeightRef& ir = rs.get("ids");
    std::vector<int32_t> ids(static_cast<const int32_t*>(ir.data), static_cast<const int32_t*>(ir.data) + ir.numel());
    Qwen3TextEncoder tenc(ts, 0);
    Ctx tc(0, size_t(1) << 30, tenc.scratch_bytes());
    TextVisionInput tv = build_text_vision(ids, {151655}, {&f});
    const int64_t n = int64_t(ids.size()), d = tenc.dim();
    DevBuf out(size_t(n) * d * 2, 0);
    tenc.encode(tc, ids, out.as<bf16>(), &tv);
    std::vector<bf16> hb(size_t(n - drop) * d);
    MX_CUDA(cudaMemcpy(hb.data(), out.as<bf16>() + drop * d, hb.size() * 2, cudaMemcpyDeviceToHost));
    if (std::getenv("MX_DUMP")) {
      FILE* fp = std::fopen(std::getenv("MX_DUMP"), "wb");
      std::fwrite(hb.data(), 2, hb.size(), fp);
      std::fclose(fp);
    }
    {  // 分段誤差：圖片前文字、圖片列、圖片後文字
      auto got = bf16_vec(hb), exp = host_f32(rs.get("hidden"));
      const int64_t r0 = tv.rows.front() - drop, r1 = tv.rows.back() + 1 - drop;
      auto seg = [&](int64_t a, int64_t b) {
        std::vector<float> x(got.begin() + a * d, got.begin() + b * d), y(exp.begin() + a * d, exp.begin() + b * d);
        return rel_err(x, y);
      };
      // 圖片列有超大激活值通道（1838、2276），官方 bf16 對 fp32 也差 15%；文字列對 bf16 官方是 5.6%。
      double e0 = seg(0, r0), e1 = seg(r0, r1), e2 = seg(r1, n - drop);
      bool ok = e0 < 0.03 && e2 < 0.03 && e1 < 0.3;
      fails += !ok;
      std::printf("hidden   圖片前文字 %.4f、圖片列 %.4f、圖片後文字 %.4f → %s\n", e0, e1, e2, ok ? "PASS" : "FAIL");
    }
    return fails ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
