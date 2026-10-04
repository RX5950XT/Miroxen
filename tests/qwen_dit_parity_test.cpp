// Qwen-Image-2.1 DiT 與 diffusers 的數值比對（需要 models/ 底下的權重與 tools/ref/qwen_dit_ref.py 的輸出）。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "core/dequant.h"
#include "models/qwen_image/dit.h"

using namespace mx;

static std::vector<float> host_f32(const WeightRef& r) {
  std::vector<float> v(r.numel());
  dequant_cpu(r.data, r.type, r.numel(), v.data());
  return v;
}

static double rel_err(const std::vector<float>& a, const std::vector<float>& b, double& maxabs) {
  double num = 0, den = 0;
  maxabs = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = double(a[i]) - b[i];
    num += d * d;
    den += double(b[i]) * b[i];
    maxabs = std::max(maxabs, std::fabs(d));
  }
  return std::sqrt(num / (den + 1e-30));
}

int main() {
  const char* gguf = "models/qwen-image-2.1/qwen_image_2.1-Q8_0.gguf";
  const char* ref = "models/ref/qwen_dit_ref.safetensors";
  if (!std::filesystem::exists(gguf) || !std::filesystem::exists(ref)) {
    std::printf("SKIP：缺少 %s 或 %s\n", gguf, ref);
    return 0;
  }
  try {
    TensorStore st, rs;
    st.add_file(gguf);
    rs.add_file(ref, "\x01");
    const int64_t T = std::atoll(rs.meta_str("T").c_str()), H = std::atoll(rs.meta_str("H").c_str()),
                  W = std::atoll(rs.meta_str("W").c_str());
    const float t1 = float(std::atof(rs.meta_str("t1").c_str())), t2 = float(std::atof(rs.meta_str("t2").c_str()));
    QwenImageDit dit(st, 0);
    Ctx c(0, size_t(2) << 30, dit.scratch_bytes());
    const int64_t N = H * W, C = 64;
    DevBuf txt(size_t(T) * 4096 * 2, 0), lat(N * C * 4, 0), out(N * C * 4, 0);
    const WeightRef& tr = rs.get("txt");
    MX_CHECK(tr.type == GType::BF16, "txt 應為 bf16");
    MX_CUDA(cudaMemcpy(txt.get(), tr.data, tr.nbytes, cudaMemcpyHostToDevice));
    QwenImageSeq seq;
    dit.prepare(c, seq, txt.as<bf16>(), T, H, W);
    int fails = 0;
    for (int step = 0; step < 2; step++) {
      auto l = host_f32(rs.get(step == 0 ? "lat1" : "lat2"));
      auto expect = host_f32(rs.get(step == 0 ? "out1" : "out2"));
      MX_CUDA(cudaMemcpy(lat.get(), l.data(), l.size() * 4, cudaMemcpyHostToDevice));
      dit.forward(c, seq, lat.as<float>(), step == 0 ? t1 : t2, out.as<float>());
      c.sync();
      std::vector<float> got(N * C);
      MX_CUDA(cudaMemcpy(got.data(), out.get(), got.size() * 4, cudaMemcpyDeviceToHost));
      double maxabs, re = rel_err(got, expect, maxabs);
      bool ok = re < 0.03;
      fails += !ok;
      std::printf("step %d（%s）：相對誤差 %.4f、最大絕對誤差 %.4f → %s\n", step,
                  step == 0 ? "prefill" : "cached", re, maxabs, ok ? "PASS" : "FAIL");
    }
    std::printf("arena peak %.1f MiB\n", c.act.peak() / 1048576.0);
    return fails ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
