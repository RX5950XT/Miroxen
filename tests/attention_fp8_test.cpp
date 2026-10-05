// FP8/INT8 注意力：對樸素參考實作的相對誤差，並和 bf16 FlashAttention 比速度。sm < 89 略過。
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "kernels/attention.h"

using namespace mx;

struct Case {
  int sq, skv, hq, hkv, d, pad;
  float kbias;  // K 每通道偏移（DiT 的 K 常見），測平均值平滑
};

static DevBuf rand_buf(size_t n, std::mt19937& rng, float bias_scale, int period) {
  std::normal_distribution<float> nd;
  std::vector<float> bias(period);
  for (auto& b : bias) b = nd(rng) * bias_scale;
  std::vector<bf16> h(n);
  for (size_t i = 0; i < n; i++) h[i] = __float2bfloat16(nd(rng) + bias[i % period]);
  DevBuf d(n * sizeof(bf16), 0);
  MX_CUDA(cudaMemcpy(d.get(), h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  return d;
}

static std::vector<float> host(const DevBuf& b, size_t n) {
  std::vector<bf16> h(n);
  MX_CUDA(cudaMemcpy(h.data(), b.get(), n * sizeof(bf16), cudaMemcpyDeviceToHost));
  std::vector<float> f(n);
  for (size_t i = 0; i < n; i++) f[i] = __bfloat162float(h[i]);
  return f;
}

static double rel(const std::vector<float>& a, const std::vector<float>& r) {
  double e = 0, s = 0;
  for (size_t i = 0; i < a.size(); i++) e += (a[i] - r[i]) * double(a[i] - r[i]), s += double(r[i]) * r[i];
  return std::sqrt(e / s);
}

static void check(const Case& c) {
  std::mt19937 rng(c.sq * 31 + c.skv);
  AttnParams p;
  p.Sq = c.sq, p.Skv = c.skv, p.Hq = c.hq, p.Hkv = c.hkv, p.D = c.d;
  const int64_t qs = int64_t(c.hq) * c.d + c.pad, ks = int64_t(c.hkv) * c.d + c.pad;
  if (c.pad) p.q_stride = p.o_stride = qs, p.kv_stride = ks;
  DevBuf q = rand_buf(size_t(c.sq) * qs, rng, 0.f, 1), k = rand_buf(size_t(c.skv) * ks, rng, c.kbias, int(ks)),
         v = rand_buf(size_t(c.skv) * ks, rng, 0.5f, int(ks));
  DevBuf o(size_t(c.sq) * qs * 2, 0), r(size_t(c.sq) * qs * 2, 0), b(size_t(c.sq) * qs * 2, 0);
  DevBuf ws(flash_attn_fp8_workspace(p), 0);
  MX_CUDA(cudaMemset(o.get(), 0, o.size()));
  MX_CUDA(cudaMemset(r.get(), 0, r.size()));
  MX_CUDA(cudaMemset(b.get(), 0, b.size()));
  flash_attn_fp8(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), o.as<bf16>(), p, ws.get(), nullptr);
  flash_attn(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), b.as<bf16>(), p, nullptr);
  attn_reference(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), r.as<bf16>(), p, nullptr);
  MX_CUDA(cudaDeviceSynchronize());
  const size_t n = size_t(c.sq) * qs;
  auto fo = host(o, n), fr = host(r, n), fb = host(b, n);
  const double e8 = rel(fo, fr), e16 = rel(fb, fr);
  std::printf("Sq=%d Skv=%d H=%d/%d D=%d pad=%d kbias=%.0f：fp8 %.2f%%、bf16 %.2f%%\n", c.sq, c.skv, c.hq, c.hkv, c.d,
              c.pad, c.kbias, e8 * 100, e16 * 100);
  MX_CHECK(e8 < 0.04, "fp8 注意力誤差過大：%.3g", e8);
}

static void bench(int S, int H, int D) {
  std::mt19937 rng(1);
  AttnParams p;
  p.Sq = p.Skv = S, p.Hq = p.Hkv = H, p.D = D;
  const size_t n = size_t(S) * H * D;
  DevBuf q = rand_buf(n, rng, 0.f, 1), k = rand_buf(n, rng, 0.f, 1), v = rand_buf(n, rng, 0.f, 1), o(n * 2, 0);
  DevBuf ws(flash_attn_fp8_workspace(p), 0);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  double ms[2];
  for (int m = 0; m < 2; m++) {
    auto go = [&] {
      if (m) flash_attn_fp8(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), o.as<bf16>(), p, ws.get(), nullptr);
      else flash_attn(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), o.as<bf16>(), p, nullptr);
    };
    go();
    cudaEventRecord(e0);
    for (int i = 0; i < 5; i++) go();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float t = 0;
    cudaEventElapsedTime(&t, e0, e1);
    ms[m] = t / 5;
  }
  const double fl = 4.0 * S * S * H * D;
  std::printf("S=%d H=%d D=%d：bf16 %.2f ms（%.0f TFLOPS）、fp8 %.2f ms（%.0f TFLOPS，含量化）\n", S, H, D, ms[0],
              fl / ms[0] / 1e9, ms[1], fl / ms[1] / 1e9);
  cudaEventDestroy(e0), cudaEventDestroy(e1);
}

int main() {
  try {
    int major = 0, minor = 0;
    MX_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    MX_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    if (major * 10 + minor < 89) {
      std::printf("GPU0 不支援 FP8 tensor core，略過\n");
      return 0;
    }
    MX_CUDA(cudaSetDevice(0));
    check({300, 300, 4, 4, 128, 0, 0.f});
    check({257, 1000, 4, 2, 128, 0, 4.f});    // GQA、Skv 非 64 倍數、K 帶偏移
    check({130, 77, 2, 2, 64, 8, 2.f});       // D=64、自訂列距
    check({1024, 4352, 2, 2, 128, 0, 8.f});
    bench(4352, 32, 128);
    bench(16384, 24, 128);
    bench(7076, 56, 128);
    std::printf("attention_fp8_test PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
}
