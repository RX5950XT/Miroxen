// 區塊縮放 GEMM：(1) 用主機端依格式解碼的值做 f32 GEMM，驗證資料／縮放排列與 cuBLASLt 一致；
// (2) 印出相對原始 bf16 GEMM 的量化誤差。不支援的卡（sm < 100）直接略過。
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/blockscale.h"

using namespace mx;

static int64_t pad(int64_t v, int64_t m) { return (v + m - 1) / m * m; }
static size_t sf_off(int64_t r, int64_t kb, int64_t nkb_pad) {
  return size_t(r / 128) * nkb_pad * 128 + size_t(kb / 4) * 512 + (r % 32) * 16 + (r % 128) / 32 * 4 + kb % 4;
}

// 把裝置端量化矩陣解回 f32（主機端）。
static std::vector<float> decode(QMode m, const QMat& qm, int64_t rows, int64_t K) {
  std::vector<uint8_t> q(qdata_bytes(m, rows, K)), sf(qscale_bytes(m, rows, K));
  MX_CUDA(cudaMemcpy(q.data(), qm.q, q.size(), cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(sf.data(), qm.sf, sf.size(), cudaMemcpyDeviceToHost));
  float amax = 0;
  if (m == QMode::NVFP4) MX_CUDA(cudaMemcpy(&amax, qm.amax, 4, cudaMemcpyDeviceToHost));
  const int V = m == QMode::NVFP4 ? 16 : 32;
  const int64_t nkb = pad(K / V, 4);
  static const float e2m1[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  std::vector<float> out(size_t(rows) * K);
  for (int64_t r = 0; r < rows; r++)
    for (int64_t k = 0; k < K; k++) {
      const uint8_t s = sf[sf_off(r, k / V, nkb)];
      float v;
      if (m == QMode::NVFP4) {
        const float g = amax > 0 ? amax / (6.f * 448.f) : 1.f;
        __nv_fp8_e4m3 se;
        se.__x = s;
        const int nib = (q[(r * K + k) / 2] >> ((k & 1) * 4)) & 15;
        v = (nib & 8 ? -1.f : 1.f) * e2m1[nib & 7] * float(se) * g;
      } else {
        __nv_fp8_e4m3 x;
        x.__x = q[r * K + k];
        v = float(x) * std::ldexp(1.f, int(s) - 127);
      }
      out[r * K + k] = v;
    }
  return out;
}

static double rel_rms(const std::vector<float>& a, const std::vector<float>& ref) {
  double e = 0, s = 0;
  for (size_t i = 0; i < a.size(); i++) e += (a[i] - ref[i]) * double(a[i] - ref[i]), s += double(ref[i]) * ref[i];
  return std::sqrt(e / s);
}

static std::vector<float> to_f32(const DevBuf& y, size_t n) {
  std::vector<bf16> h(n);
  MX_CUDA(cudaMemcpy(h.data(), y.get(), n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> f(n);
  for (size_t i = 0; i < n; i++) f[i] = __bfloat162float(h[i]);
  return f;
}

static void run(Ctx& c, QMode m, int64_t M, int64_t N, int64_t K) {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  std::vector<bf16> x(M * K), w(N * K);
  for (auto& v : x) v = __float2bfloat16(nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f));  // 帶離群值
  for (auto& v : w) v = __float2bfloat16(nd(rng) * 0.02f);
  DevBuf dx(x.size() * 2, c.device), dw(w.size() * 2, c.device), y(M * N * 2, c.device), yr(M * N * 2, c.device);
  MX_CUDA(cudaMemcpy(dx.get(), x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(dw.get(), w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  auto alloc = [&](int64_t rows, DevBuf (&b)[3]) {
    b[0] = DevBuf(qdata_bytes(m, rows, K), c.device);
    b[1] = DevBuf(qscale_bytes(m, rows, K), c.device);
    b[2] = DevBuf(16, c.device);
    return QMat{b[0].get(), b[1].as<uint8_t>(), b[2].as<float>()};
  };
  DevBuf bx[3], bw[3];
  QMat qx = alloc(M, bx), qw = alloc(N, bw);
  quantize_blockscaled(m, dx.as<bf16>(), K, M, K, qx, c.stream);
  quantize_blockscaled(m, dw.as<bf16>(), K, N, K, qw, c.stream);
  gemm_blockscaled(c, m, qw, qx, y.as<bf16>(), N, M, N, K);
  gemm_bf16(c, dx.as<bf16>(), K, dw.as<bf16>(), K, yr.get(), false, N, M, N, K);
  c.sync();

  // (1) 依格式解碼後的 f32 GEMM。
  auto fx = decode(m, qx, M, K), fw = decode(m, qw, N, K);
  DevBuf gx(fx.size() * 4, c.device), gw(fw.size() * 4, c.device), gy(M * N * 4, c.device);
  MX_CUDA(cudaMemcpy(gx.get(), fx.data(), fx.size() * 4, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(gw.get(), fw.data(), fw.size() * 4, cudaMemcpyHostToDevice));
  gemm_f32(c, gx.as<float>(), K, gw.as<float>(), K, gy.as<float>(), N, M, N, K);
  c.sync();
  std::vector<float> ydec(M * N);
  MX_CUDA(cudaMemcpy(ydec.data(), gy.get(), ydec.size() * 4, cudaMemcpyDeviceToHost));
  auto yq = to_f32(y, M * N), yb = to_f32(yr, M * N);
  const double layout_err = rel_rms(yq, ydec), quant_err = rel_rms(yq, yb);
  std::vector<float> xf(x.size());
  for (size_t i = 0; i < x.size(); i++) xf[i] = __bfloat162float(x[i]);
  const double elem_err = rel_rms(fx, xf);
  std::printf("%s M=%lld N=%lld K=%lld：對解碼值 %.2e、啟用值元素誤差 %.2f%%、對 bf16 GEMM %.2f%%\n", qmode_name(m),
              (long long)M, (long long)N, (long long)K, layout_err, elem_err * 100, quant_err * 100);
  MX_CHECK(layout_err < 5e-3, "%s 排列錯誤：對解碼值誤差 %.3g", qmode_name(m), layout_err);
  MX_CHECK(elem_err < (m == QMode::NVFP4 ? 0.12 : 0.04), "%s 量化誤差過大 %.3g", qmode_name(m), elem_err);
  MX_CHECK(quant_err < (m == QMode::NVFP4 ? 0.15 : 0.05), "%s GEMM 誤差過大 %.3g", qmode_name(m), quant_err);
}

// 量化核心吞吐量（只印數字）。
static void bench(Ctx& c, QMode m, int64_t M, int64_t K) {
  DevBuf x(M * K * 2, c.device), q(qdata_bytes(m, M, K), c.device), sf(qscale_bytes(m, M, K), c.device),
      am(16, c.device);
  MX_CUDA(cudaMemset(x.get(), 0x3c, x.size()));
  QMat qm{q.get(), sf.as<uint8_t>(), am.as<float>()};
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  quantize_blockscaled(m, x.as<bf16>(), K, M, K, qm, c.stream);
  cudaEventRecord(e0, c.stream);
  for (int i = 0; i < 20; i++) quantize_blockscaled(m, x.as<bf16>(), K, M, K, qm, c.stream);
  cudaEventRecord(e1, c.stream);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  const double us = ms * 1000 / 20, gb = (x.size() + q.size()) / (us * 1e3);
  std::printf("量化 %s %lldx%lld：%.1f us（%.0f GB/s）\n", qmode_name(m), (long long)M, (long long)K, us, gb);
  cudaEventDestroy(e0), cudaEventDestroy(e1);
}

int main() {
  try {
    if (!qmode_supported(QMode::NVFP4, 0)) {
      std::printf("GPU0 不支援區塊縮放 GEMM，略過\n");
      return 0;
    }
    Ctx c(0, 16 << 20, 16 << 20);
    for (QMode m : {QMode::MXFP8, QMode::NVFP4}) {
      run(c, m, 300, 512, 1024);   // M 不是 128 的倍數
      run(c, m, 129, 384, 4160);   // K/32 不是 4 的倍數（縮放欄補齊）
    }
    for (QMode m : {QMode::MXFP8, QMode::NVFP4}) bench(c, m, 4096, 4096), bench(c, m, 4096, 12288);
    std::printf("blockscale_test PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
}
