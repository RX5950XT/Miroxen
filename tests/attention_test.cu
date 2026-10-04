#include "kernels/attention.h"
#include "core/tensor.h"
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace mx;

namespace {

struct Case {
  int sq, skv, hq, hkv, d, causal = 0, offset = 0, pad = 0;
  float magnitude = 1.f, scale = 0.f;
};

__device__ unsigned hash(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}

// Box-Muller：固定 seed，輸入是在 GPU 產生的獨立常態亂數。
__global__ void fill_normal(bf16* data, size_t n, unsigned seed, float magnitude) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    float u = (float(hash(unsigned(i) ^ seed) >> 8) + 1.f) / 16777217.f;
    float v = (float(hash(unsigned(i) ^ (seed + 0x9e3779b9U)) >> 8) + 1.f) / 16777217.f;
    data[i] = __float2bfloat16(magnitude * sqrtf(-2.f * logf(u)) * cosf(6.28318530718f * v));
  }
}

void fill(DevBuf& b, unsigned seed, float magnitude, cudaStream_t stream = nullptr) {
  fill_normal<<<256, 256, 0, stream>>>(b.as<bf16>(), b.size() / sizeof(bf16), seed, magnitude);
  MX_KCHECK();
}

AttnParams params(const Case& c) {
  AttnParams p;
  p.Sq = c.sq; p.Skv = c.skv; p.Hq = c.hq; p.Hkv = c.hkv; p.D = c.d;
  p.causal = c.causal; p.q_offset = c.offset; p.scale = c.scale;
  if (c.pad) {
    p.q_stride = int64_t(c.hq) * c.d + c.pad;
    p.kv_stride = int64_t(c.hkv) * c.d + c.pad + 2;
    p.o_stride = int64_t(c.hq) * c.d + c.pad + 4;
  }
  return p;
}

std::vector<int> query_rows(const Case& c) {
  std::vector<int> rows;
  if (c.sq <= 129 || (c.sq <= 1000 && c.hq <= 4)) {
    for (int i = 0; i < c.sq; ++i) rows.push_back(i);
  } else {
    for (int i : {0, 1, 16, 17, 63, 64, 127, 128, c.sq / 2, c.sq - 2, c.sq - 1})
      if (i < c.sq) rows.push_back(i);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  }
  return rows;
}

void cpu_oracle(const Case& c, const AttnParams& p, const DevBuf& q, const DevBuf& k,
                const DevBuf& v, const std::vector<bf16>& expected) {
  if (c.sq > 17 || c.skv > 193) return;
  std::vector<bf16> hq(q.size() / 2), hk(k.size() / 2), hv(v.size() / 2);
  MX_CUDA(cudaMemcpy(hq.data(), q.get(), q.size(), cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(hk.data(), k.get(), k.size(), cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(hv.data(), v.get(), v.size(), cudaMemcpyDeviceToHost));
  int64_t qs = p.q_stride ? p.q_stride : int64_t(c.hq) * c.d;
  int64_t ks = p.kv_stride ? p.kv_stride : int64_t(c.hkv) * c.d;
  double scale = p.scale == 0.f ? 1. / std::sqrt(double(c.d)) : p.scale;
  for (int row = 0; row < c.sq; ++row) {
    int end = c.causal ? int(std::max(int64_t(0), std::min(int64_t(c.skv), int64_t(c.offset) + row + 1))) : c.skv;
    for (int head = 0; head < c.hq; ++head) {
      int kvhead = head / (c.hq / c.hkv);
      std::vector<double> weights(end);
      double maximum = -std::numeric_limits<double>::infinity(), sum = 0.;
      for (int key = 0; key < end; ++key) {
        double dot = 0.;
        for (int d = 0; d < c.d; ++d)
          dot += double(__bfloat162float(hq[row * qs + head * c.d + d])) *
                 __bfloat162float(hk[key * ks + kvhead * c.d + d]);
        weights[key] = dot * scale; maximum = std::max(maximum, weights[key]);
      }
      for (double& weight : weights) { weight = std::exp(weight - maximum); sum += weight; }
      for (int d = 0; d < c.d; ++d) {
        double output = 0.;
        for (int key = 0; key < end; ++key)
          output += weights[key] * __bfloat162float(hv[key * ks + kvhead * c.d + d]);
        output = sum > 0. ? output / sum : 0.;
        float ref = __bfloat162float(expected[(size_t(row) * c.hq + head) * c.d + d]);
        MX_CHECK(std::abs(ref - output) < .01 * std::max(1., std::abs(output)), "GPU reference disagrees with CPU oracle");
      }
    }
  }
}

void compare_outputs(const Case& c, int device, size_t rows, const std::vector<bf16>& actual,
                     const std::vector<bf16>& expected) {
  float error = 0.f, maximum = 0.f;
  double square_error = 0., square_expected = 0.;
  for (size_t i = 0; i < actual.size(); ++i) {
    float a = __bfloat162float(actual[i]), b = __bfloat162float(expected[i]);
    MX_CHECK(std::isfinite(a) && std::isfinite(b), "nonfinite output at %zu", i);
    error = std::max(error, std::abs(a - b)); maximum = std::max(maximum, std::abs(b));
    square_error += double(a - b) * (a - b); square_expected += double(b) * b;
  }
  float tolerance = .01f * std::max(1.f, maximum);
  double relative_rms = std::sqrt(square_error / std::max(square_expected, 1e-30));
  std::printf("GPU%d correctness Sq=%d Skv=%d H=%d/%d D=%d causal=%d offset=%d pad=%d input=%.1f rows=%zu max_abs=%.6g limit=%.6g rel_rms=%.6g\n",
              device, c.sq, c.skv, c.hq, c.hkv, c.d, c.causal, c.offset, c.pad, c.magnitude,
              rows, error, tolerance, relative_rms);
  MX_CHECK(error <= tolerance && relative_rms < .01, "flash/reference mismatch");
}

void check_guards(const DevBuf& out, const bf16* output, const Case& c, int64_t os,
                  const std::vector<int>& rows) {
  std::vector<unsigned short> guards(32);
  MX_CUDA(cudaMemcpy(guards.data(), out.get(), 32, cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(guards.data() + 16, output + size_t(os) * c.sq, 32, cudaMemcpyDeviceToHost));
  for (auto x : guards) MX_CHECK(x == 0xffff, "output guard was overwritten");
  if (os > c.hq * c.d) {
    std::vector<unsigned short> padding(size_t(os - c.hq * c.d));
    for (int row : rows) {
      MX_CUDA(cudaMemcpy(padding.data(), output + row * os + c.hq * c.d,
                         padding.size() * 2, cudaMemcpyDeviceToHost));
      for (auto x : padding) MX_CHECK(x == 0xffff, "output token padding was overwritten");
    }
  }
}

void check_case(const Case& c, int device, unsigned seed) {
  AttnParams p = params(c);
  int64_t qs = p.q_stride ? p.q_stride : int64_t(c.hq) * c.d;
  int64_t ks = p.kv_stride ? p.kv_stride : int64_t(c.hkv) * c.d;
  int64_t os = p.o_stride ? p.o_stride : int64_t(c.hq) * c.d;
  size_t rowbytes = size_t(c.hq) * c.d * sizeof(bf16);
  DevBuf q(size_t(qs) * c.sq * 2, device), k(size_t(ks) * c.skv * 2, device);
  DevBuf v(k.size(), device), out(size_t(os) * c.sq * 2 + 64, device);
  cudaStream_t stream;
  MX_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  fill(q, seed, c.magnitude, stream); fill(k, seed + 1, c.magnitude, stream); fill(v, seed + 2, c.magnitude, stream);
  MX_CUDA(cudaMemsetAsync(out.get(), 0xff, out.size(), stream));
  bf16* output = out.as<bf16>() + 16;
  flash_attn(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), output, p, stream);
  MX_CUDA(cudaStreamSynchronize(stream));
  std::vector<int> rows = query_rows(c);
  DevBuf sampled_q(rows.size() * rowbytes, device), reference(rows.size() * rowbytes, device);
  std::vector<bf16> actual(rows.size() * rowbytes / 2), expected(actual.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    MX_CUDA(cudaMemcpy(sampled_q.as<bf16>() + i * c.hq * c.d, q.as<bf16>() + rows[i] * qs,
                       rowbytes, cudaMemcpyDeviceToDevice));
    MX_CUDA(cudaMemcpy(actual.data() + i * c.hq * c.d, output + rows[i] * os,
                       rowbytes, cudaMemcpyDeviceToHost));
  }
  AttnParams rp = p;
  rp.Sq = int(rows.size()); rp.q_stride = 0; rp.o_stride = 0;
  if (!c.causal) {
    attn_reference(sampled_q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), reference.as<bf16>(), rp, stream);
  } else {
    rp.Sq = 1;
    for (size_t i = 0; i < rows.size(); ++i) {
      rp.q_offset = int(std::min(int64_t(std::numeric_limits<int>::max()), int64_t(c.offset) + rows[i]));
      attn_reference(sampled_q.as<bf16>() + i * c.hq * c.d, k.as<bf16>(), v.as<bf16>(),
                      reference.as<bf16>() + i * c.hq * c.d, rp, stream);
    }
  }
  MX_CUDA(cudaStreamSynchronize(stream));
  MX_CUDA(cudaStreamDestroy(stream));
  MX_CUDA(cudaMemcpy(expected.data(), reference.get(), reference.size(), cudaMemcpyDeviceToHost));
  cpu_oracle(c, p, q, k, v, expected);
  compare_outputs(c, device, rows.size(), actual, expected);
  check_guards(out, output, c, os, rows);
}

void invalid_cases(int device) {
  DevBuf b(256, device);
  AttnParams good; good.Sq = 1; good.Skv = 1; good.Hq = 1; good.Hkv = 1; good.D = 64;
  for (int i = 0; i < 11; ++i) {
    AttnParams p = good;
    switch (i) {
      case 0: p.Sq = 0; break;
      case 1: p.Skv = -1; break;
      case 2: p.Hq = 3; p.Hkv = 2; break;
      case 3: p.D = 80; break;
      case 4: p.causal = 2; break;
      case 5: p.Hkv = 0; break;
      case 6: p.q_stride = 63; break;
      case 7: p.kv_stride = -1; break;
      case 8: p.o_stride = 63; break;
      case 9: p.scale = std::numeric_limits<float>::quiet_NaN(); break;
      case 10: p.kv_stride = std::numeric_limits<int64_t>::max(); break;
    }
    for (auto function : {flash_attn, attn_reference}) {
      bool rejected = false;
      try { function(b.as<bf16>(), b.as<bf16>(), b.as<bf16>(), b.as<bf16>(), p, nullptr); }
      catch (const std::runtime_error&) { rejected = true; }
      MX_CHECK(rejected, "invalid parameters %d were accepted", i);
    }
  }
  bool rejected = false;
  try { flash_attn(nullptr, b.as<bf16>(), b.as<bf16>(), b.as<bf16>(), good, nullptr); }
  catch (const std::runtime_error&) { rejected = true; }
  MX_CHECK(rejected, "null query was accepted");
  std::printf("GPU%d invalid input checks: 23 passed\n", device);
}

int correctness(int device, bool quick) {
  const Case cases[] = {
      {1, 1, 1, 1, 64}, {1, 17, 4, 1, 72, 1, 0, 5, 3.f},
      {17, 1, 3, 1, 128, 1, 2, 8, 2.f}, {17, 129, 4, 2, 64, 0, 0, 5, 2.f},
      {129, 17, 8, 2, 72, 1, 5, 8, 3.f}, {129, 129, 4, 1, 128, 0, 0, 5, 3.f},
      {1000, 1000, 2, 1, 64}, {1000, 1000, 2, 1, 72, 0, 0, 0, 3.f},
      {1000, 1117, 4, 1, 128, 1, 117, 5, 2.f}, {16, 16, 32, 8, 128, 1},
      {129, 193, 64, 8, 128, 1, 64, 8, 3.f}, {2048, 2048, 32, 8, 128, 1},
      {4096, 4225, 32, 8, 128, 0, 0, 0, 2.f}, {16384, 18384, 32, 32, 128},
      {4096, 4096, 32, 32, 128}, {65536, 65536, 32, 32, 128},
      {1024, 1024, 8, 8, 64}, {16384, 16384, 8, 8, 72},
      {17, 1000, 4, 1, 72, 0, 0, 6, 2.f, .25f},
      {17, 129, 2, 1, 128, 1, 1000, 0, 3.f},
      {129, 129, 2, 1, 128, 0, 0, 0, 2.f, -.125f},
      {17, 129, 2, 1, 128, 1, -17, 0, 2.f},
      {129, 129, 2, 1, 72, 1, -17, 5, 3.f},
      {17, 129, 4, 1, 64, 1, std::numeric_limits<int>::max()},
  };
  unsigned seed = 1234;
  int count = 0;
  for (const Case& c : cases) {
    if (quick && c.sq > 1000) continue;
    check_case(c, device, seed += 7);
    ++count;
  }
  invalid_cases(device);
  std::printf("GPU%d correctness: %d cases passed\n", device, count);
  return count;
}

double dense_gemm(int device) {
  constexpr int n = 4096, iterations = 4;
  DevBuf a(size_t(n) * n * 2, device), b(a.size(), device), c(size_t(n) * n * 4, device);
  fill(a, 911, 1.f); fill(b, 912, 1.f);
  cublasHandle_t handle;
  MX_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
  float alpha = 1.f, beta = 0.f;
  auto run = [&] {
    cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, a.get(), CUDA_R_16BF, n, b.get(), CUDA_R_16BF, n,
        &beta, c.get(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    MX_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed: %d", int(status));
  };
  run(); MX_CUDA(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  MX_CUDA(cudaEventCreate(&begin)); MX_CUDA(cudaEventCreate(&end));
  MX_CUDA(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) run();
  MX_CUDA(cudaEventRecord(end)); MX_CUDA(cudaEventSynchronize(end));
  float elapsed;
  MX_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
  MX_CUDA(cudaEventDestroy(begin)); MX_CUDA(cudaEventDestroy(end));
  MX_CHECK(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "cublasDestroy failed");
  double ms = elapsed / iterations, tflops = 2. * n * n * n / (ms * 1e9);
  std::printf("DENSE GPU%d cuBLAS bf16/fp32 GEMM M=N=K=%d %.3f ms %.3f TFLOPS (measured baseline, not hardware peak)\n",
              device, n, ms, tflops);
  return tflops;
}

void benchmark(int device, int length, int heads, double dense) {
  Case c{length, length, heads, heads, 128};
  AttnParams p = params(c);
  size_t bytes = size_t(length) * heads * 128 * 2;
  DevBuf q(bytes, device), k(bytes, device), v(bytes, device), o(bytes, device);
  fill(q, 711, 1.f); fill(k, 712, 1.f); fill(v, 713, 1.f);
  flash_attn(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), o.as<bf16>(), p, nullptr);
  MX_CUDA(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  MX_CUDA(cudaEventCreate(&begin)); MX_CUDA(cudaEventCreate(&end));
  // 每個形狀只量兩次，最長形狀量一次；整個 benchmark 避免拖超過 30 秒。
  int iterations = length < 32768 ? 2 : 1;
  MX_CUDA(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i)
    flash_attn(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), o.as<bf16>(), p, nullptr);
  MX_CUDA(cudaEventRecord(end)); MX_CUDA(cudaEventSynchronize(end));
  float elapsed;
  MX_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
  MX_CUDA(cudaEventDestroy(begin)); MX_CUDA(cudaEventDestroy(end));
  double ms = elapsed / iterations;
  double tflops = 4. * length * length * heads * 128 / (ms * 1e9);
  std::printf("BENCH GPU%d D=128 S=%d H=%d causal=0 %.3f ms %.3f TFLOPS %.1f%% measured dense GEMM (%d timed runs)\n",
              device, length, heads, ms, tflops, 100. * tflops / dense, iterations);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool bench_only = false, quick = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--bench") bench_only = true;
      else if (std::string(argv[i]) == "--quick") quick = true;
      else throw std::runtime_error("unknown attention_test option");
    }
    MX_CHECK(!bench_only || !quick, "choose --bench or --quick");
    int count;
    MX_CUDA(cudaGetDeviceCount(&count));
    MX_CHECK(count >= 2, "attention_test requires device 0 and device 1");
    int passed = 0;
    for (int device = 0; device < 2 && !bench_only; ++device) {
      MX_CUDA(cudaSetDevice(device));
      cudaDeviceProp prop;
      MX_CUDA(cudaGetDeviceProperties(&prop, device));
      std::printf("GPU%d: %s sm_%d%d\n", device, prop.name, prop.major, prop.minor);
      passed += correctness(device, quick);
    }
    auto begin = std::chrono::steady_clock::now();
    for (int device = 0; device < 2 && !quick; ++device) {
      MX_CUDA(cudaSetDevice(device));
      double dense = dense_gemm(device);
      for (int s : {4096, 16384, 32768}) benchmark(device, s, 32, dense);
      benchmark(device, 16384, 56, dense);
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::printf("benchmark total: %.3f s\nattention_test: PASS (%d correctness cases, %d invalid input checks)\n",
                seconds, passed, bench_only ? 0 : 46);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "attention_test: FAIL: %s\n", e.what());
    return 1;
  }
}
