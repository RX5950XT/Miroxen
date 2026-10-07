#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <bit>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/qwen_latent.h"
#include "core/tensor.h"

using namespace mx;

static void check(int64_t n) {
  std::mt19937 rng{unsigned(n)};
  std::uniform_real_distribution<float> values(-4.f, 4.f), scales(.25f, 4.f);
  std::vector<float> mean(64), scale(64), nchw(size_t(64) * n), nhwc(nchw.size());
  for (int c = 0; c < 64; ++c) mean[c] = values(rng), scale[c] = scales(rng);
  for (auto& v : nchw) v = values(rng);
  // 用不可合併的乘加參考值檢查 decode 算術順序。
  DevBuf dm(64 * sizeof(float), 0), ds(64 * sizeof(float), 0), dx(nchw.size() * sizeof(float), 0),
      dy(nchw.size() * sizeof(float), 0), dz(nchw.size() * sizeof(float), 0);
  MX_CUDA(cudaMemcpy(dm.get(), mean.data(), dm.size(), cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(ds.get(), scale.data(), ds.size(), cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(dx.get(), nchw.data(), dx.size(), cudaMemcpyHostToDevice));
  qwen_latent::encode(dx.as<float>(), dy.as<float>(), n, dm.as<float>(), ds.as<float>(), nullptr);
  MX_CUDA(cudaMemcpy(nhwc.data(), dy.get(), dy.size(), cudaMemcpyDeviceToHost));
  for (int64_t p = 0; p < n; ++p) for (int c = 0; c < 64; ++c) {
    volatile float delta = nchw[size_t(c) * n + p] - mean[c];
    volatile float expected = delta / scale[c];
    MX_CHECK(std::bit_cast<uint32_t>(nhwc[size_t(p) * 64 + c]) == std::bit_cast<uint32_t>(float(expected)),
             "encode bits mismatch: N=%lld p=%lld c=%d",
             (long long)n, (long long)p, c);
  }
  const auto encoded = nhwc;
  qwen_latent::decode(dy.as<float>(), dz.as<float>(), n, dm.as<float>(), ds.as<float>(), nullptr);
  MX_CUDA(cudaMemcpy(nhwc.data(), dz.get(), dz.size(), cudaMemcpyDeviceToHost));
  for (int c = 0; c < 64; ++c) for (int64_t p = 0; p < n; ++p) {
    volatile float product = encoded[size_t(p) * 64 + c] * scale[c];
    volatile float expected = product + mean[c];
    MX_CHECK(std::bit_cast<uint32_t>(nhwc[size_t(c) * n + p]) == std::bit_cast<uint32_t>(float(expected)),
             "decode bits mismatch: N=%lld p=%lld c=%d",
             (long long)n, (long long)p, c);
  }
  std::printf("N=%lld encode/decode CPU 逐位元一致\n", (long long)n);
}

static void benchmark(int64_t n) {
  constexpr int kC = 64, kRuns = 20;
  const size_t bytes = size_t(n) * kC * sizeof(float);
  std::mt19937 rng{unsigned(n)};
  std::uniform_real_distribution<float> values(-4.f, 4.f), scales(.25f, 4.f);
  std::vector<float> mean(kC), scale(kC), src(size_t(n) * kC), host_in(src.size()), host_out(src.size());
  for (int c = 0; c < kC; ++c) mean[c] = values(rng), scale[c] = scales(rng);
  for (auto& v : src) v = values(rng);
  DevBuf dm(kC * sizeof(float), 0), ds(kC * sizeof(float), 0), dx(bytes, 0), old_out(bytes, 0), new_out(bytes, 0);
  MX_CUDA(cudaMemcpy(dm.get(), mean.data(), dm.size(), cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(ds.get(), scale.data(), ds.size(), cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(dx.get(), src.data(), bytes, cudaMemcpyHostToDevice));
  auto old_path = [&] {
    MX_CUDA(cudaMemcpy(host_in.data(), dx.get(), bytes, cudaMemcpyDeviceToHost));
    for (int c = 0; c < kC; ++c) for (int64_t p = 0; p < n; ++p) {
      volatile float product = host_in[size_t(p) * kC + c] * scale[c];
      host_out[size_t(c) * n + p] = product + mean[c];
    }
    MX_CUDA(cudaMemcpy(old_out.get(), host_out.data(), bytes, cudaMemcpyHostToDevice));
    MX_CUDA(cudaDeviceSynchronize());
  };
  auto new_path = [&] {
    qwen_latent::decode(dx.as<float>(), new_out.as<float>(), n, dm.as<float>(), ds.as<float>(), nullptr);
    MX_CUDA(cudaDeviceSynchronize());
  };
  std::vector<float> old_check(src.size()), new_check(src.size());
  old_path(); new_path();  // 暖機
  std::vector<double> old_ms, new_ms;
  old_ms.reserve(kRuns); new_ms.reserve(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    auto time = [](auto&& fn) {
      auto start = std::chrono::steady_clock::now(); fn();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    if (i & 1) {
      new_ms.push_back(time(new_path)); old_ms.push_back(time(old_path));
    } else {
      old_ms.push_back(time(old_path)); new_ms.push_back(time(new_path));
    }
  }
  std::sort(old_ms.begin(), old_ms.end()); std::sort(new_ms.begin(), new_ms.end());
  MX_CUDA(cudaMemcpy(old_check.data(), old_out.get(), bytes, cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(new_check.data(), new_out.get(), bytes, cudaMemcpyDeviceToHost));
  MX_CHECK(std::memcmp(old_check.data(), new_check.data(), bytes) == 0,
           "Old/new decode benchmark outputs differ for N=%lld", (long long)n);
  std::printf("benchmark N=%lld: old %.4f ms, GPU %.4f ms, host transfer saved %zu bytes/run (outputs bitwise equal)\n",
              (long long)n, (old_ms[kRuns / 2 - 1] + old_ms[kRuns / 2]) / 2,
              (new_ms[kRuns / 2 - 1] + new_ms[kRuns / 2]) / 2, 2 * bytes);
}

int main(int argc, char** argv) {
  try {
    MX_CHECK(argc <= 2 && (argc == 1 || std::strcmp(argv[1], "--benchmark") == 0),
             "Usage: qwen_latent_test [--benchmark]");
    check(37);
    check(1031);
    if (argc == 2) {
      benchmark(3772);   // 736x1312 / 256
      benchmark(16384);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
