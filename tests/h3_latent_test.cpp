#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/ctx.h"
#include "kernels/h3_latent.h"

using namespace mx;

static void check(int64_t F, int64_t H, int64_t W, int64_t A, int device, bool benchmark) {
  constexpr int offset = 3 * 96;
  const size_t nv = size_t(24 * F * H * W), na = size_t(64 * A);
  std::vector<float> rows(offset + nv), audio(offset + na), cpu(nv + na), gpu(cpu.size());
  for (size_t i = 0; i < rows.size(); ++i) rows[i] = float(int(i % 10007) - 5003) / 16;
  for (size_t i = 0; i < audio.size(); ++i) audio[i] = float(int(i % 1009) - 503) / 8;
  DevBuf vr(rows.size() * 4, 0), ar(audio.size() * 4, 0);
  MX_CUDA(cudaMemcpy(vr.get(), rows.data(), vr.size(), cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(ar.get(), audio.data(), ar.size(), cudaMemcpyHostToDevice));
  Ctx c(device, 0, 0);
  DevBuf dst(cpu.size() * 4, device), transfer((nv + na) * 4, device);
  auto old_path = [&] {
    MX_CUDA(cudaMemcpy(rows.data(), vr.get(), vr.size(), cudaMemcpyDeviceToHost));
    MX_CUDA(cudaMemcpy(audio.data(), ar.get(), ar.size(), cudaMemcpyDeviceToHost));
    for (int64_t f = 0; f < F; ++f) for (int64_t y = 0; y < H / 2; ++y)
      for (int64_t x = 0; x < W / 2; ++x) for (int ch = 0; ch < 24; ++ch)
        for (int ph = 0; ph < 2; ++ph) for (int pw = 0; pw < 2; ++pw) {
          int64_t row = (f * (H / 2) + y) * (W / 2) + x;
          cpu[((ch * F + f) * H + 2 * y + ph) * W + 2 * x + pw] = rows[offset + row * 96 + ch * 4 + ph * 2 + pw];
        }
    for (int ch = 0; ch < 2; ++ch) for (int64_t a = 0; a < A; ++a) for (int k = 0; k < 32; ++k)
      cpu[nv + (ch * 32 + k) * A + a] = audio[offset + (ch * A + a) * 32 + k];
    MX_CUDA(cudaMemcpy(dst.get(), cpu.data(), dst.size(), cudaMemcpyHostToDevice));
    c.sync();
  };
  auto new_path = [&] {
    const float* v = vr.as<float>() + offset;
    const float* a = ar.as<float>() + offset;
    if (device != 0) {
      MX_CUDA(cudaMemcpyPeer(transfer.get(), device, v, 0, nv * 4));
      MX_CUDA(cudaMemcpyPeer(transfer.as<float>() + nv, device, a, 0, na * 4));
      v = transfer.as<float>(); a = v + nv;
    }
    h3_latent::video(v, dst.as<float>(), F, H, W, c.stream);
    h3_latent::audio(a, dst.as<float>() + nv, A, c.stream);
    c.sync();
  };
  old_path(); new_path();
  MX_CUDA(cudaMemcpy(gpu.data(), dst.get(), dst.size(), cudaMemcpyDeviceToHost));
  MX_CHECK(std::memcmp(cpu.data(), gpu.data(), dst.size()) == 0, "H3 latent CPU/GPU bits differ");
  std::printf("F=%lld H=%lld W=%lld A=%lld GPU%d: CPU/GPU bitwise equal\n", F, H, W, A, device);
  if (!benchmark) return;
  std::vector<double> old_ms, new_ms;
  auto time = [](auto&& fn) { auto start = std::chrono::steady_clock::now(); fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); };
  for (int i = 0; i < 20; ++i) {
    if (i & 1) { new_ms.push_back(time(new_path)); old_ms.push_back(time(old_path)); }
    else { old_ms.push_back(time(old_path)); new_ms.push_back(time(new_path)); }
  }
  std::sort(old_ms.begin(), old_ms.end()); std::sort(new_ms.begin(), new_ms.end());
  std::printf("layout benchmark: CPU roundtrip %.3f ms -> GPU %.3f ms; saved %zu host bytes/run\n",
              (old_ms[9] + old_ms[10]) / 2, (new_ms[9] + new_ms[10]) / 2, (nv + na) * 8);
}

int main(int argc, char** argv) {
  try {
    MX_CHECK(argc <= 2 && (argc == 1 || std::strcmp(argv[1], "--benchmark") == 0), "Usage: h3_latent_test [--benchmark]");
    MX_CUDA(cudaSetDevice(0));
    check(2, 6, 10, 37, 0, false);
    int count = 0; MX_CUDA(cudaGetDeviceCount(&count));
    if (count > 1) check(2, 6, 10, 37, 1, false);
    bool rejected = false;
    try { h3_latent::video(nullptr, nullptr, 1, 3, 2, nullptr); } catch (const std::exception&) { rejected = true; }
    MX_CHECK(rejected, "Invalid H3 latent input accepted");
    if (argc == 2) { MX_CUDA(cudaSetDevice(0)); check(72, 80, 46, 405, 0, true); }
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
