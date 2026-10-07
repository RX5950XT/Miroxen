// 與官方 diffusers BF16 VAE 比對；參考資料由 tools/ref/qwen_vae_ref.py 產生。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "models/qwen_image/vae.h"

using namespace mx;
namespace {

const WeightRef& reference(const TensorStore& store, const char* name, GType type,
                           std::vector<int64_t> shape) {
  const auto& r = store.get(name);
  MX_CHECK(r.type == type && r.shape == shape, "參考張量 %s 的型別或大小不符", name);
  MX_CHECK(r.data && r.nbytes == size_t(r.numel()) * gtype_block_bytes(type),
           "參考張量 %s 的內容長度不符", name);
  if (type == GType::F32) {
    const auto* p = static_cast<const float*>(r.data);
    for (int64_t i = 0; i < r.numel(); ++i)
      MX_CHECK(std::isfinite(p[i]), "參考張量 %s 含非有限值", name);
  }
  return r;
}

template<class F> void rejects(const char* name, F fn) {
  bool threw = false;
  try { fn(); }
  catch (const std::exception& e) { threw = e.what() && e.what()[0]; }
  MX_CHECK(threw, "%s 沒有拒絕無效輸入", name);
}

void invalid_inputs(QwenImageVae& vae, Ctx& c, const float* z, float* out,
                    const uint8_t* image) {
  std::vector<uint8_t> rgba;
  rejects("decode null", [&] { vae.decode(c, nullptr, 32, 40, rgba); });
  rejects("decode bad shape", [&] { vae.decode(c, z, 0, 40, rgba); });
  rejects("encode null image", [&] { vae.encode(c, nullptr, 256, 320, out); });
  rejects("encode null output", [&] { vae.encode(c, image, 256, 320, nullptr); });
  rejects("encode bad shape", [&] { vae.encode(c, image, 255, 320, out); });
  std::printf("無效輸入：5/5 PASS\n");
}

void benchmark(QwenImageVae& vae) {
  constexpr int64_t H = 128, W = 128;
  const size_t estimate = vae.decode_act_bytes(H, W);
  constexpr size_t limit = size_t(3584) << 20;
  MX_CHECK(estimate > 0 && estimate <= limit,
           "2048x2048 activation 估計 %.3f GiB 超過 3.5 GiB", estimate / 1073741824.0);
  Ctx c(0, estimate, 0);
  std::mt19937 rng(2026);
  std::normal_distribution<float> normal;
  std::vector<float> host(64 * H * W);
  for (int64_t ch = 0; ch < 64; ++ch)
    for (int64_t i = 0; i < H * W; ++i)
      host[ch * H * W + i] = normal(rng) * vae.latents_std()[ch] + vae.latents_mean()[ch];
  DevBuf z(host.size() * sizeof(float), 0);
  MX_CUDA(cudaMemcpy(z.get(), host.data(), z.size(), cudaMemcpyHostToDevice));
  c.sync();
  std::vector<uint8_t> rgba;
  // 第一次含 cuBLAS 初始化；GPU 可能與其他工作共用，取三次中位數。
  double times[3];
  for (double& ms : times) {
    const auto start = std::chrono::steady_clock::now();
    vae.decode(c, z.as<float>(), H, W, rgba);
    c.sync();
    ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
  std::printf("GPU0 decode 2048x2048（含主機輸出）：%.3f / %.3f / %.3f ms；",
              times[0], times[1], times[2]);
  std::sort(times, times + 3);
  std::printf("中位數 %.3f ms；arena peak %.3f GiB；decode_act_bytes %.3f GiB\n", times[1],
              c.act.peak() / 1073741824.0, estimate / 1073741824.0);
  MX_CHECK(rgba.size() == size_t(2048) * 2048 * 4, "2048x2048 RGBA 大小不符");
  MX_CHECK(c.act.peak() <= estimate, "activation 實際用量超過 decode_act_bytes");
  MX_CHECK(c.act.used() == 0, "decode 沒有釋放 Arena 暫存");
}

void parity(const char* layout, QwenImageVae& vae, const TensorStore& refs) {
  const auto& latent = reference(refs, "decode_z", GType::F32, {64, 32, 40});
  const auto& expected = reference(refs, "decode_rgba", GType::U8, {512, 640, 4});
  const auto& image = reference(refs, "encode_rgba", GType::U8, {256, 320, 4});
  const auto& mean = reference(refs, "encode_mean", GType::F32, {64, 16, 20});
  const size_t estimate = vae.decode_act_bytes(32, 40);
  Ctx c(0, estimate, 0);
  DevBuf z(latent.nbytes, 0), out(mean.nbytes, 0);
  MX_CUDA(cudaMemcpy(z.get(), latent.data, latent.nbytes, cudaMemcpyHostToDevice));
  invalid_inputs(vae, c, z.as<float>(), out.as<float>(), static_cast<const uint8_t*>(image.data));
  std::vector<uint8_t> rgba;
  vae.decode(c, z.as<float>(), 32, 40, rgba);
  MX_CHECK(rgba.size() == expected.nbytes, "decode RGBA 大小不符");
  double squared = 0;
  int maxabs = 0;
  const auto* pixels = static_cast<const uint8_t*>(expected.data);
  for (size_t i = 0; i < rgba.size(); ++i) {
    const int d = int(rgba[i]) - int(pixels[i]);
    squared += double(d) * d;
    maxabs = std::max(maxabs, std::abs(d));
  }
  const double psnr = squared == 0 ? std::numeric_limits<double>::infinity()
      : 10 * std::log10(255.0 * 255 * rgba.size() / squared);
  std::printf("%s decode：PSNR %.4f dB；max abs %d → %s\n",
              layout, psnr, maxabs, psnr >= 40 ? "PASS" : "FAIL");
  vae.encode(c, static_cast<const uint8_t*>(image.data), 256, 320, out.as<float>());
  c.sync();
  std::vector<float> got(mean.numel());
  MX_CUDA(cudaMemcpy(got.data(), out.get(), mean.nbytes, cudaMemcpyDeviceToHost));
  const auto* target = static_cast<const float*>(mean.data);
  double num = 0, den = 0, maxdiff = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    MX_CHECK(std::isfinite(got[i]), "encode 輸出含非有限值");
    const double d = double(got[i]) - target[i];
    num += d * d;
    den += double(target[i]) * target[i];
    maxdiff = std::max(maxdiff, std::fabs(d));
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf("%s encode：relative L2 %.6f；max abs %.6f → %s\n",
              layout, rel, maxdiff, rel < .02 ? "PASS" : "FAIL");
  std::printf("%s parity arena peak %.3f GiB；estimate %.3f GiB\n",
              layout, c.act.peak() / 1073741824.0, estimate / 1073741824.0);
  MX_CHECK(psnr >= 40 && rel < .02, "%s VAE 未達數值比對標準", layout);
  MX_CHECK(c.act.peak() <= estimate && c.act.used() == 0, "parity Arena 用量或釋放不符");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    MX_CHECK(argc <= 2, "用法：qwen_vae_test [models目錄]");
    const std::filesystem::path root = argc == 2 ? argv[1]
        : "D:/Workspace/Personal_Project/Miroxen/models";
    const auto ref = root / "ref/qwen_vae_ref.safetensors";
    if (!std::filesystem::exists(ref)) {
      std::printf("SKIP：缺少參考資料 %s\n", ref.string().c_str());
      return 0;
    }
    TensorStore refs;
    refs.add_file(ref.string(), "\x01");
    const char* names[] = {"Comfy", "diffusers"};
    const std::filesystem::path paths[] = {
        root / "image/qwen-image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors",
        root / "image/qwen-image-2.1/hf/vae/diffusion_pytorch_model.safetensors"};
    int passed = 0, failed = 0, skipped = 0;
    bool timed = false;
    for (int i = 0; i < 2; ++i) {
      if (!std::filesystem::exists(paths[i])) {
        std::printf("SKIP：缺少 %s 權重 %s\n", names[i], paths[i].string().c_str());
        ++skipped;
        continue;
      }
      try {
        TensorStore weights;
        weights.add_file(paths[i].string());
        QwenImageVae vae(weights, 0);
        parity(names[i], vae, refs);
        if (!timed) { benchmark(vae); timed = true; }
        ++passed;
      } catch (const std::exception& e) {
        std::printf("%s FAIL：%s\n", names[i], e.what());
        ++failed;
      }
    }
    std::printf("VAE layouts：%d PASS，%d FAIL，%d SKIP\n", passed, failed, skipped);
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
