// 與官方 diffusers 比對兩種權重格式；參考產生器見 tools/ref/h3_video_vae_ref.py。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/dequant.h"
#include "models/minimax_h3/video_vae.h"

using namespace mx;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void env(const char* name, const std::string& value) {
#ifdef _WIN32
  MX_CHECK(_putenv_s(name, value.c_str()) == 0, "無法設定 %s", name);
#else
  MX_CHECK(setenv(name, value.c_str(), 1) == 0, "無法設定 %s", name);
#endif
}

static fs::path model_root() {
  if (const char* p = std::getenv("MX_MODEL_ROOT")) return p;
  if (fs::exists("models/minimax-h3")) return "models";
  return "../../Miroxen/models";
}

static std::vector<float> host_f32(const WeightRef& r) {
  MX_CHECK(r.type == GType::F32 || r.type == GType::F16 || r.type == GType::BF16,
           "%s 必須是浮點張量", r.name.c_str());
  std::vector<float> result(size_t(r.numel()));
  dequant_cpu(r.data, r.type, r.numel(), result.data());
  for (float v : result) MX_CHECK(std::isfinite(v), "%s 含非法數值", r.name.c_str());
  return result;
}

static double psnr(const std::vector<uint8_t>& got, const WeightRef& expected) {
  MX_CHECK(expected.type == GType::U8 && expected.nbytes == got.size(), "RGB 輸出長度或型別不符");
  const auto* ref = static_cast<const uint8_t*>(expected.data);
  double squared = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double d = double(got[i]) - ref[i];
    squared += d * d;
  }
  return squared == 0 ? INFINITY : 10 * std::log10(255.0 * 255 * got.size() / squared);
}

static double relative_error(const std::vector<float>& got, const std::vector<float>& ref) {
  MX_CHECK(got.size() == ref.size(), "編碼輸出長度不符");
  double squared = 0, norm = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    MX_CHECK(std::isfinite(got[i]), "編碼輸出含非法數值");
    double d = double(got[i]) - ref[i];
    squared += d * d;
    norm += double(ref[i]) * ref[i];
  }
  return std::sqrt(squared / std::max(norm, 1e-30));
}

static void benchmark(H3VideoVae& vae, const char* layout) {
  constexpr int64_t F = 37, H = 30, W = 54;  // 124 張、864x480、25 fps 約 5 秒。
  Ctx c(0, vae.decode_act_bytes(H, W), 0);
  DevBuf latent(size_t(24 * F * H * W) * sizeof(float), 0);
  MX_CUDA(cudaMemsetAsync(latent.get(), 0, latent.size(), c.stream));
  c.sync();
  int frames = 0, chunks = 0;
  auto start = Clock::now(), previous = start;
  vae.decode(c, latent.as<float>(), F, H, W, [&](const uint8_t* rgb, int count) {
    MX_CHECK(rgb && count > 0, "benchmark sink 收到空影格");
    c.sync();
    auto now = Clock::now();
    const bool tail=count==5;
    if(!tail) ++chunks;
    std::printf("benchmark %s %s %d: %d frames, %.2f ms\n", layout,tail?"tail":"chunk",chunks,count,
                std::chrono::duration<double, std::milli>(now - previous).count());
    frames += count;
    previous = now;
  });
  c.sync();
  MX_CHECK(frames == 124, "benchmark 預期 124 張，實際 %d", frames);
  std::printf("benchmark %s total: %.2f ms, %d chunks, %d frames, arena %.2f MiB\n", layout,
              std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
              chunks, frames, c.act.peak() / 1048576.0);
}

static int compare(const TensorStore& weights, const TensorStore& ref, const char* layout, bool bench) {
  const auto& latent_ref = ref.get("normalized_latent");
  const auto& frames_ref = ref.get("decoded_rgb");
  const auto& rgb_ref = ref.get("encoder_rgb");
  const auto& encoded_ref = ref.get("encoded_latent");
  MX_CHECK(latent_ref.shape == std::vector<int64_t>({24, 12, 16, 28}), "參考 latent shape 不符");
  MX_CHECK(frames_ref.shape == std::vector<int64_t>({39, 256, 448, 3}), "參考影片 shape 不符");
  MX_CHECK(rgb_ref.type == GType::U8 && rgb_ref.shape == std::vector<int64_t>({256, 448, 3}),
           "參考單張圖型別或 shape 不符");
  MX_CHECK(encoded_ref.shape == std::vector<int64_t>({24, 1, 16, 28}), "參考編碼 shape 不符");
  auto input = host_f32(latent_ref), expected_latent = host_f32(encoded_ref);
  // MX_TEST_VAE_COMPUTE=fp8 時用 FP8 解碼器（量畫質損失）。
  const char* qc = std::getenv("MX_TEST_VAE_COMPUTE");
  H3VideoVae vae(weights, 0, qc ? parse_qmode(qc) : QMode::BF16);
  Ctx c(0, vae.decode_act_bytes(16, 28), 0);
  DevBuf z(input.size() * sizeof(float), 0), encoded(expected_latent.size() * sizeof(float), 0);
  MX_CUDA(cudaMemcpyAsync(z.get(), input.data(), z.size(), cudaMemcpyHostToDevice, c.stream));
  std::vector<uint8_t> frames;
  frames.reserve(frames_ref.nbytes);
  int chunks = 0, count = 0;
  vae.decode(c, z.as<float>(), 12, 16, 28, [&](const uint8_t* rgb, int num_frames) {
    MX_CHECK(rgb && num_frames > 0 && count + num_frames <= 39, "sink 影格數或指標不符");
    size_t bytes = size_t(num_frames) * 256 * 448 * 3;
    frames.insert(frames.end(), rgb, rgb + bytes);
    count += num_frames;
    ++chunks;
  });
  c.sync();
  MX_CHECK(count == 39 && chunks >= 2, "必須輸出 39 張且分至少兩段：%d 張、%d 段", count, chunks);
  double score = psnr(frames, frames_ref);
  Ctx encoder_ctx(0, vae.decode_act_bytes(16, 28), 0);
  vae.encode_image(encoder_ctx, static_cast<const uint8_t*>(rgb_ref.data), 256, 448, encoded.as<float>());
  encoder_ctx.sync();
  std::vector<float> got(expected_latent.size());
  MX_CUDA(cudaMemcpy(got.data(), encoded.get(), encoded.size(), cudaMemcpyDeviceToHost));
  double error = relative_error(got, expected_latent);
  bool decode_ok = score >= 38, encode_ok = error < 0.03;
  std::printf("%s decode: PSNR %.4f dB >= 38, %d frames/%d chunks -> %s\n", layout,
              score, count, chunks, decode_ok ? "PASS" : "FAIL");
  std::printf("%s encode (posterior mean): relative error %.6f < 0.03 -> %s\n", layout,
              error, encode_ok ? "PASS" : "FAIL");
  std::printf("%s decode arena %.2f MiB / budget %.2f MiB; encoder arena %.2f MiB\n", layout,
              c.act.peak() / 1048576.0, vae.decode_act_bytes(16, 28) / 1048576.0,
              encoder_ctx.act.peak() / 1048576.0);
  if (bench && (qc || (decode_ok && encode_ok))) benchmark(vae, layout);
  int video_fail = 0;
  const char* vref = "models/ref/h3_video_encode_ref.safetensors";  // tools/ref/h3_video_encode_ref.py
  if (std::filesystem::exists(vref)) {
    TensorStore vr;
    vr.add_file(vref, "\x01");
    const auto& vrgb = vr.get("rgb");
    auto vexpect = host_f32(vr.get("encoded"));
    MX_CHECK(vrgb.shape == std::vector<int64_t>({22, 256, 448, 3}), "參考影片 shape 不符");
    MX_CHECK(H3VideoVae::encoded_frames(22) == 7, "encoded_frames(22) 應為 7");
    Ctx vc(0, size_t(6) << 30, 0);
    DevBuf vz(vexpect.size() * sizeof(float), 0);
    vae.encode_video(vc, static_cast<const uint8_t*>(vrgb.data), 22, 256, 448, vz.as<float>());  // 暖機
    auto ve0 = Clock::now();
    vae.encode_video(vc, static_cast<const uint8_t*>(vrgb.data), 22, 256, 448, vz.as<float>());
    std::printf("%s encode_video 22 幀 256×448：%.0f ms\n", layout,
                std::chrono::duration<double, std::milli>(Clock::now() - ve0).count());
    std::vector<float> vgot(vexpect.size());
    MX_CUDA(cudaMemcpy(vgot.data(), vz.get(), vz.size(), cudaMemcpyDeviceToHost));
    double ve = relative_error(vgot, vexpect);
    video_fail = ve >= 0.03;
    std::printf("%s encode_video（22 幀→7 latent 幀）：relative error %.6f < 0.03 -> %s，arena %.0f MiB\n", layout, ve,
                video_fail ? "FAIL" : "PASS", vc.act.peak() / 1048576.0);
  }
  return int(!decode_ok) + int(!encode_ok) + video_fail;
}

int main(int argc, char** argv) {
  try {
    bool bench = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--benchmark") == 0) bench = true;
      else if (std::strcmp(argv[i], "--layout") == 0 && i + 1 < argc) only = argv[++i];
      else throw std::runtime_error("用法：h3_video_vae_test [--benchmark] [--layout unsloth|diffusers]");
    }
    MX_CHECK(only.empty() || only == "unsloth" || only == "diffusers", "權重 layout 不合法");
    env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    const fs::path root = model_root(), folder = root / "minimax-h3/hf/vae";
    const fs::path reference = root / "ref/h3_video_vae_ref.safetensors";
    if (!fs::exists(reference) || !fs::exists(folder / "config.json")) {
      std::printf("SKIP: 缺少 reference 或 config.json（%s）\n", root.string().c_str());
      return 0;
    }
    env("MX_H3_VIDEO_VAE_CONFIG", fs::absolute(folder / "config.json").string());
    TensorStore ref;
    ref.add_file(reference.string(), "\x01");
    int failures = 0, layouts = 0;
    for (const char* layout : {"unsloth", "diffusers"}) {
      if (!only.empty() && only != layout) continue;
      std::vector<fs::path> files;
      if (std::strcmp(layout, "unsloth") == 0)
        files.push_back(root / "minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors");
      else
        for (int i = 1; i <= 3; ++i)
          files.push_back(folder / strf("diffusion_pytorch_model-%05d-of-00003.safetensors", i));
      bool complete = std::all_of(files.begin(), files.end(), [](const auto& p) { return fs::exists(p); });
      if (!complete) {
        std::printf("SKIP %s: 缺少權重檔\n", layout);
        continue;
      }
      TensorStore weights;
      for (const auto& p : files) weights.add_file(p.string());
      failures += compare(weights, ref, layout, bench);
      ++layouts;
    }
    std::printf("summary: %d layouts, %d checks passed, %d failed\n", layouts, layouts * 2 - failures, failures);
    return failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
}
