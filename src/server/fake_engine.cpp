#include "server/fake_engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <thread>

#include "core/common.h"
#include "server/image_io.h"

namespace mx {
namespace {

constexpr int kStepMs = 25;

int64_t pick_seed(int64_t seed) {
  if (seed != -1) return seed;
  std::random_device rd;
  std::mt19937_64 gen((static_cast<uint64_t>(rd()) << 32) ^
                      static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::uniform_int_distribution<int64_t> dist(0, 0x7fffffffffffffffLL);
  return dist(gen);
}

void tick(const ProgressFn& cb, const std::atomic<bool>* cancel, const char* stage, int step, int total) {
  if (cancel && cancel->load()) throw std::runtime_error("已取消");
  if (cb) cb(Progress{stage, step, total});
  std::this_thread::sleep_for(std::chrono::milliseconds(kStepMs));
  if (cancel && cancel->load()) throw std::runtime_error("已取消");
}

void simulate(const ProgressFn& cb, const std::atomic<bool>* cancel, int steps) {
  int n = steps < 1 ? 1 : steps;
  tick(cb, cancel, "encode", 0, n);
  for (int i = 1; i <= n; ++i) tick(cb, cancel, "denoise", i, n);
  tick(cb, cancel, "decode", n, n);
}

bool ffmpeg_available() {
#ifdef _WIN32
  int rc = _wsystem(L"ffmpeg -version >NUL 2>&1");
#else
  int rc = std::system("ffmpeg -version >/dev/null 2>&1");
#endif
  return rc == 0;
}

}  // namespace

std::vector<ModelInfo> FakeEngine::models() {
  return {
      {"qwen-image-2.1", "image", true, "fake"},
      {"minimax-h3", "video", true, "fake"},
  };
}

std::vector<GpuInfo> FakeEngine::gpus() {
  std::vector<GpuInfo> out;
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
    out.push_back({0, "fake", 0, 0});
    return out;
  }
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(i);
    cudaMemGetInfo(&free_b, &total_b);
    out.push_back({i, prop.name, total_b, free_b});
  }
  if (out.empty()) out.push_back({0, "fake", 0, 0});
  return out;
}

ImageResult FakeEngine::generate_image(const ImageRequest& r, const ProgressFn& cb,
                                       const std::atomic<bool>* cancel) {
  MX_CHECK(r.width > 0 && r.height > 0 && r.width <= 8192 && r.height <= 8192, "圖片尺寸不合法");
  auto t0 = std::chrono::steady_clock::now();
  simulate(cb, cancel, r.steps);
  int64_t seed = pick_seed(r.seed);
  std::vector<uint8_t> rgb(static_cast<size_t>(r.width) * static_cast<size_t>(r.height) * 3);
  int denom_x = r.width > 1 ? r.width - 1 : 1;
  int denom_y = r.height > 1 ? r.height - 1 : 1;
  for (int y = 0; y < r.height; ++y) {
    for (int x = 0; x < r.width; ++x) {
      uint8_t* p = rgb.data() + (static_cast<size_t>(y) * r.width + x) * 3;
      p[0] = static_cast<uint8_t>(x * 255 / denom_x);
      p[1] = static_cast<uint8_t>(y * 255 / denom_y);
      p[2] = static_cast<uint8_t>((seed + x * 13 + y * 7) & 255);
    }
  }
  ImageResult result;
  result.png = encode_rgb_png(rgb.data(), r.width, r.height);
  result.width = r.width;
  result.height = r.height;
  result.seed = seed;
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

VideoResult FakeEngine::generate_video(const VideoRequest& r, const ProgressFn& cb,
                                       const std::atomic<bool>* cancel) {
  MX_CHECK(r.width > 0 && r.height > 0 && r.width <= 8192 && r.height <= 8192, "影片尺寸不合法");
  auto t0 = std::chrono::steady_clock::now();
  simulate(cb, cancel, r.steps);
  if (!ffmpeg_available()) {
    throw std::runtime_error("找不到 ffmpeg（PATH 中沒有 ffmpeg），無法輸出影片");
  }
  auto path = std::filesystem::temp_directory_path() /
              ("miroxen-" + std::to_string(
                                std::chrono::steady_clock::now().time_since_epoch().count()) +
               ".mp4");
  std::wstring wpath = path.wstring();
#ifdef _WIN32
  std::wstring cmd = L"ffmpeg -y -loglevel error -f lavfi -i color=c=0x24483c:s=" +
                     std::to_wstring(r.width) + L"x" + std::to_wstring(r.height) +
                     L":r=8:d=0.4 -f lavfi -i anullsrc=r=44100:cl=stereo -shortest "
                     L"-c:v libx264 -pix_fmt yuv420p -c:a aac \"" +
                     wpath + L"\" >NUL 2>&1";
  int rc = _wsystem(cmd.c_str());
#else
  std::string cmd = "ffmpeg -y -loglevel error -f lavfi -i color=c=0x24483c:s=" +
                    std::to_string(r.width) + "x" + std::to_string(r.height) +
                    ":r=8:d=0.4 -f lavfi -i anullsrc=r=44100:cl=stereo -shortest "
                    "-c:v libx264 -pix_fmt yuv420p -c:a aac \"" +
                    path.string() + "\" >/dev/null 2>&1";
  int rc = std::system(cmd.c_str());
#endif
  MX_CHECK(rc == 0 && std::filesystem::is_regular_file(path), "ffmpeg 輸出影片失敗（結束碼 %d）", rc);
  VideoResult result;
  result.mp4_path = path.string();
  result.seed = pick_seed(r.seed);
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

}  // namespace mx
