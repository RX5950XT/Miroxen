#pragma once
// 引擎對外介面：HTTP 伺服器與 CLI 都只透過這裡呼叫推論。
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mx {

struct LoraSpec {
  std::string path;     // .safetensors
  float scale = 1.0f;
};

struct ImageRequest {
  std::string prompt;
  std::string negative_prompt;  // 只有 cfg_scale > 1 時使用
  int width = 1024, height = 1024;
  int steps = 40;
  float cfg_scale = 1.0f;
  int64_t seed = -1;            // -1 表示隨機
  std::vector<std::string> ref_images;  // 編輯用參考圖（檔案路徑，PNG/JPG）
  std::vector<LoraSpec> loras;
};

struct VideoRequest {
  std::string prompt;
  int width = 864, height = 480;
  int num_frames = 56;          // 會對齊到 17k+5
  int steps = 30;
  int64_t seed = -1;
  std::string first_frame, last_frame;  // 選填：首／尾幀圖片路徑
  std::vector<LoraSpec> loras;
};

struct Progress {
  std::string stage;  // "encode" / "denoise" / "decode" ...
  int step = 0, total = 0;
};
using ProgressFn = std::function<void(const Progress&)>;

struct ImageResult {
  std::vector<uint8_t> png;  // PNG 檔內容（可能含 alpha）
  int width = 0, height = 0;
  int64_t seed = 0;
  double seconds = 0;
};

struct VideoResult {
  std::string mp4_path;  // 輸出檔（含音軌）
  int64_t seed = 0;
  double seconds = 0;
};

struct ModelInfo {
  std::string id;      // "qwen-image-2.1" / "minimax-h3"
  std::string kind;    // "image" / "video"
  bool loaded = false;
  std::string detail;  // 量化型別、裝置配置等
};

struct GpuInfo {
  int index = 0;
  std::string name;
  size_t total = 0, free = 0;
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::vector<ModelInfo> models() = 0;
  virtual std::vector<GpuInfo> gpus() = 0;
  // cancel 由呼叫端設成 true 時，引擎在下一步之間中止並丟出例外。
  virtual ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                                     const std::atomic<bool>* cancel) = 0;
  virtual VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                                     const std::atomic<bool>* cancel) = 0;
};

}  // namespace mx
