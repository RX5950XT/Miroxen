#pragma once
// 引擎對外介面：HTTP 伺服器與 CLI 都只透過這裡呼叫推論。
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
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
  int width = 1024, height = 1024;  // 有參考圖時 0 = 依最後一張參考圖的比例（約 1024² 像素）
  int steps = 40;
  float cfg_scale = 1.0f;
  int64_t seed = -1;            // -1 表示隨機
  std::vector<std::string> ref_images;  // 編輯用參考圖（檔案路徑，PNG/JPG）
  std::vector<LoraSpec> loras;
  std::string compute;          // DiT 運算精度 "bf16"/"fp8"/"fp4"；空 = 引擎預設
  bool shift_terminal = true;   // 排程尾端拉伸到 0.02（官方預設）；步數蒸餾 LoRA（如 Turbo8）要關掉
  std::vector<float> sigmas;    // 選填：自訂未位移的 sigma 節點（遞減、首項 1），步數 = 節點數
  std::map<std::string, std::string> model_files;  // 已安裝的模型檔 ID，由 /v1/models 提供
};

// MiniMax-H3 Ref2VA 參考（依提示詞裡 <Picture i>／<Video k>／<Audio j> 的順序）。
struct MediaRef {
  std::string kind;  // "image"（≤9）／"video"（≤3，自帶音軌也會一起參考）／"audio"（≤3，需搭配圖或影片）
  std::string path;
};

struct VideoRequest {
  std::string prompt;
  int width = 864, height = 480;
  int num_frames = 56;          // 會對齊到 17k+5
  int steps = 30;               // 取樣點數；實際前向次數 = steps − 1（8 步加速 LoRA 用 9）
  int64_t seed = -1;
  std::string first_frame, last_frame;  // 選填：首／尾幀圖片路徑
  float video_shift = 12.f;     // 影片排程 shift（官方 12；LightX2V 768p 加速 LoRA 建議 6）
  std::vector<MediaRef> references;  // 非空 = Ref2VA 參考生成（換用 Ref2VA checkpoint，不可同時給首尾幀）
  int ref_image_short_edge = 2048;   // 參考圖縮放到的短邊（官方 2048；調小可省時間與顯存）
  bool ref_native_canvas = false;    // 參考影片畫布：false = 不超過輸出畫布（快，同 LightX2V match）；true = 官方 768p
  std::vector<LoraSpec> loras;
  std::string compute;          // 同 ImageRequest::compute
  std::map<std::string, std::string> model_files;
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
  std::map<std::string, double> timings;  // 各階段牆鐘秒數，不含首次模型載入
};

struct VideoResult {
  std::string mp4_path;  // 輸出檔（含音軌）
  int64_t seed = 0;
  double seconds = 0;
  std::map<std::string, double> timings;  // 各階段牆鐘秒數，不含首次模型載入
};

struct ModelInfo {
  std::string id;      // "qwen-image-2.1" / "minimax-h3"
  std::string kind;    // "image" / "video"
  bool loaded = false;
  std::string detail;  // 量化型別、裝置配置等
  std::map<std::string, std::vector<std::string>> files;
  std::map<std::string, std::string> defaults, active_files;
  std::string compute;
};

struct GpuInfo {
  int index = 0;
  std::string name;
  size_t total = 0, free = 0;
  std::string pci_bus;
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::vector<ModelInfo> models() = 0;
  virtual std::vector<GpuInfo> gpus() = 0;
  virtual void load_model(const std::string& id, const std::map<std::string, std::string>& files, const std::string& compute) = 0;
  virtual void unload_models() = 0;
  // cancel 由呼叫端設成 true 時，引擎在下一步之間中止並丟出例外。
  virtual ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                                     const std::atomic<bool>* cancel) = 0;
  virtual VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                                     const std::atomic<bool>* cancel) = 0;
};

}  // namespace mx
