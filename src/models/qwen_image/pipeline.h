#pragma once
// Qwen-Image-2.1 完整生成流程：tokenizer → Qwen3-VL 文字編碼 → DiT 取樣 → VAE 解碼。
#include <memory>
#include <string>

#include "app/engine.h"
#include "core/ctx.h"
#include "models/qwen_image/dit.h"
#include "models/qwen_image/vae.h"
#include "server/image_ops.h"
#include "text/qwen3_text.h"
#include "text/qwen3_vision.h"
#include "text/tokenizer.h"

namespace mx {

struct QwenImagePaths {
  std::string dit;           // DiT GGUF / safetensors
  std::string text_encoder;  // Qwen3-VL-8B GGUF
  std::string vae;           // VAE safetensors
  std::string tokenizer;     // tokenizer.json
  std::string mmproj;        // Qwen3-VL 視覺塔（編輯用，第一次用到才載入）
  bool turbo = false;        // Qwen-Image-2.1-Turbo：預設用固定 8 步排程，不位移
};

struct QwenImageDevices {
  int dit = 0;   // 主力卡
  int text = 0;  // 文字編碼器所在卡（-1 = 自動：放得下就放第二張卡）
  int vae = 0;
  QMode compute = QMode::BF16;  // DiT 運算精度
};

class QwenImagePipeline {
 public:
  QwenImagePipeline(const QwenImagePaths& p, QwenImageDevices d);
  ~QwenImagePipeline();
  ImageResult generate(const ImageRequest& r, const ProgressFn& cb, const std::atomic<bool>* cancel);
  std::string describe() const;

  // 文字（＋條件圖）→ 編碼器隱藏狀態（已去掉 system 前綴），bf16 主機副本 [T, 4096]。
  // refs 非空時用編輯範本；slots 回傳每張圖第一個 <|image_pad|> 在輸出中的列。
  std::vector<bf16> encode_prompt(const std::string& prompt, int64_t& T, const std::vector<Rgba>& refs = {},
                                  std::vector<int64_t>* slots = nullptr);

 private:
  void apply_loras(const std::vector<LoraSpec>& loras);
  void ensure_dit_ctx(int64_t rows, size_t min_bytes = 0);

  QwenImagePaths paths_;
  QwenImageDevices dev_;
  std::unique_ptr<Tokenizer> tok_;
  TensorStore te_st_, dit_st_, vae_st_;
  std::unique_ptr<Ctx> te_ctx_, dit_ctx_, vae_ctx_;
  std::unique_ptr<Qwen3TextEncoder> te_;
  std::unique_ptr<QwenImageDit> dit_;
  std::unique_ptr<QwenImageVae> vae_;
  DevBuf latent_stats_;  // VAE 裝置上依序放 64 個 mean 與 std
  TensorStore vis_st_;
  std::unique_ptr<Qwen3VisionEncoder> vis_;
  std::unique_ptr<Ctx> vis_ctx_;
  std::vector<LoraSpec> active_loras_;
  int drop_idx_ = 0;
  bool compact_ = false;  // 編碼器與 DiT 同卡：啟用值先配小、按需放大
};

// FlowMatch Euler（Qwen-Image-2.1 設定：exponential dynamic shift + shift_terminal 0.02）。
// 回傳 steps+1 個 sigma（最後一個為 0）。
// nodes 非空時用自訂的未位移節點（如 viggle turbo 的 [1, 0.9375, 0.875, 0.75, 0.5, 0.25]），steps 取其長度。
// raw=true（Turbo）：nodes 原值直接用，不做位移。
std::vector<float> qwen_image_sigmas(int steps, int64_t image_tokens, bool shift_terminal = true,
                                     const std::vector<float>& nodes = {}, bool raw = false);
// 依目標面積與長寬比算出 32 倍數的寬高（diffusers calculate_dimensions）。
void qwen_image_dims(double area, double ratio, int& w, int& h);

}  // namespace mx
