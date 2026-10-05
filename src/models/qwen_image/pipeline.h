#pragma once
// Qwen-Image-2.1 完整生成流程：tokenizer → Qwen3-VL 文字編碼 → DiT 取樣 → VAE 解碼。
#include <memory>
#include <string>

#include "app/engine.h"
#include "core/ctx.h"
#include "models/qwen_image/dit.h"
#include "models/qwen_image/vae.h"
#include "text/qwen3_text.h"
#include "text/tokenizer.h"

namespace mx {

struct QwenImagePaths {
  std::string dit;           // DiT GGUF / safetensors
  std::string text_encoder;  // Qwen3-VL-8B GGUF
  std::string vae;           // VAE safetensors
  std::string tokenizer;     // tokenizer.json
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

  // 文字 → 編碼器隱藏狀態（已去掉 system 前綴），bf16 主機副本 [T, 4096]。
  std::vector<bf16> encode_prompt(const std::string& prompt, int64_t& T);

 private:
  void apply_loras(const std::vector<LoraSpec>& loras);

  QwenImagePaths paths_;
  QwenImageDevices dev_;
  std::unique_ptr<Tokenizer> tok_;
  TensorStore te_st_, dit_st_, vae_st_;
  std::unique_ptr<Ctx> te_ctx_, dit_ctx_, vae_ctx_;
  std::unique_ptr<Qwen3TextEncoder> te_;
  std::unique_ptr<QwenImageDit> dit_;
  std::unique_ptr<QwenImageVae> vae_;
  std::vector<LoraSpec> active_loras_;
  int drop_idx_ = 0;
};

// FlowMatch Euler（Qwen-Image-2.1 設定：exponential dynamic shift + shift_terminal 0.02）。
// 回傳 steps+1 個 sigma（最後一個為 0）。
std::vector<float> qwen_image_sigmas(int steps, int64_t image_tokens, bool shift_terminal = true);

}  // namespace mx
