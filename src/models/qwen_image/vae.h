#pragma once
// Qwen-Image-2.1 VAE（AutoencoderKLQwenImage21，單張圖特化的 2D 版本）。
#include <cstdint>
#include <vector>

#include "core/ctx.h"
#include "core/model_file.h"

namespace mx {

class QwenImageVae {
 public:
  // st：Comfy-Org 的 qwen_image_2.1_vae_bf16.safetensors 或 diffusers 格式 vae/diffusion_pytorch_model.safetensors。
  QwenImageVae(const TensorStore& st, int device);
  ~QwenImageVae();

  // z：裝置記憶體 [64, H, W] f32，已反正規化（z*std + mean）。
  // 輸出：主機端 RGBA8，大小 [16H, 16W, 4]，像素 = clamp((x+1)/2, 0, 1)*255 四捨五入。
  void decode(Ctx& c, const float* z, int64_t H, int64_t W, std::vector<uint8_t>& rgba);
  // 編輯用：主機端 RGBA8 [h, w, 4]（h、w 為 16 的倍數）→ 裝置 [64, h/16, w/16] f32（取分佈平均、未正規化）。
  void encode(Ctx& c, const uint8_t* rgba, int64_t h, int64_t w, float* z);

  // 解碼一張 HxW latent 需要的啟用值記憶體上限估計（位元組），給呼叫端配置 Ctx。
  size_t decode_act_bytes(int64_t H, int64_t W) const;
  static const float* latents_mean();  // 64 個
  static const float* latents_std();   // 64 個

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace mx
