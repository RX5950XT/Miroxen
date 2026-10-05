#pragma once
// MiniMax-H3 影片 VAE（f16t4d24：因果 3D 卷積編碼器 + 36 層 ViT 解碼器）。
#include <cstdint>
#include <functional>

#include "core/ctx.h"
#include "core/model_file.h"
#include "kernels/blockscale.h"

namespace mx {

class H3VideoVae {
 public:
  // st：unsloth/Comfy 的 minimax_h3_video_vae_fp16.safetensors 或 diffusers 格式 vae/ 分片（皆需支援）。
  // compute：ViT 解碼器 36 層線性層的運算精度（FP8/FP4 用區塊縮放 tensor core；不支援時退回 bf16）。
  H3VideoVae(const TensorStore& st, int device, QMode compute = QMode::BF16);
  ~H3VideoVae();

  // z：裝置 [24, F_lat, H, W] f32，DiT 輸出的「正規化」latent（本類別負責乘 std 加 mean）。
  // 依序產生 RGB8 影格（每格 [16H, 16W, 3]，已做 ImageNet 反正規化並 clamp），每解完一段就呼叫 sink，
  // 讓呼叫端邊解邊寫 ffmpeg。總影格數 = 17*(F_lat-2)/5 + 5。
  using FrameSink = std::function<void(const uint8_t* rgb, int num_frames)>;
  void decode(Ctx& c, const float* z, int64_t F_lat, int64_t H, int64_t W, const FrameSink& sink);
  // 單張圖（首／尾幀條件）：RGB8 [h, w, 3]（h、w 為 32 的倍數）→ 裝置 [24, 1, h/16, w/16] 正規化 latent。
  void encode_image(Ctx& c, const uint8_t* rgb, int64_t h, int64_t w, float* z);
  // 參考影片：主機 RGB8 [F, h, w, 3] → 裝置 [24, encoded_frames(F), h/16, w/16] 正規化 latent（分佈平均）。
  // 每 17 幀一段各自因果編碼（不足補最後一幀），串接後丟掉尾端 3 個 latent 幀；F = 17n+5 時得 5n+2 幀。
  void encode_video(Ctx& c, const uint8_t* rgb, int64_t F, int64_t h, int64_t w, float* z);
  static int64_t encoded_frames(int64_t F) { return F == 1 ? 1 : (F + 16) / 17 * 5 - 3; }
  size_t decode_act_bytes(int64_t H, int64_t W) const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace mx
