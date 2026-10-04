#pragma once
// MiniMax-H3 音訊 VAE（32 kHz、40 Hz latent、32 通道；左右聲道共用同一個單聲道編解碼器）。
#include <cstdint>
#include <vector>

#include "core/ctx.h"
#include "core/model_file.h"

namespace mx {

class H3AudioVae {
 public:
  // st：unsloth/Comfy 的 minimax_h3_audio_vae_fp32.safetensors 或 diffusers 格式 audio_vae/（皆需支援）。
  H3AudioVae(const TensorStore& st, int device);
  ~H3AudioVae();

  // z：裝置 [2, 32, A] f32「正規化」latent（第 0 維為左右聲道；本類別負責乘 std 加 mean）。
  // 輸出：主機端交錯立體聲 float PCM（L R L R ...），取樣率 32000。
  void decode(Ctx& c, const float* z, int64_t A, std::vector<float>& pcm);
  // 參考音訊：交錯立體聲 32 kHz PCM → 裝置 [2, 32, A] 正規化 latent，A = round(秒數*40)。
  void encode(Ctx& c, const float* pcm, int64_t samples, float* z, int64_t& A);
  static constexpr int kSampleRate = 32000;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace mx
