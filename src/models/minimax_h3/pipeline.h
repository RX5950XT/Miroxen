#pragma once
// MiniMax-H3 T2VA 完整流程：tokenizer → Qwen3-VL-32B（第 50 層，串流）→ 打包序列 → DiT 雙排程取樣
// → 影片 VAE（邊解邊送 ffmpeg）+ 音訊 VAE → MP4（H.264 + AAC）。
#include <memory>
#include <string>

#include "app/engine.h"
#include "core/ctx.h"
#include "models/minimax_h3/audio_vae.h"
#include "models/minimax_h3/dit.h"
#include "models/minimax_h3/video_vae.h"
#include "text/qwen3_text.h"
#include "text/tokenizer.h"

namespace mx {

struct H3Paths {
  std::string dit, text_encoder, video_vae, audio_vae, tokenizer;
  std::string out_dir = "out";
};

struct H3Devices {
  int dit = 0, text = -1, video_vae = 0, audio_vae = -1;  // -1 = 有第二張卡就用第二張
  QMode compute = QMode::BF16;                            // DiT 運算精度
};

// 打包序列版面（[文字 | 條件幀 | 音訊 | 影片]）。
struct H3Layout {
  int64_t T = 0, A = 0, F_lat = 0, H = 0, W = 0;  // H、W 為 latent 尺寸（像素 / 16）
  int cond_frames = 0;
  std::vector<float> positions;  // [S, 3]
  H3Seq seq;
};
// anchors：每個條件幀是 "first" 或 "last"（T2VA 為空）。
H3Layout h3_build_layout(int64_t T, int64_t F_lat, int64_t H, int64_t W, int64_t A,
                         const std::vector<std::string>& anchors);
int h3_align_frames(int frames);        // 對齊到 17n+5
int64_t h3_latent_frames(int frames);   // (F-5)/17*5+2
int64_t h3_audio_latents(int frames);   // round(F/24*40)

class H3Pipeline {
 public:
  H3Pipeline(const H3Paths& p, H3Devices d);
  ~H3Pipeline();
  VideoResult generate(const VideoRequest& r, const ProgressFn& cb, const std::atomic<bool>* cancel);
  std::string describe() const;

 private:
  void apply_loras(const std::vector<LoraSpec>& loras);
  H3Paths paths_;
  H3Devices dev_;
  std::unique_ptr<Tokenizer> tok_;
  TensorStore te_st_, dit_st_, vvae_st_, avae_st_;
  std::unique_ptr<Qwen3TextEncoder> te_;
  std::unique_ptr<H3Dit> dit_;
  std::unique_ptr<H3VideoVae> vvae_;
  std::unique_ptr<H3AudioVae> avae_;
  std::unique_ptr<Ctx> te_ctx_, dit_ctx_, avae_ctx_;
  std::vector<LoraSpec> active_loras_;
};

}  // namespace mx
