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
#include "server/image_ops.h"
#include "text/qwen3_text.h"
#include "text/qwen3_vision.h"
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
// text_tags：每個文字列的模態（關鍵幀的視覺區塊標成 H3_VIDEO）；空 = 全部 H3_TEXT。
H3Layout h3_build_layout(int64_t T, int64_t F_lat, int64_t H, int64_t W, int64_t A,
                         const std::vector<std::string>& anchors, const std::vector<H3Tag>& text_tags = {});
// MiniMax-H3 畫布：短邊 short_edge、面積上限 max_pixels、兩軸取最接近的 32 倍數（resolve_canvas_size）。
void h3_canvas(double aspect_w, double aspect_h, int short_edge, int64_t max_pixels, int& w, int& h);
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
  // 文字（＋關鍵幀）→ 第 50 層隱藏狀態；tags 回傳每列模態。
  std::vector<bf16> encode_prompt(const std::string& prompt, const std::vector<Rgba>& keys, std::vector<H3Tag>& tags);
  H3Paths paths_;
  H3Devices dev_;
  std::unique_ptr<Tokenizer> tok_;
  TensorStore te_st_, dit_st_, vvae_st_, avae_st_;
  std::unique_ptr<Qwen3TextEncoder> te_;
  std::unique_ptr<H3Dit> dit_;
  std::unique_ptr<H3VideoVae> vvae_;
  std::unique_ptr<H3AudioVae> avae_;
  std::unique_ptr<Qwen3VisionEncoder> vis_;  // 32B 編碼器 GGUF 內的 visual.*，第一次用到才載入
  std::unique_ptr<Ctx> vis_ctx_;
  std::unique_ptr<Ctx> te_ctx_, dit_ctx_, avae_ctx_;
  std::vector<LoraSpec> active_loras_;
};

}  // namespace mx
