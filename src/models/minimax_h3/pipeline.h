#pragma once
// MiniMax-H3 完整流程（T2VA／FL2VA／Ref2VA）：tokenizer＋視覺塔 → Qwen3-VL-32B（第 50 層，串流）→ 打包序列
// → DiT 雙排程取樣 → 影片 VAE（邊解邊送 ffmpeg）+ 音訊 VAE → MP4（H.264 + AAC）。
#include <functional>
#include <memory>
#include <random>
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
  std::string dit_ref;  // Ref2VA checkpoint（參考生成時才換上；兩個 DiT 不同時載入）
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
// Ref2VA 的一個參考（依請求順序）：圖片／影片給 latent 幾何，音訊（獨立或影片音軌）給每聲道 latent 數。
struct H3RefGeom {
  enum Kind { IMAGE, VIDEO, AUDIO } kind = IMAGE;
  int64_t frames = 0, h = 0, w = 0;  // latent 幀數、latent 高寬（像素 / 16）
  int64_t audio = 0;                 // 每聲道音訊 latent 數（0 = 無音訊）
};
// [文字 | 參考區塊（影片的音軌在其影片列前）| 目標音訊 | 目標影片]。參考影片列 t = max(t_video, 0.999)、
// 參考音訊列 t = 1（timesteps 索引 2、3）；共用的時間軸從文字結束處開始，每個參考往前推它佔的時間。
H3Layout h3_build_ref2va_layout(int64_t T, const std::vector<H3Tag>& text_tags, const std::vector<H3RefGeom>& refs,
                                int64_t F_lat, int64_t H, int64_t W, int64_t A);
// MiniMax-H3 畫布：短邊 short_edge、面積上限 max_pixels、兩軸取最接近的 32 倍數（resolve_canvas_size）。
void h3_canvas(double aspect_w, double aspect_h, int short_edge, int64_t max_pixels, int& w, int& h);
int h3_align_frames(int frames);        // 對齊到 17n+5，最少 22 幀（解碼器需求）
int64_t h3_latent_frames(int frames);   // (F-5)/17*5+2
int64_t h3_audio_latents(int frames);   // round(F/24*40)

class H3Pipeline {
 public:
  H3Pipeline(const H3Paths& p, H3Devices d);
  ~H3Pipeline();
  VideoResult generate(const VideoRequest& r, const ProgressFn& cb, const std::atomic<bool>* cancel);
  std::string describe() const;

 private:
  using Report = std::function<void(const char*, int, int)>;
  // 呈現的一段：標籤文字（可空）＋可選的視覺區塊（video = 影片時間 patch，用 <|video_pad|>）。
  struct Piece {
    std::string text;
    const VisionFeatures* vision = nullptr;
    bool video = false;
  };
  // 取樣前的輸入：版面、文字嵌入、參考／條件列（影片列已加雜訊、音訊列乾淨）、接著抽雜訊用的亂數。
  struct Prepared {
    H3Layout L;
    std::vector<bf16> txt;
    double text_seconds = 0;
    std::vector<float> vref, aref;
    int frames = 0;
    int64_t seed = 0;
    std::mt19937_64 rng;
  };
  void ensure_dit(bool ref);
  // 序列平行：兩張 sm_120 以上的卡（或 MX_H3_SP=1 強制、=0 關閉）時，第二張卡也放一份 DiT 一起取樣。
  void ensure_sp();
  // 單卡顯存不夠時，取樣前卸下 VAE／視覺塔／編碼器暫存，解碼時再載回（避免 WDDM 把顯存溢出到系統記憶體）。
  void ensure_vaes();
  void release_aux();
  void apply_loras(const std::vector<LoraSpec>& loras);
  // pieces 依序 → 第 50 層隱藏狀態；tags 回傳每列模態（視覺區塊含 start/end 標成影片）。
  std::vector<bf16> encode_prompt(const std::vector<Piece>& pieces, const std::string& prompt, std::vector<H3Tag>& tags);
  VisionFeatures vision(const uint8_t* rgb0, const uint8_t* rgb1, int h, int w);
  Prepared prepare_fl2va(const VideoRequest& r, const Report& report);
  Prepared prepare_ref2va(const VideoRequest& r, const Report& report);
  VideoResult sample_and_decode(const VideoRequest& r, Prepared& p, const Report& report);
  H3Paths paths_;
  H3Devices dev_;
  std::unique_ptr<Tokenizer> tok_;
  TensorStore te_st_, dit_st_, dit_ref_st_, vvae_st_, avae_st_;
  bool ref_active_ = false;  // 目前載入的是 Ref2VA DiT
  std::unique_ptr<Qwen3TextEncoder> te_;
  std::unique_ptr<H3Dit> dit_;
  std::unique_ptr<H3Dit> dit2_;  // 序列平行的第二份 DiT（dev_.text 那張卡）
  std::unique_ptr<H3VideoVae> vvae_;
  std::unique_ptr<H3AudioVae> avae_;
  std::unique_ptr<Qwen3VisionEncoder> vis_;  // 32B 編碼器 GGUF 內的 visual.*，第一次用到才載入
  std::unique_ptr<Ctx> vis_ctx_;
  std::unique_ptr<Ctx> te_ctx_, dit_ctx_, avae_ctx_;
  std::vector<LoraSpec> active_loras_;
};

}  // namespace mx
