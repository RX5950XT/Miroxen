#pragma once
// Qwen-Image-2.1 DiT（32 層 single-stream、共用調變、block-causal 注意力、prefix KV cache）。
#include <memory>
#include <vector>

#include "core/streamer.h"
#include "core/weights.h"

namespace mx {

struct QwenImageDitConfig {
  int64_t dim = 4096, ffn = 12288, in_ch = 64, ctx_dim = 4096;
  int heads = 32, head_dim = 128, layers = 32;
  float eps = 1e-6f;
};

// 編輯用條件圖：文字編碼器序列中從 slot0 起連續 (H/2)·(W/2) 個 <|image_pad|>（每個代表 32×32 像素），
// 在 DiT 序列裡展開成 H·W 個 latent token（每個 16×16 像素），內容換成 VAE latent。
struct QwenImageCond {
  int64_t slot0 = 0;
  int64_t H = 0, W = 0;            // latent 網格
  const float* latents = nullptr;  // 裝置 [H·W, 64] f32，已正規化
};

// 一次生成（固定 prompt）的狀態：序列版面、RoPE 表、prefix K/V 快取。
// prefix = 文字段與條件圖段交錯；文字段因果、圖片段內部雙向，每段都看得到前面所有列。
struct QwenImageSeq {
  struct Seg {
    int64_t start, len;
    bool text;
  };
  int64_t T = 0;        // prefix 列數（文字 + 條件圖）
  int64_t H = 0, W = 0; // 目標 latent 網格
  int64_t N() const { return H * W; }
  int64_t S() const { return T + N(); }
  std::vector<Seg> segs;  // 依序覆蓋 [0, T)
  DevBuf cos, sin;      // [S, 64]
  DevBuf txt;           // prefix 列（txt_in 後的文字、img_in 後的條件圖）f32 [T, dim]
  std::vector<DevBuf> kcache, vcache;  // 每層 [T, H, D] bf16
  bool cached = false;
};

class QwenImageDit {
 public:
  // compute：區塊內大矩陣的運算精度（FP8/FP4 需 sm_100+，不支援時退回 bf16）。
  // stream=true：區塊權重留在檔案（mmap），每層計算前才搬上 GPU（顯存不夠放整個 DiT 時用）。
  QwenImageDit(const TensorStore& st, int device, QMode compute = QMode::BF16, bool stream = false);
  bool streaming() const { return streamer_ != nullptr; }
  QMode compute() const { return compute_; }
  const QwenImageDitConfig& cfg() const { return cfg_; }

  // 準備序列：txt 為文字編碼器輸出 [T, ctx_dim]（bf16，裝置；含條件圖的 <|image_pad|> 列）。
  void prepare(Ctx& c, QwenImageSeq& seq, const bf16* txt, int64_t T, int64_t H, int64_t W,
               const std::vector<QwenImageCond>& conds = {});
  // 預測速度場：latents [N, 64] f32 → v [N, 64] f32。t 為 sigma（0..1）。
  // 第一次呼叫算完整序列並填 prefix 快取，之後只算目標 token。
  void forward(Ctx& c, QwenImageSeq& seq, const float* latents, float t, float* v);

  size_t scratch_bytes() const;
  size_t weight_bytes() const { return dw_.bytes() + (streamer_ ? streamer_->weight_bytes() : 0); }
  std::vector<Linear*> linears();
  std::vector<std::string> linear_names();
  // LoRA 名稱（diffusers / ComfyUI 命名）→ 目標層。
  bool resolve_lora(const std::string& name, LoraTarget& t);
  // 量化模式下把目前的 LoRA 併進權重（換 LoRA 時呼叫一次）。
  void merge_loras(Ctx& c) {
    if (streamer_) return streamer_->invalidate();  // 串流槽在搬運時併入 LoRA
    for (Linear* l : linears()) dw_.merge_loras(c, *l);
  }

 private:
  struct Block {
    Linear q, k, v, o, gate_up, out;
    const float *norm_q, *norm_k;
  };
  void time_mod(Ctx& c, float t, float* mod, float* out_scale);
  void run_block(Ctx& c, QwenImageSeq& seq, int li, float* x, int64_t rows, bool prefill,
                 const float* mod);

  QwenImageDitConfig cfg_;
  QMode compute_ = QMode::BF16;
  bool attn8_ = false;  // 注意力走 INT8/FP8：要求低精度且 sm_89 以上（線性層退回 bf16 時也照用，例如 L4）
  DeviceWeights dw_;
  Linear img_in_, txt_in1_, txt_in2_, t1_, t2_, modulation_, norm_out_, proj_out_;
  DevBuf txt_norm_w_;  // zero-centered → 已加 1
  std::vector<Block> blocks_;
  std::unique_ptr<BlockStreamer> streamer_;
};

}  // namespace mx
