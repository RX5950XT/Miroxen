#pragma once
// Qwen-Image-2.1 DiT（32 層 single-stream、共用調變、block-causal 注意力、prefix KV cache）。
#include <vector>

#include "core/weights.h"

namespace mx {

struct QwenImageDitConfig {
  int64_t dim = 4096, ffn = 12288, in_ch = 64, ctx_dim = 4096;
  int heads = 32, head_dim = 128, layers = 32;
  float eps = 1e-6f;
};

// 一次生成（固定 prompt）的狀態：序列版面、RoPE 表、prefix K/V 快取。
struct QwenImageSeq {
  int64_t T = 0;        // 文字 token 數（prefix）
  int64_t H = 0, W = 0; // 目標 latent 網格
  int64_t N() const { return H * W; }
  int64_t S() const { return T + N(); }
  DevBuf cos, sin;      // [S, 64]
  DevBuf txt;           // txt_in 之後的文字列 f32 [T, dim]
  std::vector<DevBuf> kcache, vcache;  // 每層 [T, H, D] bf16
  bool cached = false;
};

class QwenImageDit {
 public:
  // compute：區塊內大矩陣的運算精度（FP8/FP4 需 sm_100+，不支援時退回 bf16）。
  QwenImageDit(const TensorStore& st, int device, QMode compute = QMode::BF16);
  QMode compute() const { return compute_; }
  const QwenImageDitConfig& cfg() const { return cfg_; }

  // 準備序列：txt 為文字編碼器輸出 [T, ctx_dim]（bf16，裝置）。
  void prepare(Ctx& c, QwenImageSeq& seq, const bf16* txt, int64_t T, int64_t H, int64_t W);
  // 預測速度場：latents [N, 64] f32 → v [N, 64] f32。t 為 sigma（0..1）。
  // 第一次呼叫算完整序列並填 prefix 快取，之後只算目標 token。
  void forward(Ctx& c, QwenImageSeq& seq, const float* latents, float t, float* v);

  size_t scratch_bytes() const;
  size_t weight_bytes() const { return dw_.bytes(); }
  std::vector<Linear*> linears();
  std::vector<std::string> linear_names();
  // LoRA 名稱（diffusers / ComfyUI 命名）→ 目標層。
  bool resolve_lora(const std::string& name, LoraTarget& t);

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
  DeviceWeights dw_;
  Linear img_in_, txt_in1_, txt_in2_, t1_, t2_, modulation_, norm_out_, proj_out_;
  DevBuf txt_norm_w_;  // zero-centered → 已加 1
  std::vector<Block> blocks_;
};

}  // namespace mx
