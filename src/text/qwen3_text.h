#pragma once
// Qwen3 / Qwen3-VL 語言模型（只做編碼：跑完所有層，回傳最後一層輸出、不套最終 norm）。
// 讀 llama.cpp 命名的 GGUF（token_embd / blk.N.attn_q ...）。Qwen-Image-2.1 用 8B 全部 36 層；
// MiniMax-H3 用截到 50 層的 32B。
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/streamer.h"
#include "core/weights.h"
#include "text/qwen3_vision.h"

namespace mx {

// 圖文輸入：ids 中每張圖以連續的 <|image_pad|> 佔位（數量 = 該圖合併後的 token 數）。
struct TextVisionInput {
  std::vector<int32_t> rows;                 // 視覺 token 所在列（依序）
  std::vector<bf16> embeds;                  // [rows, dim]，取代這些列的詞嵌入
  std::vector<std::vector<bf16>> deepstack;  // 每份 [rows, dim]，加在第 0、1、2 層輸出的視覺列
  std::vector<int32_t> pos;                  // [n, 3] mRoPE 位置（t, h, w）
};

// 完整輸入精確比對，主機最多保留兩筆編碼結果；圖文內容超出容量時不快取。
class Qwen3TextEncoderCache {
 public:
  static constexpr size_t kMaxBytes = size_t(128) << 20;
  struct Stats { size_t hits = 0, misses = 0, bypasses = 0, bytes = 0; };

  bool eligible(const std::vector<int32_t>& ids, const TextVisionInput* vis, size_t out_size) const;
  const std::vector<bf16>* get(const std::vector<int32_t>& ids, const TextVisionInput* vis);
  void put(const std::vector<int32_t>& ids, const TextVisionInput* vis, std::vector<bf16> out);
  void bypass() { stats_.bypasses++; }
  const Stats& stats() const { return stats_; }

 private:
  struct Entry {
    std::vector<int32_t> ids;
    bool has_vis = false;
    TextVisionInput vis;
    std::vector<bf16> out;
    size_t bytes = 0;
  };
  static size_t estimate(const std::vector<int32_t>& ids, const TextVisionInput* vis, size_t out_size);
  static size_t measure(const Entry& e);
  static bool equal(const Entry& e, const std::vector<int32_t>& ids, const TextVisionInput* vis);
  std::vector<Entry> entries_;  // most recent first
  Stats stats_;
};
// 依 Qwen3-VL get_rope_index 排位置：文字三軸同步遞增，每個視覺區塊 t 固定、h/w 為網格座標（皆加上起點），
// 之後的位置從起點 + max(gh, gw) 接續。imgs 依出現順序對應每段連續的佔位 token（pad_ids 中任一種：
// <|image_pad|> 或 <|video_pad|>；影片的每個時間 patch 是獨立的一段）。
TextVisionInput build_text_vision(const std::vector<int32_t>& ids, const std::vector<int32_t>& pad_ids,
                                  const std::vector<const VisionFeatures*>& imgs);

class Qwen3TextEncoder {
 public:
  // stream=true：權重留在鎖頁 RAM 逐層搬上 GPU（32B 這種放不進顯卡的模型用）。
  Qwen3TextEncoder(const TensorStore& st, int device, bool stream = false);
  // ids → out [ids.size(), dim()]（bf16，裝置記憶體，呼叫端配置）。因果注意力；vis 非空時為圖文輸入。
  void encode(Ctx& c, const std::vector<int32_t>& ids, bf16* out, const TextVisionInput* vis = nullptr);
  int64_t dim() const { return dim_; }
  int layers() const { return int(blocks_.size()); }
  size_t scratch_bytes() const;
  // encode() n 個 token 需要的啟用值記憶體上限（含圖文輸入與呼叫端的輸出緩衝）。
  size_t act_bytes(int64_t n) const;
  size_t weight_bytes() const { return dw_.bytes() + (streamer_ ? streamer_->weight_bytes() : 0); }
  bool streaming() const { return streamer_ != nullptr; }
  const Qwen3TextEncoderCache::Stats& cache_stats() const { return cache_.stats(); }

 private:
  struct Block {
    Linear q, k, v, o, gate, up, down;
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm;
  };
  void embed(Ctx& c, const std::vector<int32_t>& ids, float* x);
  void block(Ctx& c, const Block& b, float* x, int64_t n, const float* cs, const float* sn);

  const TensorStore& st_;
  DeviceWeights dw_;
  std::string embed_name_;
  std::vector<Block> blocks_;
  std::unique_ptr<BlockStreamer> streamer_;
  Qwen3TextEncoderCache cache_;
  int64_t dim_ = 0, ffn_ = 0;
  int heads_ = 0, kv_heads_ = 0, head_dim_ = 0;
  float eps_ = 1e-6f, theta_ = 5e6f;
};

}  // namespace mx
