#pragma once
// MiniMax-H3 Omni-Transformer（50 層 dense single-stream、5376 維、56 頭×128、模態別 AdaLN）。
// 讀原始 MiniMax 命名的 GGUF（unsloth/leejet），目前支援 pruned 版（adaln_t_table 曲線表）。
// 權重逐層串流：VRAM 只需兩層權重，其餘留給長序列啟用值。
#include <memory>
#include <vector>

#include "core/streamer.h"

namespace mx {

enum H3Tag { H3_VIDEO = 0, H3_TEXT = 1, H3_AUDIO = 2 };

// 打包序列的一段：連續的列共用同一個 (timestep 索引, 模態)。
struct H3Segment {
  int64_t start = 0, len = 0;
  int t_idx = 0;  // 對應 forward() 傳入的 timesteps 陣列
  H3Tag tag = H3_VIDEO;
};

struct H3Block {
  int64_t start = 0, len = 0;
};

struct H3Seq {
  int64_t S = 0;
  std::vector<H3Segment> segs;  // 依序覆蓋 [0, S)
  int64_t text_start = 0, text_len = 0;
  // 條件幀（FL2VA 首／尾幀）與參考（Ref2VA）的影片列、音訊列區塊：資料依序放在 forward 輸入的前段。
  std::vector<H3Block> vref, aref;
  int64_t audio_start = 0, audio_len = 0;   // 目標音訊 2A 列（聲道優先）
  int64_t video_start = 0, video_len = 0;   // 目標影片列
  int64_t vref_len() const {
    int64_t n = 0;
    for (auto& b : vref) n += b.len;
    return n;
  }
  int64_t aref_len() const {
    int64_t n = 0;
    for (auto& b : aref) n += b.len;
    return n;
  }
  DevBuf cos, sin;  // [S, 48]
  DevBuf text;      // token refiner 之後的文字列 f32 [T, dim]
};

class H3Dit {
 public:
  // max_layers：只載入前 N 層（除錯／數值比對用；0 = 全部）。
  // compute：50 層主幹與 refiner 的大矩陣運算精度（FP8/FP4 需 sm_100+，否則退回 bf16）。
  H3Dit(const TensorStore& st, int device, int max_layers = 0, QMode compute = QMode::BF16);
  QMode compute() const { return compute_; }
  int64_t dim() const { return dim_; }
  int64_t video_in() const { return video_in_.in; }  // 96 = 24×1×2×2
  int64_t audio_in() const { return audio_in_.in; }  // 32

  // positions：[S, 3]（t, h, w）浮點座標；txt：文字編碼器輸出 [T, 5120] bf16（裝置）。
  void prepare(Ctx& c, H3Seq& seq, const std::vector<float>& positions, const bf16* txt);
  // video：[vref_len + video_len, 96] f32（參考／條件列在前）、audio：[aref_len + audio_len, 32] f32；
  // timesteps 為各段使用的 t 值（0..1，1=乾淨）。輸出 v_video：[video_len, 96]、v_audio：[audio_len, 32]（f32）。
  void forward(Ctx& c, H3Seq& seq, const float* video, const float* audio,
               const std::vector<float>& timesteps, float* v_video, float* v_audio);

  size_t scratch_bytes() const;
  size_t act_bytes(int64_t S) const;  // 估計一次前向需要的啟用值
  std::vector<Linear*> linears();
  std::vector<std::string> linear_names();
  bool resolve_lora(const std::string& name, LoraTarget& t);
  // LoRA 改了之後呼叫：FP8/FP4 串流會在搬運時把 LoRA 併進權重，已預取的槽要重搬。
  void invalidate_streams() {
    st_refiner_->invalidate();
    st_blocks_->invalidate();
  }

 private:
  struct Block {
    Linear qkv, out, fc1, fc2, adaln;
    const float *norm1, *norm2, *q_norm, *k_norm;
  };
  void time_basis(Ctx& c, const std::vector<float>& ts, float* basis);
  void attn_mlp(Ctx& c, const Block& b, float* x, int64_t S, const float* cs, const float* sn,
                const std::vector<H3Segment>& segs, const float* mods, int64_t mod_stride);

  DeviceWeights dw_;
  int64_t dim_ = 0, ffn_ = 0, inner_ = 0;
  int heads_ = 56, head_dim_ = 128, rot_ = 96;
  QMode compute_ = QMode::BF16;
  float eps_ = 1e-5f;
  Linear video_in_, audio_in_, cond_proj_, final_adaln_, video_out_, audio_out_;
  const float* final_norm_ = nullptr;
  const float* refiner_norm_ = nullptr;
  std::vector<float> table_;  // [grid, basis]
  int64_t grid_ = 0, basis_ = 0;
  std::vector<float> inv_freq_;
  std::vector<Block> refiner_, blocks_;
  std::unique_ptr<BlockStreamer> st_refiner_, st_blocks_;
};

// 影片 / 音訊排程（MiniMaxH3Scheduler）：回傳 steps 個 sigma（最後一個為 0），前向次數 = steps-1。
std::vector<float> h3_sigmas(int steps, float shift);

}  // namespace mx
