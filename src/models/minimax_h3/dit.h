#pragma once
// MiniMax-H3 Omni-Transformer（50 層 dense single-stream、5376 維、56 頭×128、模態別 AdaLN）。
// 讀原始 MiniMax 命名的 GGUF（unsloth/leejet），目前支援 pruned 版（adaln_t_table 曲線表）。
// 權重逐層串流：VRAM 只需兩層權重，其餘留給長序列啟用值。
#include <algorithm>
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
  // 只複製版面、不含裝置緩衝（序列平行的第二張卡各自 prepare）。
  H3Seq layout() const {
    H3Seq s;
    s.S = S, s.segs = segs, s.text_start = text_start, s.text_len = text_len, s.vref = vref, s.aref = aref;
    s.audio_start = audio_start, s.audio_len = audio_len, s.video_start = video_start, s.video_len = video_len;
    return s;
  }
};

// 對列區間 [r0, r0+n) 與各段的交集呼叫 f(段, 起點列, 列數)。
template <class F>
inline void h3_for_rows(const std::vector<H3Segment>& segs, int64_t r0, int64_t n, F&& f) {
  for (auto& g : segs) {
    int64_t s0 = std::max(r0, g.start), s1 = std::min(r0 + n, g.start + g.len);
    if (s0 < s1) f(g, s0, s1 - s0);
  }
}

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

  // 序列平行（兩張卡）：a 負責前半列與前半注意力頭、b 負責後半，每層交換兩次（注意力前送 q/k/v、之後取回 o）。
  // 兩邊各自 prepare 過同一份版面的 seq；video／audio／輸出都在 a 的裝置上。
  static void forward_sp(H3Dit& a, Ctx& ca, H3Seq& sa, H3Dit& b, Ctx& cb, H3Seq& sb, const float* video,
                         const float* audio, const std::vector<float>& timesteps, float* v_video, float* v_audio);
  // 序列平行時本卡（負責 n 列與一半注意力頭）需要的啟用值；b 另外要放 video／audio 輸入副本。
  size_t act_bytes_sp(int64_t S, int64_t n, int64_t chunk = 0) const;

  size_t scratch_bytes() const;
  // 估計一次前向需要的啟用值；chunk ≤ 0 用目前的分段列數。
  size_t act_bytes(int64_t S, int64_t chunk = 0) const;
  // qkv／輸出投影／MLP 的分段列數：越大 GEMM 越有效率，暫存也越大（約 0.1 MiB／列）。
  static constexpr int64_t kMaxChunk = 8192, kMinChunk = 512;
  int64_t chunk() const { return chunk_; }
  void set_chunk(int64_t c) { chunk_ = c; }
  std::vector<Linear*> linears();
  std::vector<std::string> linear_names();
  bool resolve_lora(const std::string& name, LoraTarget& t);
  // LoRA 改了之後呼叫：FP8/FP4 串流會在搬運時把 LoRA 併進權重，已預取的槽要重搬。
  // 取樣期間把多餘顯存拿來常駐主幹層權重；回傳常駐層數（0 = 全部串流）。
  int set_resident_blocks(size_t bytes_budget) { return st_blocks_->set_resident(bytes_budget); }
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
  struct SpPart;
  void sp_begin(SpPart& p, const std::vector<float>& ts);
  void sp_qkv(int i, SpPart& p, const SpPart& o);
  void sp_attn(SpPart& p, const SpPart& o);
  void sp_mlp(int i, SpPart& p, const SpPart& o);
  void sp_head(SpPart& p, const SpPart& a, float* v_video, float* v_audio);
  void attn_mlp(Ctx& c, const Block& b, float* x, int64_t S, const float* cs, const float* sn,
                const std::vector<H3Segment>& segs, const float* mods, int64_t mod_stride);

  DeviceWeights dw_;
  int64_t dim_ = 0, ffn_ = 0, inner_ = 0;
  int heads_ = 56, head_dim_ = 128, rot_ = 96;
  QMode compute_ = QMode::BF16;
  float eps_ = 1e-5f;
  int64_t chunk_ = kMaxChunk;
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
