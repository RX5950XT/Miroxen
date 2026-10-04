#pragma once
// FlashAttention（bf16 輸入、fp32 累加、bf16 輸出）。
#include "core/common.h"

namespace mx {

// 版面（皆為 row-major，token 在最外層）：
//   q: [Sq, Hq, D]，k/v: [Skv, Hkv, D]，o: [Sq, Hq, D]；Hq 必須是 Hkv 的倍數（GQA）。
//   q_stride / kv_stride / o_stride 是相鄰 token 之間的元素距離（0 表示緊密排列 = H*D）。
// 遮罩：causal=0 時每個 query 看全部 Skv 個 key；
//       causal=1 時 query i（0 起算）只看 key [0, q_offset + i]（含）。
struct AttnParams {
  int Sq = 0, Skv = 0, Hq = 0, Hkv = 0, D = 0;
  float scale = 0.f;  // 0 表示 1/sqrt(D)
  int causal = 0;
  int q_offset = 0;
  int64_t q_stride = 0, kv_stride = 0, o_stride = 0;
};

// 支援 D ∈ {64, 72, 128}。
void flash_attn(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p,
                cudaStream_t s);

// 慢速參考實作（GPU 樸素版，fp32 softmax），只給測試用。
void attn_reference(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p,
                    cudaStream_t s);

}  // namespace mx
