#pragma once
// W8A8 INT8 GEMM：GeForce 卡上 INT8 tensor core 是 bf16(fp32 累加) 的 3.5–3.7 倍
// （實測 RTX 5060 Ti 146 vs 40 TOPS、RTX 3060 Ti 129 vs 37 TOPS）。
// 權重每輸出通道一個 scale、啟用值每 token 一個 scale，可選 128 區塊 Hadamard 旋轉壓低離群值。
#include <cstdint>

#include "core/common.h"

namespace mx {

// 量化前是否對 K 維做區塊 Hadamard 旋轉（區塊大小 128；K 必須是 128 的倍數）。
// 權重與啟用值必須用同一設定：x·H·(W·H)ᵀ = x·Wᵀ。
struct I8Opts {
  bool hadamard = false;
};

// 每 token 動態量化：x [M, K]（列距 ldx）→ xq [M, K] int8（緊密）、sx[M] = absmax/127（旋轉後）。
void quantize_rows_i8(const bf16* x, int64_t ldx, int64_t M, int64_t K, int8_t* xq, float* sx,
                      I8Opts o, cudaStream_t s);
// 權重量化（載入時一次）：w [N, K] f32 → wq [N, K] int8、sw[N]。
void quantize_weight_i8(const float* w, int64_t N, int64_t K, int8_t* wq, float* sw, I8Opts o,
                        cudaStream_t s);

// Y[M, N]（bf16，列距 ldy）= (xq · wqᵀ) * sx[m] * sw[n] (+ bias[n])；beta=1 時 Y += 結果。
// int32 累加、fp32 縮放。K 必須是 32 的倍數。
void gemm_i8(const int8_t* xq, const float* sx, const int8_t* wq, const float* sw, const float* bias,
             bf16* Y, int64_t ldy, int64_t M, int64_t N, int64_t K, float beta, cudaStream_t s);

}  // namespace mx
