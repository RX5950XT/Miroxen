#pragma once
#include "core/common.h"

namespace mx::qwen_latent {
// mean/std 必須是同一 GPU 上各 64 個 float 的裝置指標；輸入輸出不可重疊。
// encode: [64, N] → [N, 64]，(x - mean) / std。
// decode: [N, 64] → [64, N]，x * std + mean。
void encode(const float* x, float* y, int64_t n, const float* mean, const float* std, cudaStream_t stream);
void decode(const float* x, float* y, int64_t n, const float* mean, const float* std, cudaStream_t stream);
}  // namespace mx::qwen_latent
