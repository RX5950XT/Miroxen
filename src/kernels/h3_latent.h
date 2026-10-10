#pragma once
#include "core/common.h"

namespace mx::h3_latent {
// DiT 生成列（不含參考列）→ VAE 的聲道優先排列；僅搬位元，不改數值。
void video(const float* rows, float* z, int64_t frames, int64_t h, int64_t w, cudaStream_t stream);
void audio(const float* rows, float* z, int64_t samples, cudaStream_t stream);
}  // namespace mx::h3_latent
