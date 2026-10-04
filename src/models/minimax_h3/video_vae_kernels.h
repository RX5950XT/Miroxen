#pragma once
#include "core/common.h"

namespace mx::h3v {
void gather_latent(const float* z, bf16* out, int F, int H, int W, int start,
                   int T, int y, int x, int h, int w, const float* mean,
                   const float* std, cudaStream_t s);
void anchors(const bf16* projected, float* x, const float* registers,
             int patches, int dim, int nr, cudaStream_t s);
void rope(float* cos, float* sin, int T, int H, int W, int suffix, int rot,
          float theta, cudaStream_t s);
void swiglu(const bf16* gu, bf16* out, int rows, int f, bool gate_first, cudaStream_t s);
void unpack(const bf16* patches, float* pixels, int T, int H, int W,
            int pt, int p, cudaStream_t s);
void image_input(const uint8_t* rgb, bf16* out, int H, int W, int y, int x,
                 int h, int w, cudaStream_t s);
void im2col(const bf16* x, bf16* col, int H, int W, int C, int kh, int kw,
             int stride, int pad, int OH, int OW, int start, int rows, cudaStream_t s);
void groupnorm_silu(const bf16* x, bf16* out, const float* weight, const float* bias,
                     int spatial, int C, int groups, float eps, cudaStream_t s);
void channel_normalize(const float* moments, float* z, const float* mean,
                       const float* std, int rows, cudaStream_t s);
}  // namespace mx::h3v
