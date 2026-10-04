#pragma once
#include "core/common.h"

namespace mx::qvae {
void columns(const bf16* x, bf16* col, int h, int w, int channels, int oh, int ow,
             int kernel, int stride, int pad, bool up, int start, int rows, cudaStream_t s);
void finish(const float* x, const float* bias, bf16* y, int rows, int channels, cudaStream_t s);
void norm(const bf16* x, const float* gamma, bf16* y, int pixels, int channels, bool silu,
          cudaStream_t s);
void add(bf16* y, const bf16* x, int64_t n, cudaStream_t s);
void shortcut(bf16* y, const bf16* x, int h, int w, int ci, int co, int spatial,
              int temporal, bool up, cudaStream_t s);
void softmax(float* x, int rows, int cols, cudaStream_t s);
void attention_value(const bf16* x, float* y, int pixels, int channels, cudaStream_t s);
void attention_out(const float* x, bf16* y, int64_t n, cudaStream_t s);
void latent_in(const float* x, bf16* y, int pixels, cudaStream_t s);
void latent_out(const bf16* x, float* y, int pixels, cudaStream_t s);
void image_in(const uint8_t* x, bf16* y, int pixels, cudaStream_t s);
void image_out(const bf16* x, uint8_t* y, int h, int w, int full_w, int top, int left,
               int core_h, int core_w, int dst_top, int dst_left, cudaStream_t s);
void crop(const bf16* x, bf16* y, int w, int channels, int top, int left, int h, int cw,
          cudaStream_t s);
}  // namespace mx::qvae
