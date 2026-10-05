#pragma once
#include "core/common.h"

namespace mx::qvae {
// implicit GEMM 卷積（不展開 im2col），收尾同 finish；需 ci%8==0、co 為 144 的倍數或 4、x 與 weight 16 bytes 對齊。
// weight 版面 [co, k, k, ci]；輸出像素 (oy, ox) 讀輸入 (oy*stride - pady + ky, ox*stride - padx + kx)，
// 超出範圍視為 0（up 時以放大兩倍後的座標計）。residual 寬 rw，輸出在其中的起點為 (ry, rx)。
bool conv_supported(int ci, int co);
void conv(const bf16* x, const bf16* weight, const float* bias, const bf16* residual, int rw, int ry, int rx,
          bf16* y, int h, int w, int ci, int oh, int ow, int co, int k, int stride, int pady, int padx,
          bool up, cudaStream_t s);
void columns(const bf16* x, bf16* col, int h, int w, int channels, int oh, int ow,
             int kernel, int stride, int pad, bool up, int start, int rows, cudaStream_t s);
// y = bf16(bf16(x) + bias)；residual 非空時再加上它（皆為 [rows, channels]）。
void finish(const float* x, const float* bias, const bf16* residual, bf16* y, int rows, int channels,
            cudaStream_t s);
void norm(const bf16* x, const float* gamma, bf16* y, int pixels, int channels, bool silu,
          cudaStream_t s);
// y [oh, ow, co] += x [*, w, ci] 經 DupUp／平均池化的捷徑；(top, left) 為 y 在 x 放大後座標中的起點。
void shortcut(bf16* y, int oh, int ow, const bf16* x, int w, int ci, int co, int spatial,
              int temporal, bool up, int top, int left, cudaStream_t s);
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
