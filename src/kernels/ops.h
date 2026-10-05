#pragma once
// 模型共用的逐元素／正規化核心。殘差流用 f32，矩陣乘法輸入輸出用 bf16。
#include <cstdint>

#include "core/common.h"

namespace mx {

// 每列兩組調變向量：列索引 < prefix_len 用 *_prefix，否則用 *_target（Qwen-Image causal_condition）。
// 只有一組時把兩個指標設成同一個、prefix_len=0。
struct RowMod {
  const float* target = nullptr;
  const float* prefix = nullptr;
  int64_t prefix_len = 0;
};

// out = LayerNorm(x)（無仿射）* (1 + scale) + shift；shift 可為空。
void layernorm_mod(const float* x, int64_t rows, int64_t dim, float eps, RowMod scale, RowMod shift,
                   bf16* out, cudaStream_t s);
// out = RMSNorm(x) * w * (1 + scale) + shift；w/scale/shift 皆可為空（RowMod.target==nullptr 表示略過）。
void rmsnorm_mod(const float* x, int64_t rows, int64_t dim, float eps, const float* w, RowMod scale,
                 RowMod shift, bf16* out, cudaStream_t s);
// 同上但輸入為 bf16（文字編碼器用），輸出 bf16。
void rmsnorm_bf16(const bf16* x, int64_t rows, int64_t dim, float eps, const float* w, bf16* out,
                  cudaStream_t s);
// x += act(gate) * y；gate_tanh=1 時 act=tanh，否則 act=identity；gate.target 為空時 x += y。
void gated_residual(float* x, const bf16* y, int64_t rows, int64_t dim, RowMod gate, int gate_tanh,
                    cudaStream_t s);
// x(bf16) += y(bf16)（文字編碼器殘差）。
void add_inplace_bf16(bf16* x, const bf16* y, int64_t n, cudaStream_t s);

// RoPE 版面：
//   ROPE_INTERLEAVED：相鄰通道 (2i, 2i+1) 一組（Qwen-Image），cos/sin 為 [S, rot/2]
//   ROPE_HALF：通道 (i, i+rot/2) 一組（rotate_half；H3、Qwen3-VL），cos/sin 為 [S, rot/2]
enum RopeMode { ROPE_NONE = 0, ROPE_INTERLEAVED = 1, ROPE_HALF = 2 };

// in：[S, in_stride] 的 bf16 列，取第 h 個頭 = in[s*in_stride + col0 + h*D .. +D)。
// 每頭 RMSNorm（norm_w 為空則略過）後套 RoPE（只轉前 rot 個通道），寫到 out [S, H, D]。
void head_norm_rope(const bf16* in, int64_t S, int64_t in_stride, int64_t col0, int H, int D,
                    const float* norm_w, float eps, const float* cos, const float* sin, int rot,
                    RopeMode mode, bf16* out, cudaStream_t s);

// gu：[rows, 2F]，前半為 gate、後半為 up；out = silu(gate) * up，out [rows, F]。
void silu_mul(const bf16* gu, int64_t rows, int64_t F, bf16* out, cudaStream_t s);
// 原地 GELU（tanh 近似）。
void gelu_tanh_inplace(bf16* x, int64_t n, cudaStream_t s);
// 原地 GELU（erf 精確版，nn.GELU() 預設）。
void gelu_erf_inplace(bf16* x, int64_t n, cudaStream_t s);
// x[rows[i], :] = src[i, :]（add=1 時改為 +=）；rows 為裝置端 int32 [n]、x 為 f32 [*, dim]、src 為 bf16 [n, dim]。
void scatter_rows(float* x, int64_t dim, const int32_t* rows, const bf16* src, int64_t n, int add, cudaStream_t s);
void silu_inplace_f32(float* x, int64_t n, cudaStream_t s);
// y[r, :] += bias（bias 為 f32），bf16 / f32 兩版。
void add_bias_bf16(bf16* y, const float* bias, int64_t rows, int64_t dim, cudaStream_t s);
void add_bias_f32(float* y, const float* bias, int64_t rows, int64_t dim, cudaStream_t s);

void f32_to_bf16(const float* x, bf16* y, int64_t n, cudaStream_t s);
void bf16_to_f32(const bf16* x, float* y, int64_t n, cudaStream_t s);
// x = x + a * y（f32），flow 取樣器用。
void axpy_f32(float* x, const float* y, float a, int64_t n, cudaStream_t s);
// x = a * x + b * y（f32）。
void axpby_f32(float* x, const float* y, float a, float b, int64_t n, cudaStream_t s);

}  // namespace mx
