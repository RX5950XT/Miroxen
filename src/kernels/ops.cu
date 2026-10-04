#include "kernels/ops.h"

namespace mx {
namespace {

constexpr int kNormThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 整個 block 求和（結果廣播給所有執行緒）。
__device__ float block_sum(float v) {
  __shared__ float sh[32];
  __shared__ float total;
  v = warp_sum(v);
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < (blockDim.x >> 5) ? sh[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) total = v;
  }
  __syncthreads();
  float r = total;
  __syncthreads();
  return r;
}

__device__ __forceinline__ const float* pick(RowMod m, int64_t row) {
  return row < m.prefix_len ? m.prefix : m.target;
}

template <class T>
__device__ __forceinline__ float ld(const T* p, int64_t i) {
  if constexpr (std::is_same_v<T, float>) return p[i];
  else return __bfloat162float(p[i]);
}

// mode 0 = LayerNorm（減平均），1 = RMSNorm。
template <int MODE, class T>
__global__ void norm_mod_kernel(const T* __restrict__ x, int64_t dim, float eps,
                                const float* __restrict__ w, RowMod scale, RowMod shift,
                                bf16* __restrict__ out) {
  int64_t row = blockIdx.x;
  const T* xr = x + row * dim;
  float mean = 0.f;
  if (MODE == 0) {
    float s = 0.f;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) s += ld(xr, i);
    mean = block_sum(s) / dim;
  }
  float ss = 0.f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float d = ld(xr, i) - mean;
    ss += d * d;
  }
  float rstd = rsqrtf(block_sum(ss) / dim + eps);
  const float* sc = scale.target ? pick(scale, row) : nullptr;
  const float* sh = shift.target ? pick(shift, row) : nullptr;
  bf16* orow = out + row * dim;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = (ld(xr, i) - mean) * rstd;
    if (w) v *= w[i];
    if (sc) v *= 1.f + sc[i];
    if (sh) v += sh[i];
    orow[i] = __float2bfloat16(v);
  }
}

__global__ void gated_residual_kernel(float* __restrict__ x, const bf16* __restrict__ y, int64_t dim,
                                      RowMod gate, int gate_tanh) {
  int64_t row = blockIdx.y;
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  float v = __bfloat162float(y[row * dim + i]);
  if (gate.target) {
    float g = pick(gate, row)[i];
    v *= gate_tanh ? tanhf(g) : g;
  }
  x[row * dim + i] += v;
}

// 每個 block 處理一個 token 的一個頭；blockDim = D（64/72/128）。
__global__ void head_norm_rope_kernel(const bf16* __restrict__ in, int64_t in_stride, int64_t col0,
                                      int H, int D, const float* __restrict__ norm_w, float eps,
                                      const float* __restrict__ cs, const float* __restrict__ sn,
                                      int rot, int mode, bf16* __restrict__ out) {
  extern __shared__ float buf[];
  int64_t s = blockIdx.x;
  int h = blockIdx.y, d = threadIdx.x;
  float v = d < D ? __bfloat162float(in[s * in_stride + col0 + int64_t(h) * D + d]) : 0.f;
  if (norm_w) {
    float ss = block_sum(v * v);
    if (d < D) v = v * rsqrtf(ss / D + eps) * norm_w[d];
  }
  if (d < D) buf[d] = v;
  __syncthreads();
  if (d < rot && mode != ROPE_NONE) {
    int half = rot / 2;
    int pair, idx;
    float other;
    if (mode == ROPE_INTERLEAVED) {
      pair = d >> 1;
      idx = d & 1;
      other = buf[d ^ 1];
    } else {
      pair = d < half ? d : d - half;
      idx = d < half ? 0 : 1;
      other = buf[d < half ? d + half : d - half];
    }
    float c = cs[s * half + pair], si = sn[s * half + pair];
    // (a, b) -> (a c - b s, a s + b c)
    v = idx == 0 ? v * c - other * si : other * si + v * c;
  }
  if (d < D) out[(s * H + h) * D + d] = __float2bfloat16(v);
}

__global__ void silu_mul_kernel(const bf16* __restrict__ gu, int64_t F, bf16* __restrict__ out) {
  int64_t row = blockIdx.y;
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= F) return;
  float g = __bfloat162float(gu[row * 2 * F + i]);
  float u = __bfloat162float(gu[row * 2 * F + F + i]);
  out[row * F + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
}

template <class F>
__global__ void map_kernel(int64_t n, F f) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x)
    f(i);
}

template <class F>
void launch_map(int64_t n, cudaStream_t s, F f) {
  if (n <= 0) return;
  int64_t blocks = (n + 255) / 256;
  if (blocks > 65535 * 8) blocks = 65535 * 8;
  map_kernel<<<unsigned(blocks), 256, 0, s>>>(n, f);
  MX_KCHECK();
}

dim3 row_grid(int64_t dim, int64_t rows) {
  MX_CHECK(rows <= 65535, "列數 %lld 超過 grid.y 上限", (long long)rows);
  return dim3(unsigned((dim + 255) / 256), unsigned(rows));
}

}  // namespace

void layernorm_mod(const float* x, int64_t rows, int64_t dim, float eps, RowMod scale, RowMod shift,
                   bf16* out, cudaStream_t s) {
  if (rows <= 0) return;
  norm_mod_kernel<0, float><<<unsigned(rows), kNormThreads, 0, s>>>(x, dim, eps, nullptr, scale,
                                                                    shift, out);
  MX_KCHECK();
}

void rmsnorm_mod(const float* x, int64_t rows, int64_t dim, float eps, const float* w, RowMod scale,
                 RowMod shift, bf16* out, cudaStream_t s) {
  if (rows <= 0) return;
  norm_mod_kernel<1, float><<<unsigned(rows), kNormThreads, 0, s>>>(x, dim, eps, w, scale, shift,
                                                                    out);
  MX_KCHECK();
}

void rmsnorm_bf16(const bf16* x, int64_t rows, int64_t dim, float eps, const float* w, bf16* out,
                  cudaStream_t s) {
  if (rows <= 0) return;
  norm_mod_kernel<1, bf16><<<unsigned(rows), kNormThreads, 0, s>>>(x, dim, eps, w, RowMod{},
                                                                   RowMod{}, out);
  MX_KCHECK();
}

void gated_residual(float* x, const bf16* y, int64_t rows, int64_t dim, RowMod gate, int gate_tanh,
                    cudaStream_t s) {
  if (rows <= 0) return;
  // grid.y 上限 65535：長序列分段送出。
  for (int64_t r0 = 0; r0 < rows; r0 += 65535) {
    int64_t n = rows - r0 < 65535 ? rows - r0 : 65535;
    RowMod g = gate;
    g.prefix_len = gate.prefix_len - r0;
    gated_residual_kernel<<<row_grid(dim, n), 256, 0, s>>>(x + r0 * dim, y + r0 * dim, dim, g,
                                                           gate_tanh);
    MX_KCHECK();
  }
}

void add_inplace_bf16(bf16* x, const bf16* y, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) {
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(y[i]));
  });
}

void head_norm_rope(const bf16* in, int64_t S, int64_t in_stride, int64_t col0, int H, int D,
                    const float* norm_w, float eps, const float* cos, const float* sin, int rot,
                    RopeMode mode, bf16* out, cudaStream_t s) {
  if (S <= 0) return;
  MX_CHECK(D % 2 == 0 && D <= 256 && rot <= D && rot % 2 == 0, "head_norm_rope: D=%d rot=%d", D, rot);
  int threads = (D + 31) / 32 * 32;
  for (int64_t s0 = 0; s0 < S; s0 += 65535 * 32) {
    int64_t n = S - s0 < 65535 * 32 ? S - s0 : 65535 * 32;
    head_norm_rope_kernel<<<dim3(unsigned(n), unsigned(H)), threads, threads * sizeof(float), s>>>(
        in + s0 * in_stride, in_stride, col0, H, D, norm_w, eps, cos ? cos + s0 * (rot / 2) : nullptr,
        sin ? sin + s0 * (rot / 2) : nullptr, rot, mode, out + s0 * H * D);
    MX_KCHECK();
  }
}

void silu_mul(const bf16* gu, int64_t rows, int64_t F, bf16* out, cudaStream_t s) {
  for (int64_t r0 = 0; r0 < rows; r0 += 65535) {
    int64_t n = rows - r0 < 65535 ? rows - r0 : 65535;
    silu_mul_kernel<<<row_grid(F, n), 256, 0, s>>>(gu + r0 * 2 * F, F, out + r0 * F);
    MX_KCHECK();
  }
}

void gelu_tanh_inplace(bf16* x, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) {
    float v = __bfloat162float(x[i]);
    float t = tanhf(0.7978845608f * (v + 0.044715f * v * v * v));
    x[i] = __float2bfloat16(0.5f * v * (1.f + t));
  });
}

void silu_inplace_f32(float* x, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) { x[i] = x[i] / (1.f + __expf(-x[i])); });
}

void add_bias_bf16(bf16* y, const float* bias, int64_t rows, int64_t dim, cudaStream_t s) {
  launch_map(rows * dim, s, [=] __device__(int64_t i) {
    y[i] = __float2bfloat16(__bfloat162float(y[i]) + bias[i % dim]);
  });
}

void add_bias_f32(float* y, const float* bias, int64_t rows, int64_t dim, cudaStream_t s) {
  launch_map(rows * dim, s, [=] __device__(int64_t i) { y[i] += bias[i % dim]; });
}

void f32_to_bf16(const float* x, bf16* y, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) { y[i] = __float2bfloat16(x[i]); });
}

void bf16_to_f32(const bf16* x, float* y, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) { y[i] = __bfloat162float(x[i]); });
}

void axpy_f32(float* x, const float* y, float a, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) { x[i] += a * y[i]; });
}

void axpby_f32(float* x, const float* y, float a, float b, int64_t n, cudaStream_t s) {
  launch_map(n, s, [=] __device__(int64_t i) { x[i] = a * x[i] + b * y[i]; });
}

}  // namespace mx
