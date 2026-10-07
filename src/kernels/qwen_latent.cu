#include "kernels/qwen_latent.h"

namespace mx::qwen_latent {
namespace {
constexpr int kTile = 32, kRows = 8;

template<bool Encode>
__global__ void transform(const float* x, float* y, int64_t n, const float* mean, const float* std) {
  __shared__ float tile[kTile][kTile + 1];
  const int nx = int(blockIdx.x) * kTile, c0 = int(blockIdx.y) * kTile;
  const int tx = threadIdx.x, ty = threadIdx.y;
  #pragma unroll
  for (int j = 0; j < kTile / kRows; ++j) {
    const int c = c0 + (Encode ? ty + j * kRows : tx);
    const int p = nx + (Encode ? tx : ty + j * kRows);
    float v = 0.f;
    if (p < n) {
      if constexpr (Encode) {
        v = x[int64_t(c) * n + p];
        v = __fdiv_rn(__fsub_rn(v, mean[c]), std[c]);
      } else {
        v = x[int64_t(p) * 64 + c];
        v = __fadd_rn(__fmul_rn(v, std[c]), mean[c]);
      }
    }
    tile[ty + j * kRows][tx] = v;
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < kTile / kRows; ++j) {
    if constexpr (Encode) {
      const int c = c0 + tx, p = nx + ty + j * kRows;
      if (p < n) y[int64_t(p) * 64 + c] = tile[tx][ty + j * kRows];
    } else {
      const int c = c0 + ty + j * kRows, p = nx + tx;
      if (p < n) y[int64_t(c) * n + p] = tile[tx][ty + j * kRows];
    }
  }
}

void launch(const float* x, float* y, int64_t n, const float* mean, const float* std,
            cudaStream_t stream, bool encode) {
  MX_CHECK(x && y && mean && std && n > 0, "Invalid Qwen latent layout arguments");
  dim3 grid(unsigned((n + kTile - 1) / kTile), 2);
  dim3 block(kTile, kRows);
  if (encode) transform<true><<<grid, block, 0, stream>>>(x, y, n, mean, std);
  else transform<false><<<grid, block, 0, stream>>>(x, y, n, mean, std);
  MX_KCHECK();
}
}  // namespace

void encode(const float* x, float* y, int64_t n, const float* mean, const float* std, cudaStream_t stream) {
  launch(x, y, n, mean, std, stream, true);
}
void decode(const float* x, float* y, int64_t n, const float* mean, const float* std, cudaStream_t stream) {
  launch(x, y, n, mean, std, stream, false);
}
}  // namespace mx::qwen_latent
