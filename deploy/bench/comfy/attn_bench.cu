// 單卡注意力測速：Miroxen flash_attn（bf16）與 flash_attn_fp8，形狀同 H3（H=56、D=128、非因果）
#include "kernels/attention.h"
#include "core/tensor.h"
#include <cstdio>
#include <vector>
using namespace mx;
__global__ void fill(bf16* p, size_t n, unsigned seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) { unsigned h = (unsigned(i) ^ seed) * 2654435761u; p[i] = __float2bfloat16(float(h >> 8) / 16777216.f * 2.f - 1.f); }
}
int main() {
  for (int S : {4096, 16384, 32560, 32768}) {
    const int H = 56; size_t n = size_t(S) * H * 128;
    bf16 *q, *k, *v, *o; void* ws;
    AttnParams p; p.Sq = p.Skv = S; p.Hq = p.Hkv = H; p.D = 128;
    MX_CUDA(cudaMalloc(&q, n * 2)); MX_CUDA(cudaMalloc(&k, n * 2)); MX_CUDA(cudaMalloc(&v, n * 2)); MX_CUDA(cudaMalloc(&o, n * 2));
    MX_CUDA(cudaMalloc(&ws, flash_attn_fp8_workspace(p)));
    fill<<<(n + 255) / 256, 256>>>(q, n, 1); fill<<<(n + 255) / 256, 256>>>(k, n, 2); fill<<<(n + 255) / 256, 256>>>(v, n, 3);
    for (int mode = 0; mode < 2; ++mode) {
      auto run = [&] { if (mode) flash_attn_fp8(q, k, v, o, p, ws, nullptr); else flash_attn(q, k, v, o, p, nullptr); };
      run(); MX_CUDA(cudaDeviceSynchronize());
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      const int it = 3; cudaEventRecord(a); for (int i = 0; i < it; ++i) run(); cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); ms /= it;
      std::printf("MX %s S=%d H=%d %.3f ms %.1f TFLOPS\n", mode ? "fp8 " : "bf16", S, H, ms, 4. * S * S * H * 128 / (ms * 1e9));
    }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o); cudaFree(ws);
  }
}
