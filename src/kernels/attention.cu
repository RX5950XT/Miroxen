#include "kernels/attention.h"

#include <cmath>
#include <limits>
#include <math_constants.h>

namespace mx {
namespace {

// 16 個 query 一個 warp；Q 與 O 留在 registers，K/V 使用兩份 shared tiles。
constexpr int kWarps = 4;
constexpr int kQueries = 16 * kWarps;
constexpr int kKeys = 32;

__device__ __forceinline__ unsigned shared_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void load_a(unsigned (&a)[4], const bf16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(shared_addr(p)));
}

template <bool Transpose>
__device__ __forceinline__ void load_b(unsigned (&b)[2], const bf16* p) {
  if constexpr (Transpose)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(shared_addr(p)));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(shared_addr(p)));
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4],
                                    const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int Stride>
__device__ __forceinline__ int swizzle(int row, int col) {
  return row * Stride + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ unsigned pack_bf16(float x, float y) {
  __nv_bfloat162_raw pair = __floats2bfloat162_rn(x, y);
  return unsigned(pair.x) | (unsigned(pair.y) << 16);
}

template <int D, int SD, bool Aligned>
__device__ __forceinline__ void copy_kv(bf16* ks, bf16* vs, const bf16* k, const bf16* v,
                                       int start, int head, const AttnParams& p) {
  for (int i = threadIdx.x; i < kKeys * SD / 8; i += kWarps * 32) {
    int row = i / (SD / 8), col = (i % (SD / 8)) * 8;
    bool valid = int64_t(start) + row < p.Skv && col < D;
    int dst = swizzle<SD>(row, col);
    int64_t src = valid ? (int64_t(start) + row) * p.kv_stride + head * D + col : 0;
    if constexpr (Aligned) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                   :: "r"(shared_addr(ks + dst)), "l"(k + src), "r"(valid ? 16 : 0) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                   :: "r"(shared_addr(vs + dst)), "l"(v + src), "r"(valid ? 16 : 0) : "memory");
    } else {
      // cp.async 最小單位是 4 bytes；奇數 stride 的 bf16 位址改用安全的 scalar 載入。
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        ks[dst + j] = valid ? k[src + j] : __float2bfloat16(0.f);
        vs[dst + j] = valid ? v[src + j] : __float2bfloat16(0.f);
      }
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int PD, int SD>
__device__ __forceinline__ void qk_scores(float (&score)[kKeys / 8][4],
                                         const unsigned (&qr)[PD / 16][4], const bf16* ks, int lane) {
  #pragma unroll
  for (int d = 0; d < PD / 16; ++d) {
    #pragma unroll
    for (int n = 0; n < kKeys / 8; ++n) {
      unsigned b[2];
      load_b<false>(b, ks + swizzle<SD>(n * 8 + lane % 8, d * 16 + ((lane / 8) & 1) * 8));
      mma(score[n], qr[d], b);
    }
  }
}

template <bool Causal>
__device__ __forceinline__ void online_softmax(float (&score)[kKeys / 8][4], float (&maximum)[2],
    float (&sum)[2], float (&alpha)[2], int qrow, int lane, int64_t start, const AttnParams& p) {
  float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F}, tilesum[2] = {};
  #pragma unroll
  for (int n = 0; n < kKeys / 8; ++n) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int row = qrow + lane / 4 + (j / 2) * 8;
      int64_t col = start + n * 8 + (lane & 3) * 2 + (j & 1);
      bool valid = row < p.Sq && col < p.Skv;
      if constexpr (Causal) valid = valid && col <= int64_t(p.q_offset) + row;
      score[n][j] = valid ? score[n][j] * p.scale * 1.4426950408889634f : -CUDART_INF_F;
      tilemax[j / 2] = fmaxf(tilemax[j / 2], score[n][j]);
    }
  }
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    tilemax[r] = fmaxf(tilemax[r], __shfl_xor_sync(0xffffffff, tilemax[r], 1));
    tilemax[r] = fmaxf(tilemax[r], __shfl_xor_sync(0xffffffff, tilemax[r], 2));
    float next = fmaxf(maximum[r], tilemax[r]);
    alpha[r] = maximum[r] == -CUDART_INF_F ? 0.f : exp2f(maximum[r] - next);
    maximum[r] = next;
  }
  #pragma unroll
  for (int n = 0; n < kKeys / 8; ++n) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      score[n][j] = score[n][j] == -CUDART_INF_F ? 0.f : exp2f(score[n][j] - maximum[j / 2]);
      tilesum[j / 2] += score[n][j];
    }
  }
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    tilesum[r] += __shfl_xor_sync(0xffffffff, tilesum[r], 1);
    tilesum[r] += __shfl_xor_sync(0xffffffff, tilesum[r], 2);
    sum[r] = sum[r] * alpha[r] + tilesum[r];
  }
}

template <int PD, int SD>
__device__ __forceinline__ void pv_accumulate(float (&out)[PD / 8][4],
    const float (&score)[kKeys / 8][4], const float (&alpha)[2], const bf16* vs, int lane) {
  #pragma unroll
  for (int d = 0; d < PD / 8; ++d) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) out[d][j] *= alpha[j / 2];
  }
  // MMA 的 C fragment 改排成下一次 MMA 的 A fragment，免去 shared P tile。
  #pragma unroll
  for (int n = 0; n < kKeys / 16; ++n) {
    unsigned a[4] = {pack_bf16(score[2*n][0], score[2*n][1]),
                     pack_bf16(score[2*n][2], score[2*n][3]),
                     pack_bf16(score[2*n+1][0], score[2*n+1][1]),
                     pack_bf16(score[2*n+1][2], score[2*n+1][3])};
    #pragma unroll
    for (int d = 0; d < PD / 8; ++d) {
      unsigned b[2];
      load_b<true>(b, vs + swizzle<SD>(n * 16 + lane % 16, d * 8));
      mma(out[d], a, b);
    }
  }
}

template <int D, int PD>
__device__ __forceinline__ void store_output(bf16* o, const float (&out)[PD / 8][4],
    const float (&sum)[2], int qrow, int lane, int head, const AttnParams& p) {
  #pragma unroll
  for (int d = 0; d < PD / 8; ++d) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int row = qrow + lane / 4 + (j / 2) * 8;
      int col = d * 8 + (lane & 3) * 2 + (j & 1);
      if (row < p.Sq && col < D)
        o[int64_t(row) * p.o_stride + head * D + col] =
            __float2bfloat16(sum[j / 2] > 0.f ? out[d][j] / sum[j / 2] : 0.f);
    }
  }
}

template <int D, bool Aligned, bool Causal>
__global__ void flash_kernel(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p) {
  constexpr int PD = (D + 15) / 16 * 16, SD = (D + 63) / 64 * 64;
  extern __shared__ __align__(16) unsigned char storage[];
  bf16* sm = reinterpret_cast<bf16*>(storage);
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int qstart = blockIdx.x * kQueries, head = blockIdx.y, kvhead = head / (p.Hq / p.Hkv);
  for (int i = threadIdx.x; i < kQueries * SD; i += kWarps * 32) {
    int row = i / SD, col = i % SD;
    sm[swizzle<SD>(row, col)] = qstart + row < p.Sq && col < D
        ? q[int64_t(qstart + row) * p.q_stride + head * D + col] : __float2bfloat16(0.f);
  }
  __syncthreads();
  unsigned qr[PD / 16][4];
  #pragma unroll
  for (int d = 0; d < PD / 16; ++d)
    load_a(qr[d], sm + swizzle<SD>(warp * 16 + lane % 16, d * 16 + (lane / 16) * 8));
  __syncthreads();
  float out[PD / 8][4] = {}, maximum[2] = {-CUDART_INF_F, -CUDART_INF_F}, sum[2] = {};
  int end = p.Skv;
  if constexpr (Causal) {
    int64_t extent = int64_t(qstart) + kQueries;
    int64_t last = int64_t(p.q_offset) + (extent < p.Sq ? extent : p.Sq);
    end = static_cast<int>(last < end ? last : end);
  }
  if (end > 0) copy_kv<D, SD, Aligned>(sm, sm + kKeys * SD, k, v, 0, kvhead, p);
  for (int64_t start = 0; start < end; start += kKeys) {
    int stage = (start / kKeys) & 1;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    bf16* ks = sm + stage * (2 * kKeys * SD);
    bf16* vs = ks + kKeys * SD;
    if (start + kKeys < end)
      copy_kv<D, SD, Aligned>(sm + (stage ^ 1) * (2 * kKeys * SD),
                             sm + (stage ^ 1) * (2 * kKeys * SD) + kKeys * SD,
                             k, v, static_cast<int>(start + kKeys), kvhead, p);
    float score[kKeys / 8][4] = {}, alpha[2];
    qk_scores<PD, SD>(score, qr, ks, lane);
    online_softmax<Causal>(score, maximum, sum, alpha, qstart + warp * 16, lane, start, p);
    pv_accumulate<PD, SD>(out, score, alpha, vs, lane);
    __syncthreads();  // 所有 warps 讀完目前 stage 後，才准許下一輪覆寫它。
  }
  store_output<D, PD>(o, out, sum, qstart + warp * 16, lane, head, p);
}

// 參考版只使用 fp32 dot product 與 softmax；每個 block 處理一個 query/head。
__device__ float block_reduce(float value, float* scratch, bool maximum) {
  __syncthreads();  // 先讓上一個 reduction 的所有 readers 離開 scratch。
  for (int offset = 16; offset; offset /= 2) {
    float other = __shfl_down_sync(0xffffffff, value, offset);
    value = maximum ? fmaxf(value, other) : value + other;
  }
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  value = threadIdx.x < 4 ? scratch[threadIdx.x] : (maximum ? -CUDART_INF_F : 0.f);
  if (threadIdx.x < 32) {
    for (int offset = 16; offset; offset /= 2) {
      float other = __shfl_down_sync(0xffffffff, value, offset);
      value = maximum ? fmaxf(value, other) : value + other;
    }
    if (threadIdx.x == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void reference_kernel(const bf16* q, const bf16* k, const bf16* v, bf16* o,
                                 AttnParams p) {
  __shared__ float weights[128], scratch[4];
  int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
  int kvhead = head / (p.Hq / p.Hkv), end = p.Skv;
  if (p.causal) {
    int64_t last = int64_t(p.q_offset) + row + 1;
    end = static_cast<int>(last < end ? last : end);
  }
  float maximum = -CUDART_INF_F, sum = 0.f, output = 0.f;
  for (int64_t start = 0; start < end; start += 128) {
    float score = -CUDART_INF_F;
    if (start + tid < end) {
      score = 0.f;
      for (int d = 0; d < p.D; ++d)
        score += __bfloat162float(q[int64_t(row) * p.q_stride + head * p.D + d]) *
                 __bfloat162float(k[int64_t(start + tid) * p.kv_stride + kvhead * p.D + d]);
      score *= p.scale;
    }
    float next = fmaxf(maximum, block_reduce(score, scratch, true));
    float alpha = maximum == -CUDART_INF_F ? 0.f : expf(maximum - next);
    float weight = score == -CUDART_INF_F ? 0.f : expf(score - next);
    weights[tid] = weight;
    float tilesum = block_reduce(weight, scratch, false);
    if (tid < p.D) {
      output *= alpha;
      for (int j = 0; j < 128 && start + j < end; ++j)
        output += weights[j] * __bfloat162float(v[int64_t(start + j) * p.kv_stride + kvhead * p.D + tid]);
    }
    maximum = next;
    sum = sum * alpha + tilesum;
    __syncthreads();
  }
  if (tid < p.D)
    o[int64_t(row) * p.o_stride + head * p.D + tid] = __float2bfloat16(sum > 0.f ? output / sum : 0.f);
}

AttnParams validate(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p) {
  MX_CHECK(p.Sq > 0 && p.Skv > 0 && p.Hq > 0 && p.Hkv > 0 && p.Hq % p.Hkv == 0,
           "attention: lengths/heads must be positive and Hq must be a multiple of Hkv");
  MX_CHECK(p.D == 64 || p.D == 72 || p.D == 128, "attention: unsupported head dimension %d", p.D);
  MX_CHECK(p.causal == 0 || p.causal == 1, "attention: causal must be 0 or 1");
  MX_CHECK(q && k && v && o, "attention: null tensor pointer");
  MX_CHECK(std::isfinite(p.scale), "attention: scale must be finite");
  const int64_t qdense = int64_t(p.Hq) * p.D, kdense = int64_t(p.Hkv) * p.D;
  if (!p.q_stride) p.q_stride = qdense;
  if (!p.kv_stride) p.kv_stride = kdense;
  if (!p.o_stride) p.o_stride = qdense;
  MX_CHECK(p.q_stride >= qdense && p.kv_stride >= kdense && p.o_stride >= qdense,
           "attention: token stride is smaller than a row");
  const int64_t limit = std::numeric_limits<int64_t>::max() / sizeof(bf16);
  MX_CHECK(p.q_stride <= limit / p.Sq && p.kv_stride <= limit / p.Skv && p.o_stride <= limit / p.Sq,
           "attention: tensor address range overflows");
  MX_CHECK(p.Hq <= 65535, "attention: too many heads for CUDA grid");
  if (p.scale == 0.f) p.scale = 1.f / std::sqrt(float(p.D));
  return p;
}

template <int D, bool Aligned, bool Causal>
void launch(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p, cudaStream_t s) {
  constexpr int bytes = 4 * kKeys * ((D + 63) / 64 * 64) * sizeof(bf16);
  MX_CUDA(cudaFuncSetAttribute(flash_kernel<D, Aligned, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  flash_kernel<D, Aligned, Causal><<<dim3((p.Sq - 1) / kQueries + 1, p.Hq), kWarps * 32, bytes, s>>>(q, k, v, o, p);
}

template <int D>
void dispatch(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p, cudaStream_t s) {
  bool aligned = p.kv_stride % 8 == 0 && (reinterpret_cast<uintptr_t>(k) & 15) == 0 &&
                 (reinterpret_cast<uintptr_t>(v) & 15) == 0;
  if (aligned && p.causal) launch<D, true, true>(q, k, v, o, p, s);
  else if (aligned) launch<D, true, false>(q, k, v, o, p, s);
  else if (p.causal) launch<D, false, true>(q, k, v, o, p, s);
  else launch<D, false, false>(q, k, v, o, p, s);
}

}  // namespace

void flash_attn(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p, cudaStream_t s) {
  AttnParams args = validate(q, k, v, o, p);
  switch (p.D) {
    case 64: dispatch<64>(q, k, v, o, args, s); break;
    case 72: dispatch<72>(q, k, v, o, args, s); break;
    case 128: dispatch<128>(q, k, v, o, args, s); break;
  }
  MX_KCHECK();
}

void attn_reference(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p, cudaStream_t s) {
  AttnParams args = validate(q, k, v, o, p);
  reference_kernel<<<dim3(args.Sq, args.Hq), 128, 0, s>>>(q, k, v, o, args);
  MX_KCHECK();
}

}  // namespace mx
