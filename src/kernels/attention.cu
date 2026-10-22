#include "kernels/attention.h"

#include <cmath>
#include <limits>
#include <math_constants.h>

namespace mx {
namespace {

// Q 留在 shared memory，O 在 registers，K/V 使用兩份 shared tiles。
// 一組 tile 設定：每個 warp 處理 M 個 16 列 query（同一次讀進來的 K/V fragment 供 M 個 tile 共用），
// 每個 block W 個 warps，每輪 K 個 key。
template <int M_, int W_, int K_>
struct Tile {
  static constexpr int M = M_, keys = K_, threads = W_ * 32, queries = 16 * M_ * W_;
};
// K 固定 32：每列的累加順序與舊版相同。K=64 在 G4 再快約 4%，但 h3_dit_parity 的音訊誤差會超標。
using WideTile = Tile<2, 4, 32>;  // 每 SM shared memory 夠放兩個 block 的卡（A100 實測最快）
using DeepTile = Tile<2, 8, 32>;  // 每 SM 約 100KB 的卡（G4／RTX PRO 6000 實測最快）
using SmallTile = Tile<1, 4, 32>;

__device__ __forceinline__ unsigned shared_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// ldmatrix 位址一律是 32 位元 shared 位址；呼叫端傳「暫存器＋編譯期常數」，ptxas 會折成 [R+imm]。
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int Stride>
__device__ __forceinline__ int swizzle(int row, int col) {
  return row * Stride + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ unsigned pack_bf16(float x, float y) {
  __nv_bfloat162_raw pair = __floats2bfloat162_rn(x, y);
  return unsigned(pair.x) | (unsigned(pair.y) << 16);
}

template <int D, int SD, int kKeys, int kThreads, bool Aligned>
__device__ __forceinline__ void copy_kv(bf16* ks, bf16* vs, const bf16* k, const bf16* v,
                                       int start, int head, const AttnParams& p) {
  for (int i = threadIdx.x; i < kKeys * SD / 8; i += kThreads) {
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

template <int kKeys, bool Causal>
__device__ __forceinline__ void online_softmax(float (&score)[kKeys / 8][4], float (&maximum)[2],
    float (&sum)[2], float (&alpha)[2], int qrow, int lane, int64_t start, const AttnParams& p) {
  const float scale = p.scale;
  // 非因果且整塊 key 都在範圍內時不用遮罩（超出 Sq 的 query 列不會寫回）；分支整個 block 一致。
  const bool edge = Causal || start + kKeys > p.Skv;
  float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F}, tilesum[2] = {}, base[2];
  #pragma unroll
  for (int n = 0; n < kKeys / 8; ++n) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float s = score[n][j] * scale * 1.4426950408889634f;
      if (edge) {
        int row = qrow + lane / 4 + (j / 2) * 8;
        int64_t col = start + n * 8 + (lane & 3) * 2 + (j & 1);
        bool valid = row < p.Sq && col < p.Skv;
        if constexpr (Causal) valid = valid && col <= int64_t(p.q_offset) + row;
        if (!valid) s = -CUDART_INF_F;
      }
      score[n][j] = s;
      tilemax[j / 2] = fmaxf(tilemax[j / 2], s);
    }
  }
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    tilemax[r] = fmaxf(tilemax[r], __shfl_xor_sync(0xffffffff, tilemax[r], 1));
    tilemax[r] = fmaxf(tilemax[r], __shfl_xor_sync(0xffffffff, tilemax[r], 2));
    float next = fmaxf(maximum[r], tilemax[r]);
    alpha[r] = maximum[r] == -CUDART_INF_F ? 0.f : exp2f(maximum[r] - next);
    maximum[r] = next;
    // 整列仍全被遮罩時用 0 當基準，exp2f(-inf - 0) = 0，避免 -inf - -inf 變 NaN。
    base[r] = next == -CUDART_INF_F ? 0.f : next;
  }
  #pragma unroll
  for (int n = 0; n < kKeys / 8; ++n) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      score[n][j] = exp2f(score[n][j] - base[j / 2]);
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

template <int D, class T, bool Aligned, bool Causal>
__global__ void __launch_bounds__(T::threads) flash_kernel(const bf16* q, const bf16* k, const bf16* v,
                                                           bf16* o, AttnParams p) {
  constexpr int M = T::M, kKeys = T::keys, kThreads = T::threads, kQ = T::queries;
  constexpr int PD = (D + 15) / 16 * 16, SD = (D + 63) / 64 * 64;
  constexpr int kRowBytes = SD * 2, kTileBytes = kKeys * kRowBytes;  // 一份 K 或 V tile
  extern __shared__ __align__(16) unsigned char storage[];
  bf16* sm = reinterpret_cast<bf16*>(storage);
  bf16* kv = sm + kQ * SD;
  const unsigned qbase = shared_addr(sm), kvbase = shared_addr(kv);
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int qstart = blockIdx.x * kQ, head = blockIdx.y, kvhead = head / (p.Hq / p.Hkv);
  for (int i = threadIdx.x; i < kQ * SD; i += kThreads) {
    int row = i / SD, col = i % SD;
    sm[swizzle<SD>(row, col)] = qstart + row < p.Sq && col < D
        ? q[int64_t(qstart + row) * p.q_stride + head * D + col] : __float2bfloat16(0.f);
  }
  // ldmatrix 每個 lane 的位址：tile 起點都是 8 列的倍數，所以 swizzle 的 XOR 只由 lane 決定。
  // 欄位 j*16（j<4）會碰到 XOR 位元，先算好 4 組；再往後每 64 欄與 XOR 不重疊，直接加常數。
  const int x = (lane & 7) * 8;
  const int ya = ((lane / 16) * 8) ^ x;       // Q 的 A fragment、V 的轉置 B fragment
  const int yk = (((lane / 8) & 1) * 8) ^ x;  // K 的 B fragment（一次載兩個 8-key tile）
  unsigned qa[4], ka[4], va[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    qa[j] = qbase + ((warp * M * 16 + (lane & 15)) * SD + ((j * 16) ^ ya)) * 2;
    ka[j] = (((lane / 16) * 8 + (lane & 7)) * SD + ((j * 16) ^ yk)) * 2;
    va[j] = ((lane & 15) * SD + ((j * 16) ^ ya)) * 2 + kTileBytes;
  }
  // K/V 搬運：每個 thread 固定負責同一欄的 16 bytes，每輪往下 kStep 列（kStep 是 8 的倍數，swizzle 不變）。
  constexpr int kChunks = SD / 8, kStep = kThreads / kChunks;
  static_assert(kThreads % kChunks == 0 && kStep % 8 == 0 && kKeys % kStep == 0, "K/V copy layout");
  const int crow = threadIdx.x / kChunks, ccol = (threadIdx.x % kChunks) * 8;
  const unsigned cdst = (crow * SD + (ccol ^ ((crow & 7) * 8))) * 2;
  const bf16* kp = k + int64_t(crow) * p.kv_stride + kvhead * D + ccol;
  const bf16* vp = v + int64_t(crow) * p.kv_stride + kvhead * D + ccol;
  auto prefetch = [&](int stage, int start) {
    if constexpr (Aligned) {
      const unsigned dst = kvbase + stage * 2 * kTileBytes + cdst;
      #pragma unroll
      for (int j = 0; j < kKeys / kStep; ++j) {
        int row = start + crow + j * kStep;
        bool valid = ccol < D && row < p.Skv;
        int64_t off = (int64_t(start) + j * kStep) * p.kv_stride;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(dst + j * kStep * kRowBytes), "l"(valid ? kp + off : k), "r"(valid ? 16 : 0) : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(dst + kTileBytes + j * kStep * kRowBytes), "l"(valid ? vp + off : v), "r"(valid ? 16 : 0)
                     : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
      bf16* ks = kv + stage * 2 * kKeys * SD;
      copy_kv<D, SD, kKeys, kThreads, Aligned>(ks, ks + kKeys * SD, k, v, start, kvhead, p);
    }
  };
  float out[M][PD / 8][4] = {}, maximum[M][2], sum[M][2] = {};
  #pragma unroll
  for (int m = 0; m < M; ++m) maximum[m][0] = maximum[m][1] = -CUDART_INF_F;
  int end = p.Skv;
  if constexpr (Causal) {
    int64_t extent = int64_t(qstart) + kQ;
    int64_t last = int64_t(p.q_offset) + (extent < p.Sq ? extent : p.Sq);
    end = static_cast<int>(last < end ? last : end);
  }
  if (end > 0) prefetch(0, 0);
  for (int64_t start = 0; start < end; start += kKeys) {
    int stage = (start / kKeys) & 1;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // 過了這道 barrier，Q 已寫好，且所有 warps 都已讀完上一輪的另一個 stage，可以直接預取覆寫它。
    __syncthreads();
    if (start + kKeys < end) prefetch(stage ^ 1, static_cast<int>(start + kKeys));
    const unsigned sbase = kvbase + stage * 2 * kTileBytes;
    unsigned kk[4], vv[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) kk[j] = sbase + ka[j], vv[j] = sbase + va[j];

    float score[M][kKeys / 8][4] = {}, alpha[M][2];
    #pragma unroll
    for (int d = 0; d < PD / 16; ++d) {
      unsigned a[M][4];
      #pragma unroll
      for (int m = 0; m < M; ++m) ldsm_x4(a[m], qa[d & 3] + ((d >> 2) * 64 + m * 16 * SD) * 2);
      #pragma unroll
      for (int t = 0; t < kKeys / 16; ++t) {
        unsigned b[4];
        ldsm_x4(b, kk[d & 3] + ((d >> 2) * 64 + t * 16 * SD) * 2);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          mma(score[m][2 * t], a[m], b[0], b[1]);
          mma(score[m][2 * t + 1], a[m], b[2], b[3]);
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m)
      online_softmax<kKeys, Causal>(score[m], maximum[m], sum[m], alpha[m], qstart + (warp * M + m) * 16, lane, start, p);
    #pragma unroll
    for (int m = 0; m < M; ++m) {
      // 跑到後面最大值很少再變（alpha 全是 1），整個 warp 一起跳過這 PD/2 次乘法。
      if (__all_sync(0xffffffff, alpha[m][0] == 1.f && alpha[m][1] == 1.f)) continue;
      #pragma unroll
      for (int d = 0; d < PD / 8; ++d) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) out[m][d][j] *= alpha[m][j / 2];
      }
    }
    // MMA 的 C fragment 改排成下一次 MMA 的 A fragment，免去 shared P tile。
    #pragma unroll
    for (int n = 0; n < kKeys / 16; ++n) {
      unsigned a[M][4];
      #pragma unroll
      for (int m = 0; m < M; ++m) {
        a[m][0] = pack_bf16(score[m][2*n][0], score[m][2*n][1]);
        a[m][1] = pack_bf16(score[m][2*n][2], score[m][2*n][3]);
        a[m][2] = pack_bf16(score[m][2*n+1][0], score[m][2*n+1][1]);
        a[m][3] = pack_bf16(score[m][2*n+1][2], score[m][2*n+1][3]);
      }
      #pragma unroll
      for (int u = 0; u < PD / 16; ++u) {
        unsigned b[4];
        ldsm_x4_trans(b, vv[u & 3] + ((u >> 2) * 64 + n * 16 * SD) * 2);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          mma(out[m][2 * u], a[m], b[0], b[1]);
          mma(out[m][2 * u + 1], a[m], b[2], b[3]);
        }
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; ++m)
    store_output<D, PD>(o, out[m], sum[m], qstart + (warp * M + m) * 16, lane, head, p);
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

// Q 常駐一塊，後面接兩份 K/V tiles。
template <int D, class T>
constexpr int smem_bytes = (T::queries + 4 * T::keys) * ((D + 63) / 64 * 64) * sizeof(bf16);

template <int D, class T, bool Aligned, bool Causal>
void launch(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p, cudaStream_t s) {
  constexpr int bytes = smem_bytes<D, T>;
  MX_CUDA(cudaFuncSetAttribute(flash_kernel<D, T, Aligned, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  flash_kernel<D, T, Aligned, Causal><<<dim3((p.Sq - 1) / T::queries + 1, p.Hq), T::threads, bytes, s>>>(q, k, v, o, p);
}

template <int D, class T>
void dispatch(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p, cudaStream_t s) {
  bool aligned = p.kv_stride % 8 == 0 && (reinterpret_cast<uintptr_t>(k) & 15) == 0 &&
                 (reinterpret_cast<uintptr_t>(v) & 15) == 0;
  if (aligned && p.causal) launch<D, T, true, true>(q, k, v, o, p, s);
  else if (aligned) launch<D, T, true, false>(q, k, v, o, p, s);
  else if (p.causal) launch<D, T, false, true>(q, k, v, o, p, s);
  else launch<D, T, false, false>(q, k, v, o, p, s);
}

// 依每 SM 的 shared memory 選 tile：放得下兩個 WideTile block 就用它，
// 否則用單一 block 佔滿一個 SM 的 DeepTile，連它都放不下才退回 SmallTile。
template <int D>
void dispatch(const bf16* q, const bf16* k, const bf16* v, bf16* o, AttnParams p, cudaStream_t s) {
  int device, per_sm, per_block, reserved;
  MX_CUDA(cudaGetDevice(&device));
  MX_CUDA(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  MX_CUDA(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  MX_CUDA(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));
  if (per_sm >= 2 * (smem_bytes<D, WideTile> + reserved)) dispatch<D, WideTile>(q, k, v, o, p, s);
  else if (per_block >= smem_bytes<D, DeepTile> && per_sm >= smem_bytes<D, DeepTile> + reserved)
    dispatch<D, DeepTile>(q, k, v, o, p, s);
  else dispatch<D, SmallTile>(q, k, v, o, p, s);
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
