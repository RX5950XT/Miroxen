// SageAttention2 式低精度注意力：Q·Kᵀ INT8（m16n8k32 s8）、P·V FP8 e4m3（m16n8k32 e4m3），fp32 累加。
// 前處理把 Q、K 量化成 [H][S][D] int8、V 量化並轉置成 [H][D][S] fp8，主核心只讀量化資料。
#include "kernels/attention.h"

#include <cuda_fp8.h>
#include <math_constants.h>

#include <climits>
#include <cmath>

namespace mx {
namespace {

constexpr int kW = 4;          // 每個 CTA 的 warp 數；縮小 Q tile，讓 D=128 不必為兩個大型 CTA 壓到 128 registers
constexpr int kQ = 16 * kW;    // 每個 CTA 的 query 數
constexpr int kK = 64;         // 每個 K/V tile 的 key 數
constexpr int kStatTok = 256;  // kv_stats 每個 block 處理的 token 數

int64_t pad_to(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

struct Layout {
  int64_t sq, skv;
  size_t qi, ki, vt, qs, ks, km, va, total;
};

Layout layout(const AttnParams& p) {
  Layout l;
  l.sq = pad_to(p.Sq, kQ);
  l.skv = pad_to(p.Skv, kK);
  size_t off = 0;
  auto take = [&](size_t b) {
    size_t o = off;
    off += (b + 255) / 256 * 256;
    return o;
  };
  l.qi = take(size_t(p.Hq) * l.sq * p.D);
  l.ki = take(size_t(p.Hkv) * l.skv * p.D);
  l.vt = take(size_t(p.Hkv) * p.D * l.skv);
  l.qs = take(sizeof(float) * p.Hq * (l.sq / 16));
  l.ks = take(sizeof(float) * p.Hkv * (l.skv / kK));
  l.km = take(sizeof(float) * p.Hkv * p.D);
  l.va = take(sizeof(float) * p.Hkv * p.D);
  l.total = off;
  return l;
}

// ---------------- 前處理 ----------------

// K 每通道平均（softmax 平移不變，減掉後 INT8 誤差小很多）與 V 每通道最大絕對值。
template <int D>
__global__ void kv_stats_kernel(const bf16* k, const bf16* v, AttnParams p, float* kmean, float* vamax) {
  const int h = blockIdx.x, d = threadIdx.x;
  const int64_t t0 = int64_t(blockIdx.y) * kStatTok, t1 = min(t0 + kStatTok, int64_t(p.Skv));
  float s = 0.f, m = 0.f;
  for (int64_t t = t0; t < t1; ++t) {
    s += __bfloat162float(k[t * p.kv_stride + h * D + d]);
    m = fmaxf(m, fabsf(__bfloat162float(v[t * p.kv_stride + h * D + d])));
  }
  atomicAdd(&kmean[h * D + d], s / float(p.Skv));
  atomicMax(reinterpret_cast<int*>(&vamax[h * D + d]), __float_as_int(m));  // 非負 float 可當 int 比大小
}

// 每 G 列（一個 head）一個縮放：out = round((x − mean) · 127 / amax)，scale = amax / 127 · mult。
template <int D, int G>
__global__ void __launch_bounds__(128) quant_rows_kernel(const bf16* x, int64_t stride, int S, int64_t s_pad,
                                                         const float* mean, float mult, int8_t* out, float* scale) {
  constexpr int NV = G * D / 8 / 128;
  __shared__ float red[4];
  const int h = blockIdx.y, grp = blockIdx.x;
  float vals[NV][8], am = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int vi = threadIdx.x + i * 128, r = vi / (D / 8), c = (vi % (D / 8)) * 8;
    const int64_t row = int64_t(grp) * G + r;
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (row < S) raw = *reinterpret_cast<const uint4*>(x + row * stride + h * D + c);
    const bf16* b = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float f = row < S ? __bfloat162float(b[j]) - (mean ? mean[h * D + c + j] : 0.f) : 0.f;
      vals[i][j] = f;
      am = fmaxf(am, fabsf(f));
    }
  }
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = am;
  __syncthreads();
  am = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  const float inv = am > 0.f ? 127.f / am : 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int vi = threadIdx.x + i * 128, r = vi / (D / 8), c = (vi % (D / 8)) * 8;
    unsigned w[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j / 4] |= unsigned(__float2int_rn(vals[i][j] * inv) & 0xff) << ((j % 4) * 8);
    *reinterpret_cast<uint2*>(out + (int64_t(h) * s_pad + int64_t(grp) * G + r) * D + c) = make_uint2(w[0], w[1]);
  }
  if (threadIdx.x == 0) scale[int64_t(h) * (s_pad / G) + grp] = am / 127.f * mult;
}

// 32 個 key 內的排列：P 的 C fragment 直接當下一個 MMA 的 A fragment 用（免 shuffle），V 的 key 順序配合它。
// 位置 p = 16·half + 4·t + i 放的是 key 16·half + (i < 2 ? 2t + i : 8 + 2t + i − 2)。
__device__ __forceinline__ int key_pos(int key) {
  const int half = key & 16, r = key & 15;
  return r < 8 ? half + 4 * (r >> 1) + (r & 1) : half + 4 * ((r - 8) >> 1) + 2 + (r & 1);
}

// V → fp8（每通道 448/amax），轉置成 [H][D][S_pad] 並套用 key_pos 排列。
template <int D>
__global__ void v_quant_kernel(const bf16* v, AttnParams p, int64_t skv_pad, const float* vamax, uint8_t* vt) {
  __shared__ __align__(16) uint8_t tile[D][kK];
  const int h = blockIdx.y, k0 = blockIdx.x * kK;
  for (int i = threadIdx.x; i < kK * D; i += blockDim.x) {
    const int key = i / D, d = i % D;
    const int64_t gk = k0 + key;
    const float a = vamax[h * D + d];
    const float f = gk < p.Skv && a > 0.f ? __bfloat162float(v[gk * p.kv_stride + h * D + d]) * (448.f / a) : 0.f;
    tile[d][(key & 32) | key_pos(key & 31)] = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < D * kK / 16; i += blockDim.x) {
    const int d = i / (kK / 16), u = i % (kK / 16);
    *reinterpret_cast<uint4*>(vt + (int64_t(h) * D + d) * skv_pad + k0 + u * 16) =
        *reinterpret_cast<const uint4*>(&tile[d][u * 16]);
  }
}

// ---------------- 主核心 ----------------

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

// 列寬 R bytes 的 smem tile：以 16 bytes 為單位做 XOR swizzle，讓 ldmatrix 的 8 列落在不同 bank。
template <int R>
__device__ __forceinline__ int swz(int r, int u) {
  constexpr int U = R / 16, div = R >= 128 ? 1 : 128 / R;
  return r * R + ((u ^ ((r / div) % U)) * 16);
}

// |i| < 2²² 的 int32 → float：加上 1.5·2²³ 的位元樣式再減回（整數加法 + 浮點加法，避開慢的 I2F）。
// Q·K 的 int8 點積最大 128·127² ≈ 2.06M < 2²²。
__device__ __forceinline__ float i2f(int i) { return __int_as_float(i + 0x4B400000) - 12582912.f; }

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ unsigned pack_fp8x4(float a, float b, float c, float d) {
  const unsigned lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const unsigned hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

// O += A·B（e4m3，fp32 累加）。GeForce Blackwell 上一般 FP8 MMA 的 fp32 累加只有半速（實測 90 vs 180 TFLOPS），
// sm_120a 的區塊縮放版本（縮放固定 2⁰）是全速。
__device__ __forceinline__ void mma_fp8(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  const unsigned one = 0x7f7f7f7f;  // ue8m0 127 = 1.0
  const unsigned short sel = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%11}, %10, {%11,%11};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(one), "h"(sel));
#elif __CUDA_ARCH__ >= 890
  // 此路徑的累加器實際精度不足 fp32（L4 實測 S=32560 誤差 0.11），呼叫端用 kSplitAcc 每個 tile 另外累加。
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// P·V 是否每個 K/V tile 先累加到暫存再加回 fp32（兩段累加，同 SageAttention2）。sm_120a 區塊縮放版不需要。
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
constexpr bool kSplitAcc = false;
#else
constexpr bool kSplitAcc = true;
#endif

template <int D>
// 每個 SM 至少 3 個小 CTA：sm_120a 的 D=128 可用 168 registers，免去迴圈內的 spill。
__global__ void __launch_bounds__(kW * 32, 3) fp8_attn_kernel(const int8_t* qi, const int8_t* ki, const uint8_t* vt,
                                                          const float* qs, const float* ks, const float* va,
                                                          bf16* o, AttnParams p, int64_t sq_pad, int64_t skv_pad) {
#if __CUDA_ARCH__ >= 890
  constexpr int KT = kK * D, VT = D * kK;
  extern __shared__ __align__(128) unsigned char sm[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int head = blockIdx.y, kvh = head / (p.Hq / p.Hkv);
  const int q0 = blockIdx.x * kQ + warp * 16;

  // Q 的 A fragment 常駐 registers（m16n8k32 s8：a0 列 g、a1 列 g+8、a2/a3 再 +16 bytes）。
  unsigned qa[D / 32][4];
  const int8_t* qh = qi + (int64_t(head) * sq_pad + q0) * D;
#pragma unroll
  for (int c = 0; c < D / 32; ++c) {
    qa[c][0] = *reinterpret_cast<const unsigned*>(qh + g * D + c * 32 + 4 * t);
    qa[c][1] = *reinterpret_cast<const unsigned*>(qh + (g + 8) * D + c * 32 + 4 * t);
    qa[c][2] = *reinterpret_cast<const unsigned*>(qh + g * D + c * 32 + 16 + 4 * t);
    qa[c][3] = *reinterpret_cast<const unsigned*>(qh + (g + 8) * D + c * 32 + 16 + 4 * t);
  }
  const float qscale = qs[int64_t(head) * (sq_pad / 16) + q0 / 16];
  const float* kscales = ks + int64_t(kvh) * (skv_pad / kK);

  auto load = [&](int tile, int stage) {
    unsigned char* kd = sm + stage * (KT + VT);
    unsigned char* vd = kd + KT;
    const int8_t* kg = ki + (int64_t(kvh) * skv_pad + int64_t(tile) * kK) * D;
    for (int i = threadIdx.x; i < KT / 16; i += kW * 32) {
      const int r = i / (D / 16), u = i % (D / 16);
      cp_async16(kd + swz<D>(r, u), kg + r * D + u * 16);
    }
    const uint8_t* vg = vt + int64_t(kvh) * D * skv_pad + int64_t(tile) * kK;
    for (int i = threadIdx.x; i < VT / 16; i += kW * 32) {
      const int r = i / (kK / 16), u = i % (kK / 16);
      cp_async16(vd + swz<kK>(r, u), vg + int64_t(r) * skv_pad + u * 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  float acc[D / 8][4] = {}, mx[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.f, 0.f};
  const int ntiles = (p.Skv + kK - 1) / kK;
  load(0, 0);
  for (int it = 0; it < ntiles; ++it) {
    const int st = it & 1;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();  // 目前 stage 已就緒；另一個 stage 所有 warp 都用完了
    if (it + 1 < ntiles) load(it + 1, st ^ 1);
    const unsigned char* kd = sm + st * (KT + VT);
    const unsigned char* vd = kd + KT;

    // S = Q·Kᵀ（int32）。ldmatrix.x4 一次取兩個 32 維 chunk 的 B fragment。
    int s[kK / 8][4] = {};
#pragma unroll
    for (int n = 0; n < kK / 8; ++n) {
#pragma unroll
      for (int c = 0; c < D / 32; c += 2) {
        unsigned b[4];
        ldsm_x4(b, kd + swz<D>(n * 8 + (lane & 7), c * 2 + (lane >> 3)));
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+r"(s[n][0]), "+r"(s[n][1]), "+r"(s[n][2]), "+r"(s[n][3])
            : "r"(qa[c][0]), "r"(qa[c][1]), "r"(qa[c][2]), "r"(qa[c][3]), "r"(b[0]), "r"(b[1]));
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+r"(s[n][0]), "+r"(s[n][1]), "+r"(s[n][2]), "+r"(s[n][3])
            : "r"(qa[c + 1][0]), "r"(qa[c + 1][1]), "r"(qa[c + 1][2]), "r"(qa[c + 1][3]), "r"(b[2]), "r"(b[3]));
      }
    }

    // online softmax（log2 域；softmax scale·log2e 已併進 Q 的縮放）。
    // 最大值先在整數上取（縮放為正，順序不變），再一次 FFMA + ex2 算機率。
    const float sc = qscale * kscales[it];
    const bool partial = it == ntiles - 1 && p.Skv % kK;
    if (partial) {  // 超出 Skv 的 key：取最大值時當最小值，算完機率再歸零
#pragma unroll
      for (int n = 0; n < kK / 8; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (it * kK + n * 8 + 2 * t + (j & 1) >= p.Skv) s[n][j] = -(1 << 22);
    }
    float f[kK / 8][4], ts[2] = {0.f, 0.f}, alpha[2], nm[2];
    int im[2] = {INT_MIN, INT_MIN};
#pragma unroll
    for (int n = 0; n < kK / 8; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) im[j >> 1] = max(im[j >> 1], s[n][j]);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      im[r] = max(im[r], __shfl_xor_sync(0xffffffff, im[r], 1));
      im[r] = max(im[r], __shfl_xor_sync(0xffffffff, im[r], 2));
      const float next = fmaxf(mx[r], i2f(im[r]) * sc);
      alpha[r] = ex2(mx[r] - next);  // 第一個 tile：exp2(−inf) = 0
      mx[r] = next;
      // 機率直接算成 448·p（log2 448 併進指數，省一次乘法）；i2f 的減法也併進同一個 FFMA（只捨入一次）。
      nm[r] = 8.807354922057604f - next - 12582912.f * sc;
    }
#pragma unroll
    for (int n = 0; n < kK / 8; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        f[n][j] = ex2(fmaf(__int_as_float(s[n][j] + 0x4B400000), sc, nm[j >> 1]));
        if (partial && it * kK + n * 8 + 2 * t + (j & 1) >= p.Skv) f[n][j] = 0.f;
        ts[j >> 1] += f[n][j];
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      ts[r] += __shfl_xor_sync(0xffffffff, ts[r], 1);
      ts[r] += __shfl_xor_sync(0xffffffff, ts[r], 2);
      l[r] = l[r] * alpha[r] + ts[r];
    }
    // 最大值沒變時 alpha 恰為 1（ex2(0) = 1），整個 warp 都是就跳過重新縮放（後段 tile 多半如此）。
    if (!__all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) {
#pragma unroll
      for (int d = 0; d < D / 8; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[d][j] *= alpha[j >> 1];
    }

    // O += P·V（f 已是 448·p，用滿 e4m3 範圍）。
    unsigned a[kK / 32][4];
#pragma unroll
    for (int kc = 0; kc < kK / 32; ++kc) {
      a[kc][0] = pack_fp8x4(f[4 * kc][0], f[4 * kc][1], f[4 * kc + 1][0], f[4 * kc + 1][1]);
      a[kc][1] = pack_fp8x4(f[4 * kc][2], f[4 * kc][3], f[4 * kc + 1][2], f[4 * kc + 1][3]);
      a[kc][2] = pack_fp8x4(f[4 * kc + 2][0], f[4 * kc + 2][1], f[4 * kc + 3][0], f[4 * kc + 3][1]);
      a[kc][3] = pack_fp8x4(f[4 * kc + 2][2], f[4 * kc + 2][3], f[4 * kc + 3][2], f[4 * kc + 3][3]);
    }
#pragma unroll
    for (int dn = 0; dn < D / 8; dn += 2) {
      float t0[4] = {}, t1[4] = {};
      float(&c0)[4] = kSplitAcc ? t0 : acc[dn];
      float(&c1)[4] = kSplitAcc ? t1 : acc[dn + 1];
#pragma unroll
      for (int kc = 0; kc < kK / 32; ++kc) {
        unsigned b[4];
        const int j = lane >> 3;
        ldsm_x4(b, vd + swz<kK>((dn + (j >> 1)) * 8 + (lane & 7), kc * 2 + (j & 1)));
        mma_fp8(c0, a[kc], b[0], b[1]);
        mma_fp8(c1, a[kc], b[2], b[3]);
      }
      if (kSplitAcc) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[dn][j] += t0[j], acc[dn + 1][j] += t1[j];
      }
    }
  }

  const float* vh = va + kvh * D;
#pragma unroll
  for (int d = 0; d < D / 8; ++d)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = q0 + g + r * 8, col = d * 8 + 2 * t;
      if (row >= p.Sq) continue;
      const float inv = 1.f / (448.f * l[r]);  // l 已含 P 的 448；另一個 448 是 V 的 448/amax
      *reinterpret_cast<__nv_bfloat162*>(o + int64_t(row) * p.o_stride + head * D + col) =
          __floats2bfloat162_rn(acc[d][2 * r] * vh[col] * inv, acc[d][2 * r + 1] * vh[col + 1] * inv);
    }
#else
  __trap();
#endif
}

template <int D>
void run(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p, char* ws, cudaStream_t s) {
  const Layout L = layout(p);
  int8_t* qi = reinterpret_cast<int8_t*>(ws + L.qi);
  int8_t* ki = reinterpret_cast<int8_t*>(ws + L.ki);
  uint8_t* vt = reinterpret_cast<uint8_t*>(ws + L.vt);
  float* qs = reinterpret_cast<float*>(ws + L.qs);
  float* ks = reinterpret_cast<float*>(ws + L.ks);
  float* km = reinterpret_cast<float*>(ws + L.km);
  float* va = reinterpret_cast<float*>(ws + L.va);
  MX_CUDA(cudaMemsetAsync(km, 0, L.total - L.km, s));
  kv_stats_kernel<D><<<dim3(p.Hkv, unsigned((p.Skv + kStatTok - 1) / kStatTok)), D, 0, s>>>(k, v, p, km, va);
  quant_rows_kernel<D, 16><<<dim3(unsigned(L.sq / 16), p.Hq), 128, 0, s>>>(q, p.q_stride, p.Sq, L.sq, nullptr,
                                                                           p.scale * 1.4426950408889634f, qi, qs);
  quant_rows_kernel<D, kK><<<dim3(unsigned(L.skv / kK), p.Hkv), 128, 0, s>>>(k, p.kv_stride, p.Skv, L.skv, km, 1.f,
                                                                             ki, ks);
  v_quant_kernel<D><<<dim3(unsigned(L.skv / kK), p.Hkv), 256, 0, s>>>(v, p, L.skv, va, vt);
  const int smem = 2 * (kK * D + D * kK);
  fp8_attn_kernel<D><<<dim3(unsigned(L.sq / kQ), p.Hq), kW * 32, smem, s>>>(qi, ki, vt, qs, ks, va, o, p, L.sq,
                                                                            L.skv);
  MX_KCHECK();
}

bool fp8_ok(const bf16* q, const bf16* k, const bf16* v, const bf16* o, const AttnParams& p) {
  auto al = [](const void* x) { return (reinterpret_cast<uintptr_t>(x) & 15) == 0; };
  if (p.causal || (p.D != 64 && p.D != 128) || p.Sq <= 0 || p.Skv <= 0 || p.Hkv <= 0 || p.Hq % p.Hkv) return false;
  if (p.q_stride % 8 || p.kv_stride % 8 || p.o_stride % 2 || !al(q) || !al(k) || !al(v) || !al(o)) return false;
  if (p.Hq > 65535 || pad_to(p.Sq, kQ) / kQ > 0x7fffffff) return false;
  int dev = 0;
  MX_CUDA(cudaGetDevice(&dev));
  return flash_attn_fp8_supported(dev);
}

AttnParams with_defaults(AttnParams p) {
  if (!p.q_stride) p.q_stride = int64_t(p.Hq) * p.D;
  if (!p.kv_stride) p.kv_stride = int64_t(p.Hkv) * p.D;
  if (!p.o_stride) p.o_stride = int64_t(p.Hq) * p.D;
  if (p.scale == 0.f && p.D > 0) p.scale = 1.f / std::sqrt(float(p.D));
  return p;
}

}  // namespace

size_t flash_attn_fp8_workspace(const AttnParams& p) {
  return p.Sq > 0 && p.Skv > 0 && p.D > 0 ? layout(p).total : 0;
}

bool flash_attn_fp8_supported(int device) {
  int major = 0, minor = 0;
  MX_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MX_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return major * 10 + minor >= 89;
}

bool flash_attn_fp8_ok(const bf16* q, const bf16* k, const bf16* v, const bf16* o, const AttnParams& p) {
  return fp8_ok(q, k, v, o, with_defaults(p));
}

void flash_attn_fp8(const bf16* q, const bf16* k, const bf16* v, bf16* o, const AttnParams& p, void* ws,
                    cudaStream_t s) {
  const AttnParams a = with_defaults(p);
  if (!ws || !fp8_ok(q, k, v, o, a)) return flash_attn(q, k, v, o, p, s);
  MX_CHECK((reinterpret_cast<uintptr_t>(ws) & 255) == 0, "flash_attn_fp8: workspace must be 256-byte aligned");
  if (a.D == 64) run<64>(q, k, v, o, a, static_cast<char*>(ws), s);
  else run<128>(q, k, v, o, a, static_cast<char*>(ws), s);
}

}  // namespace mx
