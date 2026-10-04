#include "core/dequant.h"

#include <cstring>
#include <limits>
#include <type_traits>

namespace mx {
namespace {
// 位元排列參照 ggml (MIT)：ggml-common.h / ggml-quants.c。
#define HD __host__ __device__ __forceinline__
HD uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
HD uint32_t u32(const uint8_t* p) { return uint32_t(u16(p)) | uint32_t(u16(p + 2)) << 16; }
HD float bits_float(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}
HD float half_float(const uint8_t* p) { return __half2float(__ushort_as_half(u16(p))); }
HD float mul(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
HD float add(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
HD float sub(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}
HD float fp8(uint8_t v, bool e4) {
  const int mb = e4 ? 3 : 2, e = (v & 127) >> mb, m = v & ((1 << mb) - 1);
  const uint32_t sign = uint32_t(v & 128) << 24;
  if (e == 0) return (v & 128 ? -1.f : 1.f) * m * (e4 ? 0x1p-9f : 0x1p-16f);
  if (e4 && e == 15 && m == 7) return bits_float(sign | 0x7fc00000);
  if (!e4 && e == 31) return bits_float(sign | 0x7f800000 | uint32_t(m) << 21);
  return bits_float(sign | uint32_t(e + (e4 ? 120 : 112)) << 23 | uint32_t(m) << (23 - mb));
}
HD int iq4(int q) {
  const int8_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                            1, 13, 25, 38, 53, 69, 89, 113};
  return table[q];
}
HD void scale_min(const uint8_t* p, int j, int& s, int& m) {
  if (j < 4) { s = p[j] & 63; m = p[j + 4] & 63; }
  else { s = (p[j + 4] & 15) | (p[j - 4] >> 6) << 4;
         m = (p[j + 4] >> 4) | (p[j] >> 6) << 4; }
}

template<GType T> constexpr int block_elements() {
  if constexpr (T == GType::Q2_K || T == GType::Q3_K || T == GType::Q4_K ||
                T == GType::Q5_K || T == GType::Q6_K || T == GType::IQ4_XS) return 256;
  if constexpr (T == GType::Q4_0 || T == GType::Q4_1 || T == GType::Q5_0 ||
                T == GType::Q5_1 || T == GType::Q8_0 || T == GType::IQ4_NL) return 32;
  return 1;
}
template<GType T> constexpr int block_bytes() {
  if constexpr (T == GType::F32) return 4;
  if constexpr (T == GType::F16 || T == GType::BF16) return 2;
  if constexpr (T == GType::Q4_0 || T == GType::IQ4_NL) return 18;
  if constexpr (T == GType::Q4_1) return 20;
  if constexpr (T == GType::Q5_0) return 22;
  if constexpr (T == GType::Q5_1) return 24;
  if constexpr (T == GType::Q8_0) return 34;
  if constexpr (T == GType::Q2_K) return 84;
  if constexpr (T == GType::Q3_K) return 110;
  if constexpr (T == GType::Q4_K) return 144;
  if constexpr (T == GType::Q5_K) return 176;
  if constexpr (T == GType::Q6_K) return 210;
  if constexpr (T == GType::IQ4_XS) return 136;
  return 1;
}

template<GType T> HD float decode(const uint8_t* p, int k) {
  if constexpr (T == GType::F32) return bits_float(u32(p));
  if constexpr (T == GType::F16) return half_float(p);
  if constexpr (T == GType::BF16) return bits_float(uint32_t(u16(p)) << 16);
  if constexpr (T == GType::F8_E4M3 || T == GType::F8_E5M2) return fp8(*p, T == GType::F8_E4M3);
  if constexpr (T == GType::Q8_0) return mul(half_float(p), int8_t(p[2 + k]));
  if constexpr (T == GType::Q4_0 || T == GType::Q4_1 || T == GType::Q5_0 ||
                T == GType::Q5_1 || T == GType::IQ4_NL) {
    constexpr bool affine = T == GType::Q4_1 || T == GType::Q5_1;
    constexpr bool five = T == GType::Q5_0 || T == GType::Q5_1;
    constexpr int off = 2 + (affine ? 2 : 0);
    int q = (p[off + (five ? 4 : 0) + k % 16] >> (k / 16 * 4)) & 15;
    if constexpr (five) q |= ((u32(p + off) >> k) & 1) << 4;
    if constexpr (T == GType::IQ4_NL) return mul(half_float(p), iq4(q));
    if constexpr (affine) return add(mul(half_float(p), q), half_float(p + 2));
    return mul(half_float(p), q - (five ? 16 : 8));
  }
  if constexpr (T == GType::Q2_K) {
    const int sc = p[k / 16], q = (p[16 + k / 128 * 32 + k % 32] >> (k % 128 / 32 * 2)) & 3;
    return sub(mul(mul(half_float(p + 80), sc & 15), q), mul(half_float(p + 82), sc >> 4));
  }
  if constexpr (T == GType::Q3_K) {
    const int j = k / 16;
    const int sc = ((p[96 + j % 8] >> (j / 8 * 4)) & 15) |
                   (((p[104 + j % 4] >> (j / 4 * 2)) & 3) << 4);
    const int lo = (p[32 + k / 128 * 32 + k % 32] >> (k % 128 / 32 * 2)) & 3;
    const int q = lo - ((p[k % 32] & (1 << (k / 32))) ? 0 : 4);
    return mul(mul(half_float(p + 108), sc - 32), q);
  }
  if constexpr (T == GType::Q4_K || T == GType::Q5_K) {
    int sc, mn;
    scale_min(p + 4, k / 32, sc, mn);
    constexpr int off = T == GType::Q4_K ? 16 : 48;
    int q = (p[off + k / 64 * 32 + k % 32] >> (k % 64 / 32 * 4)) & 15;
    if constexpr (T == GType::Q5_K) q |= ((p[16 + k % 32] >> (k / 32)) & 1) << 4;
    return sub(mul(mul(half_float(p), sc), q), mul(half_float(p + 2), mn));
  }
  if constexpr (T == GType::Q6_K) {
    const int j = k % 128 / 32, l = k % 32;
    const int lo = (p[k / 128 * 64 + (j % 2) * 32 + l] >> (j / 2 * 4)) & 15;
    const int hi = (p[128 + k / 128 * 32 + l] >> (j * 2)) & 3;
    return mul(mul(half_float(p + 208), int8_t(p[192 + k / 16])), (lo | hi << 4) - 32);
  }
  if constexpr (T == GType::IQ4_XS) {
    const int j = k / 32;
    const int sc = ((p[4 + j / 2] >> (j % 2 * 4)) & 15) | ((u16(p + 2) >> (j * 2)) & 3) << 4;
    const int q = (p[8 + j * 16 + k % 16] >> (k % 32 / 16 * 4)) & 15;
    return mul(mul(half_float(p), sc - 32), iq4(q));
  }
  return 0;
}

// 每個 CTA 處理 8 個 super-block；連續寫入，bf16 用成對的 32-bit store。
template<GType T, class Out> __global__ void convert(const uint8_t* src, int64_t n, Out* dst, bool aligned) {
  constexpr int B = block_elements<T>(), S = block_bytes<T>();
  const int64_t start = int64_t(blockIdx.x) * 2048;
  for (int step = 0; step < 4; ++step) {
    const int64_t i = start + step * 512 + threadIdx.x * 2;
    if (i >= n) continue;
    const float a = decode<T>(src + i / B * S, int(i % B));
    const float b = i + 1 < n ? decode<T>(src + (i + 1) / B * S, int((i + 1) % B)) : 0;
    if constexpr (std::is_same_v<Out, bf16>) {
      if (i + 1 < n && aligned) reinterpret_cast<__nv_bfloat162*>(dst)[i / 2] = __floats2bfloat162_rn(a, b);
      else { dst[i] = __float2bfloat16_rn(a); if (i + 1 < n) dst[i + 1] = __float2bfloat16_rn(b); }
    } else {
      if (i + 1 < n && aligned) reinterpret_cast<float2*>(dst)[i / 2] = make_float2(a, b);
      else { dst[i] = a; if (i + 1 < n) dst[i + 1] = b; }
    }
  }
}

#define MX_DEQUANT_TYPES(X) \
  X(F32) X(F16) X(BF16) X(F8_E4M3) X(F8_E5M2) X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) \
  X(Q8_0) X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K) X(IQ4_NL) X(IQ4_XS)

void validate(const void* src, GType t, int64_t n, const void* dst) {
  MX_CHECK(dequant_supported(t), "Unsupported dequantization type %u", unsigned(t));
  MX_CHECK(n >= 0 && n % gtype_block(t) == 0, "Invalid dequantization element count: %lld", (long long)n);
  MX_CHECK(n <= int64_t(std::numeric_limits<size_t>::max() / sizeof(float)), "Dequantization size overflow");
  MX_CHECK(n == 0 || (src && dst), "Null dequantization source or destination");
}
template<class Out> void launch(const void* src, GType t, int64_t n, Out* dst, cudaStream_t s) {
  validate(src, t, n, dst);
  if (!n) return;
  const int64_t grid = (n + 2047) / 2048;
  MX_CHECK(grid <= 2147483647, "Dequantization exceeds CUDA grid limit");
  switch (t) {
#define CASE(T) case GType::T: convert<GType::T><<<unsigned(grid), 256, 0, s>>>(static_cast<const uint8_t*>(src), n, dst, uintptr_t(dst) % (2 * sizeof(Out)) == 0); break;
    MX_DEQUANT_TYPES(CASE)
#undef CASE
    default: break;  // validate 已拒絕未知型別。
  }
  MX_KCHECK();
}
template<GType T> void cpu(const uint8_t* src, int64_t n, float* dst) {
  constexpr int B = block_elements<T>(), S = block_bytes<T>();
  for (int64_t b = 0; b < n / B; ++b)
    for (int k = 0; k < B; ++k) dst[b * B + k] = decode<T>(src + b * S, k);
}
}  // namespace

bool dequant_supported(GType t) {
  switch (t) {
#define CASE(T) case GType::T:
    MX_DEQUANT_TYPES(CASE)
#undef CASE
      return true;
    default: return false;
  }
}
void dequant_to_bf16(const void* src, GType t, int64_t n, bf16* dst, cudaStream_t s) { launch(src, t, n, dst, s); }
void dequant_to_f32(const void* src, GType t, int64_t n, float* dst, cudaStream_t s) { launch(src, t, n, dst, s); }
void dequant_cpu(const void* src, GType t, int64_t n, float* dst) {
  validate(src, t, n, dst);
  switch (t) {
#define CASE(T) case GType::T: cpu<GType::T>(static_cast<const uint8_t*>(src), n, dst); break;
    MX_DEQUANT_TYPES(CASE)
#undef CASE
    default: break;
  }
}
}  // namespace mx
