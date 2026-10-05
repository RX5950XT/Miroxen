#include "kernels/blockscale.h"

#include <cublasLt.h>
#include <cuda_fp8.h>

namespace mx {

#define MX_LT(x) MX_BLAS(cublasStatus_t(x))

const char* qmode_name(QMode m) {
  return m == QMode::NVFP4 ? "fp4" : m == QMode::MXFP8 ? "fp8" : "bf16";
}

QMode parse_qmode(const std::string& s) {
  if (s == "fp4" || s == "nvfp4") return QMode::NVFP4;
  if (s == "fp8" || s == "mxfp8") return QMode::MXFP8;
  MX_CHECK(s == "bf16" || s.empty(), "unknown compute precision %s (use bf16 / fp8 / fp4)", s.c_str());
  return QMode::BF16;
}

bool qmode_supported(QMode m, int device) {
  if (m == QMode::BF16) return true;
  int major = 0;
  MX_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  return major >= 10;
}

namespace {

constexpr int vec_of(QMode m) { return m == QMode::NVFP4 ? 16 : 32; }
// NVFP4 每張量縮放：區塊縮放 = 區塊 absmax / 6 / g 一定 ≤ 448（e4m3 上限）。
constexpr float kFp4Range = 6.f * 448.f;

__host__ __device__ int64_t pad_to(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

// 縮放採 cuBLASLt 的 128×4 交錯排列：每 128 列 × 4 個縮放為一塊 512 位元組（見 quant_kernel 的 sfr）。

// float → e2m1 編碼（最近偶數捨入、飽和到 ±6）。__nv_cvt_float2_to_fp4x2 的硬體指令只在 sm_120a
// 這類架構專屬目標才有，通用 sm_120 會退回很慢的軟體模擬，所以自己比較門檻。
__device__ __forceinline__ uint32_t e2m1_code(float x) {
  const float a = fabsf(x);
  const uint32_t c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
  return c | (x < 0.f ? 8u : 0u);
}

// 每個 block 掃若干列（16 位元組向量讀取），block 內歸約後一次 atomicMax。
__global__ void amax_kernel(const bf16* x, int64_t ldx, int64_t rows, int64_t K, float* out) {
  float m = 0.f;
  const int64_t nv = K / 8;
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const uint4* src = reinterpret_cast<const uint4*>(x + r * ldx);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
      uint4 u = src[j];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int t = 0; t < 4; t++) {
        float2 f = __bfloat1622float2(h[t]);
        m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < int(blockDim.x / 32); w++) m = fmaxf(m, red[w]);
    atomicMax(reinterpret_cast<int*>(out), __float_as_int(m));  // 非負 float 可當 int 比
  }
}

// 每個 block 處理若干列，執行緒沿列內的區塊走：一個執行緒一個縮放區塊。
template <QMode M>
__global__ void quant_kernel(const bf16* x, int64_t ldx, int64_t rows, int64_t K, uint8_t* q, uint8_t* sf,
                             const float* amax) {
  constexpr int V = vec_of(M);
  const int nb = int(K / V), nkb_pad = (nb + 3) / 4 * 4;
  float g = 1.f, inv6g = 1.f;
  if constexpr (M == QMode::NVFP4) g = *amax > 0.f ? *amax / kFp4Range : 1.f, inv6g = 1.f / (6.f * g);
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const uint4* row = reinterpret_cast<const uint4*>(x + r * ldx);
    uint8_t* sfr = sf + (r / 128) * int64_t(nkb_pad) * 128 + (r % 32) * 16 + (r % 128) / 32 * 4;
    for (int kb = threadIdx.x; kb < nb; kb += blockDim.x) {
      float v[V];
      float bmax = 0.f;
#pragma unroll
      for (int j = 0; j < V / 8; j++) {
        uint4 u = row[kb * (V / 8) + j];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int t = 0; t < 4; t++) {
          float2 f = __bfloat1622float2(h[t]);
          v[j * 8 + t * 2] = f.x, v[j * 8 + t * 2 + 1] = f.y;
          bmax = fmaxf(bmax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
      }
      uint8_t sbyte;
      if constexpr (M == QMode::NVFP4) {
        const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(bmax * inv6g, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 sq;
        sq.__x = se;
        const float sd = float(sq) * g;
        const float inv = sd > 0.f ? __frcp_rn(sd) : 0.f;
        uint32_t w[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 16; j++) w[j / 8] |= e2m1_code(v[j] * inv) << (j % 8 * 4);  // 低半位元組在前
        *reinterpret_cast<uint2*>(q + (r * K + int64_t(kb) * V) / 2) = make_uint2(w[0], w[1]);
        sbyte = se;
      } else {
        // ue8m0：2 的冪次縮放，取最小的 2^e 使 bmax / 2^e ≤ 448。
        int e = -127;
        if (bmax > 0.f) {
          int ex;
          const float mant = frexpf(bmax / 448.f, &ex);  // bmax/448 = mant·2^ex，mant ∈ [0.5, 1)
          e = mant == 0.5f ? ex - 1 : ex;
          e = max(-127, min(127, e));
        }
        const float inv = exp2f(float(-e));
        alignas(16) uint8_t b[32];
#pragma unroll
        for (int j = 0; j < 32; j++) b[j] = __nv_cvt_float_to_fp8(v[j] * inv, __NV_SATFINITE, __NV_E4M3);
        uint4* dst = reinterpret_cast<uint4*>(q + r * K + int64_t(kb) * V);
        dst[0] = reinterpret_cast<uint4*>(b)[0], dst[1] = reinterpret_cast<uint4*>(b)[1];
        sbyte = uint8_t(e + 127);
      }
      sfr[(kb / 4) * 512 + kb % 4] = sbyte;
    }
  }
}

// alpha = gW · gX（NVFP4 兩邊的每張量縮放相乘）。
__global__ void alpha_kernel(const float* aw, const float* ax, float* alpha) {
  const float gw = *aw > 0.f ? *aw / kFp4Range : 1.f, gx = *ax > 0.f ? *ax / kFp4Range : 1.f;
  *alpha = gw * gx;
}

}  // namespace

size_t qdata_bytes(QMode m, int64_t rows, int64_t K) {
  return size_t(rows) * K / (m == QMode::NVFP4 ? 2 : 1);
}

size_t qscale_bytes(QMode m, int64_t rows, int64_t K) {
  return size_t(pad_to(rows, 128)) * pad_to(K / vec_of(m), 4);
}

void quantize_blockscaled(QMode m, const bf16* x, int64_t ldx, int64_t rows, int64_t K, const QMat& out,
                          cudaStream_t s) {
  MX_CHECK(m != QMode::BF16, "quantize_blockscaled: bf16 is not a quantized mode");
  MX_CHECK(K % 32 == 0 && ldx % 8 == 0 && uintptr_t(x) % 16 == 0, "quantize_blockscaled: need K%%32==0, ldx%%8==0, 16B-aligned input");
  if (rows == 0) return;
  // 補齊的縮放欄／列 cuBLASLt 也會讀（0xff 在 ue8m0 是 NaN），一律先清零。
  MX_CUDA(cudaMemsetAsync(out.sf, 0, qscale_bytes(m, rows, K), s));
  if (m == QMode::NVFP4) {
    MX_CUDA(cudaMemsetAsync(out.amax, 0, sizeof(float), s));
    amax_kernel<<<unsigned(std::min<int64_t>(rows, 1024)), 256, 0, s>>>(x, ldx, rows, K, out.amax);
    MX_KCHECK();
  }
  const unsigned grid = unsigned(std::min<int64_t>(rows, 65535));
  auto* q = static_cast<uint8_t*>(out.q);
  if (m == QMode::NVFP4) quant_kernel<QMode::NVFP4><<<grid, 128, 0, s>>>(x, ldx, rows, K, q, out.sf, out.amax);
  else quant_kernel<QMode::MXFP8><<<grid, 128, 0, s>>>(x, ldx, rows, K, q, out.sf, nullptr);
  MX_KCHECK();
}

// row-major Y = X·Wᵀ 等同 column-major Yᵀ[N,M] = W(op T)·Xᵀ，FP8/FP4 規定的 TN 排列。
void gemm_blockscaled(Ctx& c, QMode m, const QMat& W, const QMat& X, bf16* Y, int64_t ldy, int64_t M,
                      int64_t N, int64_t K, float beta) {
  if (M == 0 || N == 0) return;
  MX_CHECK(beta == 0.f || beta == 1.f, "gemm_blockscaled: beta must be 0 or 1");
  c.ensure_lt();
  float* scal = c.lt_scalars.as<float>();  // [alpha, 1, 0]
  const float* alpha = scal + 1;
  if (m == QMode::NVFP4) {
    alpha_kernel<<<1, 1, 0, c.stream>>>(W.amax, X.amax, scal);
    MX_KCHECK();
    alpha = scal;
  }
  const float* betap = beta == 0.f ? scal + 2 : scal + 1;
  const cudaDataType_t ab = m == QMode::NVFP4 ? CUDA_R_4F_E2M1 : CUDA_R_8F_E4M3;
  const int32_t smode = m == QMode::NVFP4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
                                          : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  cublasLtMatmulDesc_t op;
  MX_LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &W.sf, sizeof(void*)));
  MX_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &X.sf, sizeof(void*)));
  cublasLtMatrixLayout_t la, lb, ly;
  MX_LT(cublasLtMatrixLayoutCreate(&la, ab, uint64_t(K), uint64_t(N), K));
  MX_LT(cublasLtMatrixLayoutCreate(&lb, ab, uint64_t(K), uint64_t(M), K));
  MX_LT(cublasLtMatrixLayoutCreate(&ly, CUDA_R_16BF, uint64_t(N), uint64_t(M), ldy));
  cublasLtMatmulPreference_t pref;
  MX_LT(cublasLtMatmulPreferenceCreate(&pref));
  size_t wss = c.lt_ws.size();
  MX_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof(wss)));
  cublasLtMatmulHeuristicResult_t h{};
  int got = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(c.lt, op, la, lb, ly, ly, pref, 1, &h, &got);
  if (st == CUBLAS_STATUS_SUCCESS && got)
    st = cublasLtMatmul(c.lt, op, alpha, W.q, la, X.q, lb, betap, Y, ly, Y, ly, &h.algo, c.lt_ws.get(), wss,
                        c.stream);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(ly);
  cublasLtMatmulDescDestroy(op);
  MX_CHECK(got, "cuBLASLt has no %s GEMM algorithm (M=%lld N=%lld K=%lld)", qmode_name(m), (long long)M,
           (long long)N, (long long)K);
  MX_LT(st);
}

}  // namespace mx
