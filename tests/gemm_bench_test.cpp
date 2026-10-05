// 量測各 tensor core 路徑的 GEMM 吞吐量（cuBLASLt），決定加速策略。只印數字、永遠通過。
// MX_GEMM_BENCH=0 時略過（讓 ctest 快一點）。
#include <cublasLt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/ctx.h"

using namespace mx;

#define LT(x) MX_CHECK((x) == CUBLAS_STATUS_SUCCESS, "cublasLt 失敗：%s（%d）", #x, int(x))

static double bench(cublasLtHandle_t lt, cudaStream_t s, cudaDataType_t ab, cudaDataType_t cd,
                    cublasComputeType_t comp, cudaDataType_t scale, int M, int N, int K, void* A, void* B,
                    void* C, void* ws, size_t wss, int scale_mode = -1, void* sa = nullptr, void* sb = nullptr) {
  cublasLtMatmulDesc_t op;
  LT(cublasLtMatmulDescCreate(&op, comp, scale));
  if (scale_mode >= 0) {
    int32_t m = scale_mode;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof(m)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof(m)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
  }
  cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
  cublasLtMatrixLayout_t la, lb, lc;
  LT(cublasLtMatrixLayoutCreate(&la, ab, K, N, K));  // W^T：W 為 [N,K] row-major
  LT(cublasLtMatrixLayoutCreate(&lb, ab, K, M, K));  // X 為 [M,K] row-major
  LT(cublasLtMatrixLayoutCreate(&lc, cd, N, M, N));
  cublasLtMatmulPreference_t pref;
  LT(cublasLtMatmulPreferenceCreate(&pref));
  LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof(wss)));
  cublasLtMatmulHeuristicResult_t h;
  int got = 0;
  double tflops = -1;
  if (cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &h, &got) == CUBLAS_STATUS_SUCCESS && got) {
    int32_t ia = 1, ib = 0;
    float fa = 1, fb = 0;
    __half ha = __float2half(1.f), hb = __float2half(0.f);
    const void* alpha = scale == CUDA_R_32I ? (void*)&ia : scale == CUDA_R_16F ? (void*)&ha : (void*)&fa;
    const void* beta = scale == CUDA_R_32I ? (void*)&ib : scale == CUDA_R_16F ? (void*)&hb : (void*)&fb;
    auto run = [&] { LT(cublasLtMatmul(lt, op, alpha, A, la, B, lb, beta, C, lc, C, lc, &h.algo, ws, wss, s)); };
    for (int i = 0; i < 3; i++) run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    const int it = 20;
    for (int i = 0; i < it; i++) run();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    tflops = 2.0 * M * N * K * it / (ms * 1e9);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  return tflops;
}

int main() {
  const char* e = std::getenv("MX_GEMM_BENCH");
  if (e && std::atoi(e) == 0) return 0;
  try {
    int n = 0;
    MX_CUDA(cudaGetDeviceCount(&n));
    for (int d = 0; d < n; d++) {
      MX_CUDA(cudaSetDevice(d));
      cudaDeviceProp p;
      MX_CUDA(cudaGetDeviceProperties(&p, d));
      cublasLtHandle_t lt;
      LT(cublasLtCreate(&lt));
      cudaStream_t s;
      MX_CUDA(cudaStreamCreate(&s));
      const int M = 8192, N = 12288, K = 4096;
      DevBuf A(size_t(N) * K * 2, d), B(size_t(M) * K * 2, d), C(size_t(M) * N * 4, d), ws(size_t(64) << 20, d);
      MX_CUDA(cudaMemset(A.get(), 0, A.size()));
      MX_CUDA(cudaMemset(B.get(), 0, B.size()));
      struct Case { const char* name; cudaDataType_t ab, cd; cublasComputeType_t comp; cudaDataType_t sc; };
      std::vector<Case> cases = {
          {"BF16 in, FP32 acc   ", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F},
          {"FP16 in, FP32 acc   ", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F},
          {"FP16 in, FP16 acc   ", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F},
          {"INT8 in, INT32 acc  ", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I},
          {"FP8 e4m3, FP32 acc  ", CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F},
      };
      std::printf("GPU%d %s (sm_%d%d)  M=%d N=%d K=%d\n", d, p.name, p.major, p.minor, M, N, K);
      for (auto& c : cases) {
        double t = bench(lt, s, c.ab, c.cd, c.comp, c.sc, M, N, K, A.get(), B.get(), C.get(), ws.get(), ws.size());
        if (t < 0) std::printf("  %s 不支援\n", c.name);
        else std::printf("  %s %7.1f T(FL)OPS\n", c.name, t);
      }
      if (p.major >= 9) {
        DevBuf sa(size_t(N) * K / 64 * 4 + 4096, d), sb(size_t(M) * K / 64 * 4 + 4096, d);
        MX_CUDA(cudaMemset(sa.get(), 0, sa.size()));
        MX_CUDA(cudaMemset(sb.get(), 0, sb.size()));
        struct SM { const char* name; int mode; cudaDataType_t ab; };
        for (SM sm : {SM{"FP8 外積縮放(OUTER_VEC)", 3, CUDA_R_8F_E4M3}, SM{"FP8 1x128 區塊縮放     ", 4, CUDA_R_8F_E4M3},
                      SM{"FP8 128x128 區塊縮放   ", 5, CUDA_R_8F_E4M3}, SM{"MXFP8 (VEC32_UE8M0)   ", 2, CUDA_R_8F_E4M3},
                      SM{"NVFP4 (VEC16_UE4M3)   ", 1, CUDA_R_4F_E2M1}}) {
          double tf = bench(lt, s, sm.ab, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, M, N, K, A.get(), B.get(),
                            C.get(), ws.get(), ws.size(), sm.mode, sa.get(), sb.get());
          if (tf < 0) std::printf("  %s 不支援\n", sm.name);
          else std::printf("  %s %7.1f TFLOPS\n", sm.name, tf);
        }
      }
      cudaStreamDestroy(s);
      cublasLtDestroy(lt);
    }
  } catch (const std::exception& ex) {
    std::printf("bench 錯誤：%s\n", ex.what());
  }
  return 0;
}
