#pragma once
// 單一 GPU 的執行環境：stream、cuBLAS、啟用值 Arena、權重反量化暫存。
#include <cublasLt.h>
#include <cublas_v2.h>

#include <memory>

#include "core/tensor.h"

namespace mx {

#define MX_BLAS(x)                                                                                 \
  do {                                                                                             \
    cublasStatus_t mx_s_ = (x);                                                                    \
    if (mx_s_ != CUBLAS_STATUS_SUCCESS)                                                            \
      throw std::runtime_error(::mx::strf("%s:%d cuBLAS %s: %d", __FILE__, __LINE__, #x,           \
                                          int(mx_s_)));                                            \
  } while (0)

class Ctx {
 public:
  // act_bytes：啟用值 Arena 大小；scratch_bytes：權重反量化暫存（至少放得下最大一層的 bf16）。
  Ctx(int device, size_t act_bytes, size_t scratch_bytes);
  ~Ctx();
  Ctx(const Ctx&) = delete;
  Ctx& operator=(const Ctx&) = delete;

  int device;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  Arena act;
  DevBuf scratch;
  // 區塊縮放 GEMM（kernels/blockscale）用；第一次用到才建立。
  cublasLtHandle_t lt = nullptr;
  DevBuf lt_ws, lt_scalars;
  void ensure_lt();

  void sync() const { MX_CUDA(cudaStreamSynchronize(stream)); }
  // 切換目前執行緒的裝置到本 Ctx。
  void use() const { MX_CUDA(cudaSetDevice(device)); }
};

// Y[M,N] = X[M,K] · W[N,K]^T（row-major）。Y 可為 bf16 或 f32；beta=1 時累加到 Y。
void gemm_bf16(Ctx& c, const bf16* X, int64_t ldx, const bf16* W, int64_t ldw, void* Y,
               bool y_f32, int64_t ldy, int64_t M, int64_t N, int64_t K, float beta = 0.f,
               float alpha = 1.f);
void gemm_f32(Ctx& c, const float* X, int64_t ldx, const float* W, int64_t ldw, float* Y,
              int64_t ldy, int64_t M, int64_t N, int64_t K, float beta = 0.f, float alpha = 1.f);

}  // namespace mx
