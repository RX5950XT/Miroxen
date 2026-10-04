#include "core/ctx.h"

namespace mx {

Ctx::Ctx(int dev, size_t act_bytes, size_t scratch_bytes) : device(dev) {
  MX_CUDA(cudaSetDevice(dev));
  // 必須是一般（阻塞）串流：程式各處用 cudaMemcpy（預設串流）讀回結果，
  // 非阻塞串流不會和預設串流同步，會讀到尚未算完的資料（曾導致 H3 少算最後一步）。
  MX_CUDA(cudaStreamCreate(&stream));
  MX_BLAS(cublasCreate(&blas));
  MX_BLAS(cublasSetStream(blas, stream));
  act = Arena(act_bytes, dev);
  scratch = DevBuf(scratch_bytes, dev);
}

Ctx::~Ctx() {
  if (blas) cublasDestroy(blas);
  if (stream) cudaStreamDestroy(stream);
}

// row-major Y = X W^T 等同 column-major Yᵀ[N,M] = W[N,K](op T) · Xᵀ[K,M]。
void gemm_bf16(Ctx& c, const bf16* X, int64_t ldx, const bf16* W, int64_t ldw, void* Y,
               bool y_f32, int64_t ldy, int64_t M, int64_t N, int64_t K, float beta,
               float alpha) {
  if (M == 0 || N == 0) return;
  MX_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX, "gemm 維度過大");
  MX_BLAS(cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, int(N), int(M), int(K), &alpha, W,
                       CUDA_R_16BF, int(ldw), X, CUDA_R_16BF, int(ldx), &beta, Y,
                       y_f32 ? CUDA_R_32F : CUDA_R_16BF, int(ldy), CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT));
}

void gemm_f32(Ctx& c, const float* X, int64_t ldx, const float* W, int64_t ldw, float* Y,
              int64_t ldy, int64_t M, int64_t N, int64_t K, float beta) {
  if (M == 0 || N == 0) return;
  float alpha = 1.f;
  MX_BLAS(cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, int(N), int(M), int(K), &alpha, W, int(ldw),
                      X, int(ldx), &beta, Y, int(ldy)));
}

}  // namespace mx
