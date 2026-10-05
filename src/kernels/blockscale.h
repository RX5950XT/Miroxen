#pragma once
// 區塊縮放低精度 GEMM（Blackwell tensor core，經 cuBLASLt）。RTX 5060 Ti 實測（M8192 N12288 K4096）：
//   NVFP4：e2m1，每 16 個一個 e4m3 縮放 + 每張量 f32 縮放，293 TFLOPS（bf16 的 7 倍）
//   MXFP8：e4m3，每 32 個一個 2 的冪次縮放（ue8m0），152 TFLOPS（3.7 倍）
// 權重載入時量化一次，啟用值每次呼叫即時量化。需要 sm_100 以上（3060 Ti 不支援，維持 bf16）。
#include <cstdint>
#include <string>

#include "core/ctx.h"

namespace mx {

enum class QMode { BF16, MXFP8, NVFP4 };
const char* qmode_name(QMode m);
QMode parse_qmode(const std::string& s);  // "bf16" / "fp8" / "fp4"
// 這張卡能不能跑 m（不能就該退回 BF16）。
bool qmode_supported(QMode m, int device);

// 量化後的矩陣 [rows, K]（列優先、緊密）：資料、cuBLASLt 128×4 交錯排列的區塊縮放、
// NVFP4 另有整張量 absmax（裝置端 float，用來推每張量縮放）。
struct QMat {
  void* q = nullptr;
  uint8_t* sf = nullptr;
  float* amax = nullptr;
};

size_t qdata_bytes(QMode m, int64_t rows, int64_t K);
size_t qscale_bytes(QMode m, int64_t rows, int64_t K);
// K 必須是 32 的倍數；x 的列距 ldx 必須是 8 的倍數、位址 16 位元組對齊。
void quantize_blockscaled(QMode m, const bf16* x, int64_t ldx, int64_t rows, int64_t K, const QMat& out,
                          cudaStream_t s);

// Y[M,N]（bf16，列距 ldy）= X[M,K] · W[N,K]ᵀ，兩者都已用同一 QMode 量化；beta 只能是 0 或 1。
void gemm_blockscaled(Ctx& c, QMode m, const QMat& W, const QMat& X, bf16* Y, int64_t ldy, int64_t M,
                      int64_t N, int64_t K, float beta = 0.f);

}  // namespace mx
