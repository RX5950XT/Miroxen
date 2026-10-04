#pragma once
// 全專案共用：錯誤檢查、日誌、bf16 型別。
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mx {

using bf16 = __nv_bfloat16;
using f16 = __half;

inline std::string strf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char buf[2048];
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

// 等級：0=錯誤 1=資訊 2=除錯。由環境變數 MX_LOG 控制，預設 1。
int log_level();
void log_msg(int level, const char* fmt, ...);
#define MX_INFO(...) ::mx::log_msg(1, __VA_ARGS__)
#define MX_DEBUG(...) ::mx::log_msg(2, __VA_ARGS__)
#define MX_ERROR(...) ::mx::log_msg(0, __VA_ARGS__)

#define MX_CUDA(x)                                                                                 \
  do {                                                                                             \
    cudaError_t mx_e_ = (x);                                                                       \
    if (mx_e_ != cudaSuccess)                                                                      \
      throw std::runtime_error(                                                                    \
          ::mx::strf("%s:%d CUDA %s: %s", __FILE__, __LINE__, #x, cudaGetErrorString(mx_e_)));     \
  } while (0)

#define MX_CHECK(cond, ...)                                                                        \
  do {                                                                                             \
    if (!(cond)) throw std::runtime_error(::mx::strf(__VA_ARGS__));                                \
  } while (0)

// 核心呼叫後檢查啟動錯誤（不同步）。
#define MX_KCHECK() MX_CUDA(cudaGetLastError())

}  // namespace mx
