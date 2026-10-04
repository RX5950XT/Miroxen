#pragma once
// 把任意 GGUF / safetensors 型別在 GPU 上轉成 bf16 或 f32。
#include <cstdint>

#include "core/tensor.h"

namespace mx {

// src 為裝置記憶體中 n 個元素的原始資料（n 必須是 gtype_block(t) 的倍數）。
// 支援：F32 F16 BF16 F8_E4M3 F8_E5M2 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS
// 不支援的型別丟例外。
void dequant_to_bf16(const void* src, GType t, int64_t n, bf16* dst, cudaStream_t s);
void dequant_to_f32(const void* src, GType t, int64_t n, float* dst, cudaStream_t s);
bool dequant_supported(GType t);

// CPU 參考實作（測試與小張量用）：dst 為 n 個 float。
void dequant_cpu(const void* src, GType t, int64_t n, float* dst);

}  // namespace mx
