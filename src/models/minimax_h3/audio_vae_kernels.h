#pragma once
// 音訊 VAE 私有 fp32 核心；張量採 [channels, time]。
#include "core/common.h"
namespace mx::h3audio {
void im2col(const float*, float*, int, int, int, int, int, int, int, cudaStream_t);
void col2im(const float*, float*, const float*, int, int, int, int, int, int, cudaStream_t);
void bias(float*, const float*, int, int, cudaStream_t);
void snake(const float*, float*, const float*, int, int, cudaStream_t);
void alias_up(const float*, float*, const float*, const float*, const float*, int, int, int, cudaStream_t);
void alias_down(const float*, float*, const float*, int, int, int, cudaStream_t);
void transpose(const float*, float*, int, int, cudaStream_t);
void norm(const float*, float*, const float*, const float*, int, int, cudaStream_t);
void geglu(float*, const float*, int64_t, cudaStream_t);
void softmax(float*, int, int, cudaStream_t);
void pool(const float*, float*, int, int, int, int, cudaStream_t);
void normalize(const float*, float*, const float*, const float*, int, bool, cudaStream_t);
}
