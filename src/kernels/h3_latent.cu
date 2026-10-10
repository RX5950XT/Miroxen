#include "kernels/h3_latent.h"

namespace mx::h3_latent {
namespace {
__global__ void video_kernel(const float* rows, float* z, int64_t f, int64_t h, int64_t w) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 24 * f * h * w) return;
  int64_t x = i % w, y = i / w % h, t = i / (w * h) % f, c = i / (w * h * f);
  int64_t row = (t * (h / 2) + y / 2) * (w / 2) + x / 2;
  z[i] = rows[row * 96 + c * 4 + (y % 2) * 2 + x % 2];
}
__global__ void audio_kernel(const float* rows, float* z, int64_t a) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 2 * 32 * a) return;
  int64_t t = i % a, c = i / a % 32, ch = i / (32 * a);
  z[i] = rows[(ch * a + t) * 32 + c];
}
}  // namespace

void video(const float* rows, float* z, int64_t f, int64_t h, int64_t w, cudaStream_t stream) {
  MX_CHECK(rows && z && f > 0 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0,
           "Invalid H3 video latent arguments");
  video_kernel<<<unsigned((24 * f * h * w + 255) / 256), 256, 0, stream>>>(rows, z, f, h, w);
  MX_KCHECK();
}
void audio(const float* rows, float* z, int64_t a, cudaStream_t stream) {
  MX_CHECK(rows && z && a > 0, "Invalid H3 audio latent arguments");
  audio_kernel<<<unsigned((64 * a + 255) / 256), 256, 0, stream>>>(rows, z, a);
  MX_KCHECK();
}
}  // namespace mx::h3_latent
