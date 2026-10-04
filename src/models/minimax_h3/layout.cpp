#include <cmath>

#include "models/minimax_h3/pipeline.h"

namespace mx {

namespace {
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int kFramesPerLatent[5] = {1, 4, 4, 4, 4};
constexpr double kSpatialScale = 32.0;
constexpr int kAudioChannels = 2;

std::vector<double> spatial_grid(int64_t dim, double sqrt_area) {
  double ratio = double(dim) / sqrt_area, left = (1.0 - ratio) / 2.0;
  int64_t n = dim / 2;
  std::vector<double> g(n);
  for (int64_t k = 0; k < n; k++) g[k] = (left + ratio * double(k) / double(n)) * kSpatialScale;
  return g;
}
}  // namespace

int h3_align_frames(int f) {
  MX_CHECK(f >= 1, "影格數必須 >= 1");
  while (f % 17 != 5) f++;
  return f;
}
int64_t h3_latent_frames(int f) { return (f - 5) / 17 * 5 + 2; }
int64_t h3_audio_latents(int f) { return int64_t(std::llround(double(f) / 24.0 * 40.0)); }

H3Layout h3_build_layout(int64_t T, int64_t F_lat, int64_t H, int64_t W, int64_t A,
                         const std::vector<std::string>& anchors) {
  MX_CHECK(H % 2 == 0 && W % 2 == 0, "latent 尺寸必須是偶數");
  H3Layout L;
  L.T = T, L.F_lat = F_lat, L.H = H, L.W = W, L.A = A;
  L.cond_frames = int(anchors.size());
  const int64_t rpf = (H / 2) * (W / 2);
  const int64_t cond = int64_t(anchors.size()) * rpf, audio = kAudioChannels * A, video = F_lat * rpf;
  H3Seq& s = L.seq;
  s.S = T + cond + audio + video;
  s.text_start = 0, s.text_len = T;
  s.cond_start = T, s.cond_len = cond;
  s.audio_start = T + cond, s.audio_len = audio;
  s.video_start = T + cond + audio, s.video_len = video;
  L.positions.assign(s.S * 3, 0.f);
  auto set = [&](int64_t r, double t, double h, double w) {
    L.positions[r * 3] = float(t), L.positions[r * 3 + 1] = float(h), L.positions[r * 3 + 2] = float(w);
  };
  for (int64_t i = 0; i < T; i++) set(i, double(i), 0, 0);
  const double sa = std::sqrt(double(H * W));
  auto hg = spatial_grid(H, sa), wg = spatial_grid(W, sa);
  std::vector<double> ft(F_lat);  // 每個 latent 幀的時間座標
  double acc = double(T), total = 0;
  for (int64_t f = 0; f < F_lat; f++) {
    ft[f] = acc;
    double span = kFrameRescale * kFramesPerLatent[f % 5];
    acc += span;
    total += span;
  }
  auto frame_rows = [&](int64_t r0, double t) {
    for (int64_t y = 0; y < H / 2; y++)
      for (int64_t x = 0; x < W / 2; x++) set(r0 + y * (W / 2) + x, t, hg[y], wg[x]);
  };
  for (size_t i = 0; i < anchors.size(); i++) {
    MX_CHECK(anchors[i] == "first" || anchors[i] == "last", "anchor 必須是 first 或 last");
    double t = anchors[i] == "first" ? double(T) : double(T) + total - kFrameRescale;
    frame_rows(T + int64_t(i) * rpf, t);
  }
  for (int64_t c = 0; c < kAudioChannels; c++)
    for (int64_t a = 0; a < A; a++)
      set(s.audio_start + c * A + a, double(T) + double(a), 0, c == 0 ? wg.front() : wg.back());
  for (int64_t f = 0; f < F_lat; f++) frame_rows(s.video_start + f * rpf, ft[f]);
  // 段：t_idx 0 = 影片 t、1 = 音訊 t、2 = 條件幀 t。
  s.segs.clear();
  if (T) s.segs.push_back({0, T, 0, H3_TEXT});
  if (cond) s.segs.push_back({T, cond, 2, H3_VIDEO});
  s.segs.push_back({s.audio_start, audio, 1, H3_AUDIO});
  s.segs.push_back({s.video_start, video, 0, H3_VIDEO});
  return L;
}

}  // namespace mx
