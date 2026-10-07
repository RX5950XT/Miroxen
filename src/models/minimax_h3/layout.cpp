#include <algorithm>
#include <array>
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

// 文字列依模態切段（視覺區塊用影片的 AdaLN），timestep 跟目標影片一樣（索引 0）。
void text_segments(int64_t T, const std::vector<H3Tag>& tags, std::vector<H3Segment>& segs) {
  MX_CHECK(tags.empty() || int64_t(tags.size()) == T, "text_tags 長度不符");
  for (int64_t i = 0; i < T;) {
    H3Tag g = tags.empty() ? H3_TEXT : tags[i];
    int64_t j = i + 1;
    while (j < T && (tags.empty() ? H3_TEXT : tags[j]) == g) j++;
    segs.push_back({i, j - i, 0, g});
    i = j;
  }
}

// 一個 latent 幀的 (h, w) 旋轉座標（以面積正規化、置中、乘 32），列優先。
struct FrameGrid {
  std::vector<double> h, w;
  int64_t rows() const { return int64_t(h.size() * w.size()); }
};
FrameGrid frame_grid(int64_t H, int64_t W) {
  const double sa = std::sqrt(double(H * W));
  return {spatial_grid(H, sa), spatial_grid(W, sa)};
}
}  // namespace

int h3_align_frames(int f) {
  MX_CHECK(f >= 1, "影格數必須 >= 1");
  f = std::max(f, 22);  // 5 幀只有 2 個 latent，解碼器需要至少 3 個；第一個有效組為 22。
  while (f % 17 != 5) f++;
  return f;
}
int64_t h3_latent_frames(int f) { return (f - 5) / 17 * 5 + 2; }
int64_t h3_audio_latents(int f) { return int64_t(std::llround(double(f) / 24.0 * 40.0)); }

void h3_canvas(double aw, double ah, int short_edge, int64_t max_pixels, int& w, int& h) {
  MX_CHECK(aw > 0 && ah > 0, "長寬比必須為正");
  const double ratio = aw / ah;
  MX_CHECK(ratio >= 0.25 && ratio <= 4.0, "MiniMax-H3 只支援 1:4 到 4:1 的長寬比（%.3f）", ratio);
  double fw = ratio >= 1 ? short_edge * ratio : short_edge, fh = ratio >= 1 ? short_edge : short_edge / ratio;
  if (fw * fh > double(max_pixels)) {
    double s = std::sqrt(double(max_pixels) / (fw * fh));
    fw *= s, fh *= s;
  }
  h = std::max(32, int(std::nearbyint(fh / 32)) * 32);
  w = std::max(32, int(std::nearbyint(fw / 32)) * 32);
}

H3Layout h3_build_layout(int64_t T, int64_t F_lat, int64_t H, int64_t W, int64_t A,
                         const std::vector<std::string>& anchors, const std::vector<H3Tag>& text_tags) {
  MX_CHECK(text_tags.empty() || int64_t(text_tags.size()) == T, "text_tags 長度不符");
  MX_CHECK(H % 2 == 0 && W % 2 == 0, "latent 尺寸必須是偶數");
  H3Layout L;
  L.T = T, L.F_lat = F_lat, L.H = H, L.W = W, L.A = A;
  L.cond_frames = int(anchors.size());
  const int64_t rpf = (H / 2) * (W / 2);
  const int64_t cond = int64_t(anchors.size()) * rpf, audio = kAudioChannels * A, video = F_lat * rpf;
  H3Seq& s = L.seq;
  s.S = T + cond + audio + video;
  s.text_start = 0, s.text_len = T;
  if (cond) s.vref.push_back({T, cond});
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
  text_segments(T, text_tags, s.segs);
  if (cond) s.segs.push_back({T, cond, 2, H3_VIDEO});
  s.segs.push_back({s.audio_start, audio, 1, H3_AUDIO});
  s.segs.push_back({s.video_start, video, 0, H3_VIDEO});
  return L;
}

H3Layout h3_build_ref2va_layout(int64_t T, const std::vector<H3Tag>& text_tags, const std::vector<H3RefGeom>& refs,
                                int64_t F_lat, int64_t H, int64_t W, int64_t A) {
  MX_CHECK(H % 2 == 0 && W % 2 == 0, "latent 尺寸必須是偶數");
  H3Layout L;
  L.T = T, L.F_lat = F_lat, L.H = H, L.W = W, L.A = A;
  H3Seq& s = L.seq;
  s.text_start = 0, s.text_len = T;
  text_segments(T, text_tags, s.segs);
  std::vector<std::array<double, 3>> pos(T);
  for (int64_t i = 0; i < T; i++) pos[i] = {double(i), 0, 0};
  auto video_rows = [&](const FrameGrid& g, int64_t frames, double t0, bool image) {
    double t = t0;
    for (int64_t f = 0; f < frames; f++) {
      for (double h : g.h)
        for (double w : g.w) pos.push_back({t, h, w});
      if (!image) t += kFrameRescale * kFramesPerLatent[f % 5];
    }
  };
  auto audio_rows = [&](int64_t n, double t0, const FrameGrid& g) {
    for (int64_t c = 0; c < kAudioChannels; c++)
      for (int64_t a = 0; a < n; a++) pos.push_back({t0 + double(a), 0, c == 0 ? g.w.front() : g.w.back()});
  };
  auto add_block = [&](std::vector<H3Block>& dst, int64_t len, int t_idx, H3Tag tag) {
    int64_t start = int64_t(pos.size()) - len;
    dst.push_back({start, len});
    s.segs.push_back({start, len, t_idx, tag});
  };
  const FrameGrid target = frame_grid(H, W);
  double clock = double(T);
  for (auto& r : refs) {
    if (r.kind == H3RefGeom::AUDIO) {
      MX_CHECK(r.audio > 0, "參考音訊沒有內容");
      audio_rows(r.audio, clock, target);
      add_block(s.aref, kAudioChannels * r.audio, 3, H3_AUDIO);
      clock += double(r.audio);
      continue;
    }
    const FrameGrid g = frame_grid(r.h, r.w);
    if (r.kind == H3RefGeom::IMAGE) {  // 單張圖只佔一個整數時間格
      video_rows(g, 1, clock, true);
      add_block(s.vref, g.rows(), 2, H3_VIDEO);
      clock += 1.0;
      continue;
    }
    // 影片：音軌列在影片列之前、共用起點；時間往前推兩者較長者。
    if (r.audio > 0) {
      audio_rows(r.audio, clock, g);
      add_block(s.aref, kAudioChannels * r.audio, 3, H3_AUDIO);
    }
    video_rows(g, r.frames, clock, false);
    add_block(s.vref, r.frames * g.rows(), 2, H3_VIDEO);
    double span = 0;
    for (int64_t f = 0; f < r.frames; f++) span += kFrameRescale * kFramesPerLatent[f % 5];
    clock += std::max(double(r.audio), span);
  }
  audio_rows(A, clock, target);
  s.audio_start = int64_t(pos.size()) - kAudioChannels * A, s.audio_len = kAudioChannels * A;
  s.segs.push_back({s.audio_start, s.audio_len, 1, H3_AUDIO});
  video_rows(target, F_lat, clock, false);
  s.video_len = F_lat * target.rows(), s.video_start = int64_t(pos.size()) - s.video_len;
  s.segs.push_back({s.video_start, s.video_len, 0, H3_VIDEO});
  s.S = int64_t(pos.size());
  L.positions.resize(s.S * 3);
  for (int64_t i = 0; i < s.S; i++)
    for (int a = 0; a < 3; a++) L.positions[i * 3 + a] = float(pos[i][a]);
  return L;
}

}  // namespace mx
