#include "server/image_ops.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#include "core/common.h"
#include "server/image_io.h"

namespace mx {

Rgba load_rgba(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  MX_CHECK(static_cast<bool>(f), "無法讀取圖片 %s", path.c_str());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  DecodedImage d = decode_image_bytes(bytes.data(), bytes.size());
  Rgba r;
  r.w = d.width, r.h = d.height;
  r.px.resize(size_t(r.w) * r.h * 4);
  for (size_t i = 0; i < size_t(r.w) * r.h; i++) {
    const uint8_t* s = &d.pixels[i * d.channels];
    uint8_t* o = &r.px[i * 4];
    if (d.channels >= 3) {
      o[0] = s[0], o[1] = s[1], o[2] = s[2];
    } else {
      o[0] = o[1] = o[2] = s[0];
    }
    o[3] = d.channels == 4 ? s[3] : d.channels == 2 ? s[1] : 255;
  }
  return r;
}

namespace {

double lanczos(double x) {
  auto sinc = [](double v) {
    if (v == 0.0) return 1.0;
    v *= 3.14159265358979323846;
    return std::sin(v) / v;
  };
  return (x > -3.0 && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0;
}

// 一維重新取樣係數（PIL precompute_coeffs）：每個輸出點的起點與權重。
struct Coeffs {
  std::vector<int> start, count;
  std::vector<std::vector<double>> w;
};

Coeffs coeffs(int in, int out) {
  const double scale = double(in) / out, fs = std::max(scale, 1.0), support = 3.0 * fs;
  Coeffs c;
  for (int x = 0; x < out; x++) {
    double center = (x + 0.5) * scale;
    int lo = std::max(int(center - support + 0.5), 0);
    int hi = std::min(int(center + support + 0.5), in);
    std::vector<double> w;
    double sum = 0;
    for (int i = lo; i < hi; i++) {
      double v = lanczos((i - center + 0.5) / fs);
      w.push_back(v);
      sum += v;
    }
    for (auto& v : w) v = sum != 0 ? v / sum : 0;
    c.start.push_back(lo);
    c.count.push_back(hi - lo);
    c.w.push_back(std::move(w));
  }
  return c;
}

}  // namespace

Rgba resize_lanczos(const Rgba& src, int w, int h) {
  MX_CHECK(w > 0 && h > 0 && src.w > 0 && src.h > 0, "縮放尺寸無效");
  if (w == src.w && h == src.h) return src;
  // alpha 預乘（PIL 對 RGBA 先轉 RGBa 再縮放）。
  std::vector<float> pm(size_t(src.w) * src.h * 4);
  for (size_t i = 0; i < size_t(src.w) * src.h; i++) {
    float a = src.px[i * 4 + 3] / 255.f;
    for (int ch = 0; ch < 3; ch++) pm[i * 4 + ch] = src.px[i * 4 + ch] * a;
    pm[i * 4 + 3] = src.px[i * 4 + 3];
  }
  // 先水平再垂直（PIL 順序）。
  Coeffs cx = coeffs(src.w, w), cy = coeffs(src.h, h);
  std::vector<float> tmp(size_t(w) * src.h * 4);
  for (int y = 0; y < src.h; y++)
    for (int x = 0; x < w; x++)
      for (int ch = 0; ch < 4; ch++) {
        double acc = 0;
        for (int k = 0; k < cx.count[x]; k++) acc += cx.w[x][k] * pm[(size_t(y) * src.w + cx.start[x] + k) * 4 + ch];
        tmp[(size_t(y) * w + x) * 4 + ch] = float(acc);
      }
  Rgba out;
  out.w = w, out.h = h;
  out.px.resize(size_t(w) * h * 4);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float v[4];
      for (int ch = 0; ch < 4; ch++) {
        double acc = 0;
        for (int k = 0; k < cy.count[y]; k++) acc += cy.w[y][k] * tmp[(size_t(cy.start[y] + k) * w + x) * 4 + ch];
        v[ch] = float(acc);
      }
      float a = std::clamp(v[3], 0.f, 255.f);
      uint8_t* o = &out.px[(size_t(y) * w + x) * 4];
      o[3] = uint8_t(std::lround(a));
      for (int ch = 0; ch < 3; ch++) {
        float c = a > 0 ? v[ch] * 255.f / a : 0.f;
        o[ch] = uint8_t(std::lround(std::clamp(c, 0.f, 255.f)));
      }
    }
  return out;
}

Rgba cover_crop(const Rgba& src, int w, int h) {
  const double scale = std::max(double(w) / src.w, double(h) / src.h);
  const int rw = std::max(w, int(std::lround(src.w * scale))), rh = std::max(h, int(std::lround(src.h * scale)));
  Rgba r = resize_lanczos(src, rw, rh);
  const int left = std::max(0, (rw - w) / 2), top = std::max(0, (rh - h) / 2);
  Rgba out;
  out.w = w, out.h = h;
  out.px.resize(size_t(w) * h * 4);
  for (int y = 0; y < h; y++)
    std::copy_n(&r.px[(size_t(top + y) * rw + left) * 4], size_t(w) * 4, &out.px[size_t(y) * w * 4]);
  return out;
}

std::vector<uint8_t> flatten_white(const Rgba& src) {
  std::vector<uint8_t> o(size_t(src.w) * src.h * 3);
  for (size_t i = 0; i < size_t(src.w) * src.h; i++) {
    int a = src.px[i * 4 + 3];
    for (int ch = 0; ch < 3; ch++) o[i * 3 + ch] = uint8_t((src.px[i * 4 + ch] * a + 255 * (255 - a) + 127) / 255);
  }
  return o;
}

std::vector<uint8_t> drop_alpha(const Rgba& src) {
  std::vector<uint8_t> o(size_t(src.w) * src.h * 3);
  for (size_t i = 0; i < size_t(src.w) * src.h; i++)
    for (int ch = 0; ch < 3; ch++) o[i * 3 + ch] = src.px[i * 4 + ch];
  return o;
}

}  // namespace mx
