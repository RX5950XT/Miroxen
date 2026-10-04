#include "server/image_io.h"

#include <fstream>
#include <stdexcept>

#include "core/common.h"
#include "server/server_internal.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace mx {
namespace {

void png_sink(void* ctx, void* data, int size) {
  auto* out = static_cast<std::vector<uint8_t>*>(ctx);
  auto* b = static_cast<const uint8_t*>(data);
  out->insert(out->end(), b, b + size);
}

int b64_value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

}  // namespace

std::vector<uint8_t> encode_png(const uint8_t* pixels, int w, int h, int channels) {
  std::vector<uint8_t> out;
  int stride = w * channels;
  int ok = stbi_write_png_to_func(png_sink, &out, w, h, channels, pixels, stride);
  MX_CHECK(ok && !out.empty(), "PNG 編碼失敗");
  return out;
}

std::vector<uint8_t> encode_rgb_png(const uint8_t* rgb, int w, int h) {
  return encode_png(rgb, w, h, 3);
}

DecodedImage decode_image_bytes(const uint8_t* data, size_t n) {
  MX_CHECK(data && n > 0, "參考圖是空的");
  MX_CHECK(n <= static_cast<size_t>(INT32_MAX), "參考圖太大");
  int w = 0, h = 0, comp = 0;
  unsigned char* px = stbi_load_from_memory(data, static_cast<int>(n), &w, &h, &comp, 0);
  if (!px) throw std::runtime_error("參考圖不是有效的 PNG 或 JPEG");
  struct Free {
    unsigned char* p;
    ~Free() { stbi_image_free(p); }
  } guard{px};
  MX_CHECK(w > 0 && h > 0 && w <= 8192 && h <= 8192, "參考圖尺寸必須介於 1 與 8192");
  MX_CHECK(comp >= 1 && comp <= 4, "參考圖通道數無法處理");
  DecodedImage img;
  img.width = w;
  img.height = h;
  img.channels = comp;
  img.pixels.assign(px, px + static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(comp));
  return img;
}

void write_file(const std::filesystem::path& path, const uint8_t* data, size_t n) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  MX_CHECK(static_cast<bool>(out), "無法寫入 %s", path_utf8(path).c_str());
  if (n) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
  MX_CHECK(static_cast<bool>(out), "寫入 %s 失敗", path_utf8(path).c_str());
}

std::string b64_encode(const uint8_t* data, size_t n) {
  static const char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (size_t i = 0; i < n; i += 3) {
    unsigned v = static_cast<unsigned>(data[i]) << 16;
    if (i + 1 < n) v |= static_cast<unsigned>(data[i + 1]) << 8;
    if (i + 2 < n) v |= static_cast<unsigned>(data[i + 2]);
    out.push_back(kTable[(v >> 18) & 63]);
    out.push_back(kTable[(v >> 12) & 63]);
    out.push_back(i + 1 < n ? kTable[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < n ? kTable[v & 63] : '=');
  }
  return out;
}

std::vector<uint8_t> b64_decode(const std::string& in) {
  std::vector<uint8_t> out;
  out.reserve(in.size() * 3 / 4);
  int value = 0;
  int bits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    int d = b64_value(c);
    if (d < 0) throw std::runtime_error("base64 不合法");
    value = (value << 6) | d;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((value >> bits) & 0xFF));
    }
  }
  return out;
}

}  // namespace mx
