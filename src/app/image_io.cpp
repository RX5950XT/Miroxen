#include "app/image_io.h"

#include <cstdio>
#include <stdexcept>

#include "core/common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace mx {

std::vector<uint8_t> encode_png_rgba(const uint8_t* rgba, int w, int h) {
  std::vector<uint8_t> out;
  auto sink = [](void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
    v->insert(v->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
  };
  MX_CHECK(stbi_write_png_to_func(sink, &out, w, h, 4, rgba, w * 4), "PNG 編碼失敗");
  return out;
}

std::vector<uint8_t> load_image_rgba(const std::string& path, int& w, int& h) {
  int n = 0;
  uint8_t* p = stbi_load(path.c_str(), &w, &h, &n, 4);
  MX_CHECK(p, "無法讀取圖片 %s：%s", path.c_str(), stbi_failure_reason());
  std::vector<uint8_t> v(p, p + size_t(w) * h * 4);
  stbi_image_free(p);
  return v;
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
  FILE* f = std::fopen(path.c_str(), "wb");
  MX_CHECK(f, "無法寫入 %s", path.c_str());
  size_t n = std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  MX_CHECK(n == data.size(), "寫入 %s 不完整", path.c_str());
}

}  // namespace mx
