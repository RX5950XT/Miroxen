#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mx {

struct DecodedImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;
};

std::vector<uint8_t> encode_png(const uint8_t* pixels, int w, int h, int channels);
std::vector<uint8_t> encode_rgb_png(const uint8_t* rgb, int w, int h);
// 解不出來或尺寸不合理時丟 std::runtime_error。
DecodedImage decode_image_bytes(const uint8_t* data, size_t n);
void write_file(const std::filesystem::path& path, const uint8_t* data, size_t n);

std::string b64_encode(const uint8_t* data, size_t n);
std::vector<uint8_t> b64_decode(const std::string& in);

}  // namespace mx
