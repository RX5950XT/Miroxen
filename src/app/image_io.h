#pragma once
// PNG 編碼／圖片解碼（stb）。
#include <cstdint>
#include <string>
#include <vector>

namespace mx {

std::vector<uint8_t> encode_png_rgba(const uint8_t* rgba, int w, int h);
// 讀 PNG/JPG 檔成 RGBA8；失敗丟例外。
std::vector<uint8_t> load_image_rgba(const std::string& path, int& w, int& h);
void write_file(const std::string& path, const std::vector<uint8_t>& data);

}  // namespace mx
