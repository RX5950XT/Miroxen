#pragma once
// 條件圖前處理：讀檔、PIL 風格 Lanczos 縮放、裁切、合成白底。
#include <cstdint>
#include <string>
#include <vector>

namespace mx {

struct Rgba {
  int w = 0, h = 0;
  std::vector<uint8_t> px;  // [h, w, 4]
};

// 讀 PNG/JPEG 檔成 RGBA（沒有 alpha 時補 255）。
Rgba load_rgba(const std::string& path);
// PIL Image.resize(LANCZOS) 的浮點版：a=3、縮小時取樣範圍隨比例放大（抗鋸齒），RGBA 先以 alpha 預乘再縮放。
Rgba resize_lanczos(const Rgba& src, int w, int h);
// 依比例放大到蓋滿 w×h（Lanczos），再置中裁切（MiniMax-H3 第二張關鍵幀）。
Rgba cover_crop(const Rgba& src, int w, int h);
// RGBA 疊在白底上 → RGB [h, w, 3]（Qwen-Image 視覺編碼器的輸入方式）。
std::vector<uint8_t> flatten_white(const Rgba& src);
// 直接丟掉 alpha → RGB。
std::vector<uint8_t> drop_alpha(const Rgba& src);

}  // namespace mx
