#pragma once
// Unicode 分類與 NFC。表格由 Python unicodedata 產生（見 unicode_data.cpp）。
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mx {

struct Utf8Cp {
  char32_t cp;
  uint32_t off;  // 此碼位在 UTF-8 位元組串的起點
};

// 嚴格 UTF-8。過長編碼、代理對、截斷都丟例外。
std::vector<Utf8Cp> utf8_decode(std::string_view s);
std::string utf32_to_utf8(const char32_t* p, size_t n);

bool unicode_is_letter(char32_t cp);      // \p{L}
bool unicode_is_number(char32_t cp);      // \p{N}
bool unicode_is_whitespace(char32_t cp);  // Unicode White_Space（對齊 Rust regex 的 \s）
uint8_t unicode_combining_class(char32_t cp);

// NFC（UAX #15）。純 ASCII 原樣返回。無效 UTF-8 丟例外。
std::string unicode_nfc(std::string_view utf8);

}  // namespace mx
