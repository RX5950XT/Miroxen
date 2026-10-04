#include "text/tokenizer.h"
#include "text/unicode_data.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_fails = 0;
int g_shown = 0;

void fail(const std::string& msg) {
  std::cerr << "FAIL " << msg << "\n";
  ++g_fails;
}

int hex_val(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string unhex(const std::string& h) {
  if (h.size() % 2) throw std::runtime_error("hex 長度是奇數");
  std::string o(h.size() / 2, '\0');
  for (size_t i = 0; i < o.size(); ++i) {
    int hi = hex_val(h[i * 2]), lo = hex_val(h[i * 2 + 1]);
    if (hi < 0 || lo < 0) throw std::runtime_error("hex 字元無效");
    o[i] = char((hi << 4) | lo);
  }
  return o;
}

std::vector<int32_t> parse_ids(const std::string& s) {
  std::vector<int32_t> o;
  std::istringstream in(s);
  std::string w;
  while (in >> w) {
    size_t idx = 0;
    long long v = std::stoll(w, &idx, 10);
    if (idx != w.size() || v < 0 || v > 4000000) throw std::runtime_error("token id 無效：" + w);
    o.push_back(static_cast<int32_t>(v));
  }
  return o;
}

struct Case {
  std::string text;
  std::string dec;
  std::vector<int32_t> ids;
};

std::string rest_of(const std::string& line) {
  if (line.size() == 1) return {};
  if (line.size() < 2 || line[1] != ' ') throw std::runtime_error("向量行格式錯誤：" + line.substr(0, 40));
  return line.substr(2);
}

std::map<std::string, std::vector<Case>> load_vectors(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("無法讀取 " + path);
  std::map<std::string, std::vector<Case>> out;
  std::string section, line;
  Case cur;
  int phase = 0;
  bool magic = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    if (!magic) {
      if (line != "MXTOK1") throw std::runtime_error("向量檔開頭不是 MXTOK1");
      magic = true;
      continue;
    }
    if (line[0] == '[') {
      if (phase != 0) throw std::runtime_error("向量案例不完整");
      if (line.back() != ']') throw std::runtime_error("區段標頭錯誤");
      section = line.substr(1, line.size() - 2);
      if (section != "qwen" && section != "minimax") throw std::runtime_error("未知區段 " + section);
      continue;
    }
    if (section.empty()) throw std::runtime_error("向量在區段之外：" + line.substr(0, 20));
    char kind = line[0];
    if (kind == 'T') {
      if (phase != 0) throw std::runtime_error("預期 T");
      cur = Case{};
      cur.text = unhex(rest_of(line));
      phase = 1;
    } else if (kind == 'D') {
      if (phase != 1) throw std::runtime_error("預期 D");
      cur.dec = unhex(rest_of(line));
      phase = 2;
    } else if (kind == 'I') {
      if (phase != 2) throw std::runtime_error("預期 I");
      cur.ids = parse_ids(rest_of(line));
      out[section].push_back(std::move(cur));
      phase = 0;
    } else {
      throw std::runtime_error("未知的向量行");
    }
  }
  if (phase != 0) throw std::runtime_error("向量檔結尾不完整");
  if (!magic) throw std::runtime_error("向量檔是空的");
  return out;
}

void expect_eq_ids(const std::vector<int32_t>& got, const std::vector<int32_t>& exp, const std::string& what) {
  if (got == exp) return;
  ++g_fails;
  if (g_shown >= 8) return;
  ++g_shown;
  std::cerr << "FAIL " << what << "\n expected";
  for (size_t i = 0; i < exp.size() && i < 40; ++i) std::cerr << " " << exp[i];
  if (exp.size() > 40) std::cerr << " ...";
  std::cerr << "\n got     ";
  for (size_t i = 0; i < got.size() && i < 40; ++i) std::cerr << " " << got[i];
  if (got.size() > 40) std::cerr << " ...";
  std::cerr << "\n";
}

void check_nfc() {
  auto eq = [](const char* name, const std::string& got, const std::string& exp) {
    if (got != exp) fail(std::string("NFC ") + name);
  };
  eq("ascii", mx::unicode_nfc("abc"), "abc");
  eq("acute", mx::unicode_nfc("e\u0301"), "\u00e9");
  eq("angstrom", mx::unicode_nfc("\u212b"), "\u00c5");
  eq("ddots", mx::unicode_nfc("\u0044\u0307\u0323"), "\u1e0c\u0307");
  eq("hangul", mx::unicode_nfc("\u1100\u1161"), "\uac00");
  eq("hangul-block", mx::unicode_nfc("\u1100\u0301\u1161"), "\u1100\u0301\u1161");
  eq("hangul-t", mx::unicode_nfc("\uac00\u11a8"), "\uac01");
  eq("cjk-compat", mx::unicode_nfc("\uf900"), "\u8c48");
  eq("double-acute", mx::unicode_nfc("\u00e9\u0301"), "\u00e9\u0301");
  eq("ligature", mx::unicode_nfc("\ufb00"), "\ufb00");
  eq("exclusion", mx::unicode_nfc("\u0958"), "\u0915\u093c");
  if (!mx::unicode_is_letter(U'你') || !mx::unicode_is_letter(U'A') || mx::unicode_is_letter(U'1'))
    fail("letter class");
  if (!mx::unicode_is_number(U'３') || !mx::unicode_is_number(U'5') || mx::unicode_is_number(U'a'))
    fail("number class");
  if (!mx::unicode_is_whitespace(U'\u3000') || !mx::unicode_is_whitespace(U'\n') ||
      mx::unicode_is_whitespace(U'a'))
    fail("whitespace class");
  if (mx::unicode_is_letter(0x017F) == false) fail("long s");
}

template <class F>
void expect_throw(const char* name, F&& fn) {
  try {
    fn();
    fail(std::string(name) + " 沒有丟出例外");
  } catch (const std::runtime_error&) {
  } catch (const std::exception& e) {
    fail(std::string(name) + " 例外型別不對");
    std::cerr << "  " << e.what() << "\n";
  }
}

void check_api(const mx::Tokenizer& tok) {
  if (tok.vocab_size() != 151669) fail("vocab_size " + std::to_string(tok.vocab_size()));
  if (tok.token_id("!") != 0) fail("token_id !");
  if (tok.token_id("<|endoftext|>") != 151643) fail("endoftext id");
  if (tok.token_id("<|im_start|>") != 151644) fail("im_start id");
  if (tok.token_id("<|im_end|>") != 151645) fail("im_end id");
  if (tok.token_id("<think>") != 151667) fail("think id");
  if (tok.token_id("</think>") != 151668) fail("think end id");
  if (tok.token_id("no-such-token-xyz") != -1) fail("missing token");
  if (tok.token_id(" hello") != -1) fail("raw space key");
  if (tok.encode("") != std::vector<int32_t>{}) fail("encode empty");
  if (tok.decode({}) != "") fail("decode empty");
  if (tok.encode("<|im_start|>") != std::vector<int32_t>{151644}) fail("encode special");
  if (tok.decode({151644}) != "<|im_start|>") fail("decode special");
  if (tok.decode({151644, 151645}) != "<|im_start|><|im_end|>") fail("decode two specials");
  auto hi = tok.encode("<|im_start|>hi<|im_end|>");
  if (hi.size() < 3 || hi.front() != 151644 || hi.back() != 151645) fail("specials around hi");
  if (tok.decode(hi) != "<|im_start|>hi<|im_end|>") fail("special roundtrip");
  int32_t sp = tok.token_id("\u0120");
  if (sp < 0 || tok.decode({sp}) != " ") fail("space byte");
  int32_t y = tok.token_id("\u00ff");
  if (y < 0 || tok.decode({y}) != "\uFFFD") fail("0xFF byte lossy");
  if (tok.decode(tok.encode("\u00ff")) != "\u00ff") fail("y diaeresis roundtrip");
}

void check_section(const mx::Tokenizer& tok, const std::vector<Case>& rows, const char* name) {
  int n = 0;
  for (const auto& c : rows) {
    auto ids = tok.encode(c.text);
    expect_eq_ids(ids, c.ids, std::string(name) + " encode #" + std::to_string(n));
    std::string dec = tok.decode(ids);
    if (dec != c.dec) {
      ++g_fails;
      if (g_shown < 8) {
        ++g_shown;
        std::cerr << "FAIL " << name << " decode #" << n << "\n";
      }
    }
    if (tok.decode(c.ids) != c.dec) {
      ++g_fails;
      if (g_shown < 8) {
        ++g_shown;
        std::cerr << "FAIL " << name << " decode ids #" << n << "\n";
      }
    }
    ++n;
  }
  std::cout << name << " vectors: " << n << " exact-checked\n";
}

void check_speed(const mx::Tokenizer& tok) {
  std::string para =
      "The quick brown fox jumps over 123 lazy dogs.\n"
      "你好世界，這是繁體中文與简体中文。\n"
      "日本語と한국어 🙂 rock'n'roll it's 3.14\n"
      "<|im_start|>user\n畫一隻貓<|im_end|>\n";
  std::string big;
  big.reserve(120000);
  while (big.size() < 100 * 1024) big += para;
  auto t0 = std::chrono::steady_clock::now();
  auto ids = tok.encode(big);
  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  if (sec <= 0) sec = 1e-9;
  double tps = ids.size() / sec;
  std::cout << "encode 100KB: " << static_cast<long long>(tps) << " tokens/s (" << ids.size()
            << " tokens, " << sec << " s)\n";
  if (ids.empty()) fail("speed produced no tokens");
  if (sec > 10.0) fail("encode 100KB slower than 10s");
  if (tok.decode(ids).empty()) fail("speed decode empty");
}

bool file_ok(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return static_cast<bool>(in);
}

const char* kQwen =
    "D:\\Workspace\\Personal_Project\\Miroxen\\models\\qwen-image-2.1\\hf\\processor\\tokenizer.json";
const char* kMini = "D:\\Workspace\\Personal_Project\\Miroxen\\models\\minimax-h3\\hf\\tokenizer\\tokenizer.json";
const char* kVec = "tests/data/tokenizer_vectors.txt";

}  // namespace

int main() {
  try {
    check_nfc();
    expect_throw("無效 NFC", [] { mx::unicode_nfc("\xff"); });
    if (!file_ok(kQwen)) throw std::runtime_error(std::string("找不到 Qwen tokenizer：") + kQwen);
    auto vec = load_vectors(kVec);
    if (vec["qwen"].size() < 200) fail("qwen 向量少於 200");
    auto qwen = mx::Tokenizer::load(kQwen);
    check_api(*qwen);
    check_section(*qwen, vec["qwen"], "qwen");
    expect_throw("缺少檔案", [] { mx::Tokenizer::load("tests/data/no-such-tokenizer.json"); });
    expect_throw("不是 json", [] { mx::Tokenizer::load("tests/tokenizer_test.cpp"); });
    expect_throw("無效 UTF-8", [&] { qwen->encode(std::string("\xff", 1)); });
    expect_throw("過長 UTF-8", [&] { qwen->encode(std::string("\xc0\x80", 2)); });
    expect_throw("未知 id", [&] { qwen->decode({-1}); });
    expect_throw("過大 id", [&] { qwen->decode({9999999}); });
    check_speed(*qwen);
    if (!file_ok(kMini)) {
      std::cout << "skip minimax: tokenizer.json 不存在\n";
    } else if (!vec.count("minimax") || vec["minimax"].empty()) {
      fail("有 MiniMax tokenizer 但向量沒有 [minimax]");
    } else {
      auto mini = mx::Tokenizer::load(kMini);
      check_api(*mini);
      check_section(*mini, vec["minimax"], "minimax");
    }
  } catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
  }
  if (g_fails) {
    std::cerr << g_fails << " failures\n";
    return 1;
  }
  std::cout << "all tokenizer checks passed\n";
  return 0;
}
