#include "text/tokenizer.h"

#include "core/common.h"
#include "text/unicode_data.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace mx {
namespace {

constexpr size_t kNpos = size_t(-1);
constexpr int32_t kMaxId = 4000000;

// tokenizer.json 裡 Qwen2 / Qwen3 的 Split regex（JSON 解開之後的字面）。
const char kQwenRe[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";

struct Sym {
  int32_t c = 0;
  int32_t prev = -1;
  int32_t next = -1;
  int32_t len = 0;
};

struct Cand {
  uint32_t rank;
  uint32_t pos;
  int32_t nid;
};

struct CandMin {
  bool operator()(const Cand& a, const Cand& b) const {
    if (a.rank != b.rank) return a.rank > b.rank;
    return a.pos > b.pos;
  }
};

struct MergeInfo {
  uint32_t rank;
  int32_t id;
};

void append_utf8_lossy(std::string& out, const std::string& bytes) {
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  size_t i = 0, n = bytes.size();
  auto put = [&](char32_t cp) {
    char32_t one = cp;
    out += utf32_to_utf8(&one, 1);
  };
  while (i < n) {
    unsigned char c = p[i];
    size_t need = 1;
    char32_t cp = c;
    if (c < 0x80) {
      out.push_back(char(c));
      ++i;
      continue;
    }
    if ((c & 0xE0) == 0xC0) {
      need = 2;
      cp = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
      need = 3;
      cp = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
      need = 4;
      cp = c & 0x07;
    } else {
      put(0xFFFD);
      ++i;
      continue;
    }
    if (i + need > n) {
      put(0xFFFD);
      break;
    }
    bool ok = true;
    for (size_t k = 1; k < need; ++k) {
      if ((p[i + k] & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (p[i + k] & 0x3F);
    }
    if (need == 2) ok = ok && cp >= 0x80;
    if (need == 3) ok = ok && cp >= 0x800;
    if (need == 4) ok = ok && cp >= 0x10000;
    ok = ok && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
    if (!ok) {
      put(0xFFFD);
      ++i;
      continue;
    }
    put(cp);
    i += need;
  }
}

bool ci_eq(char32_t c, char lo) {
  if (c == char32_t(lo) || c == char32_t(lo - 32)) return true;
  return lo == 's' && c == 0x017F;  // ſ，regex (?i) 的 simple casefold
}

bool letter_prefix(char32_t c) {
  return c != '\r' && c != '\n' && !unicode_is_letter(c) && !unicode_is_number(c);
}

size_t match_contraction(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  if (i >= n || u[i].cp != '\'') return kNpos;
  auto has = [&](size_t k, char lo) { return i + k < n && ci_eq(u[i + k].cp, lo); };
  if (has(1, 's') || has(1, 't')) return i + 2;
  if (has(1, 'r') && has(2, 'e')) return i + 3;
  if (has(1, 'v') && has(2, 'e')) return i + 3;
  if (has(1, 'm')) return i + 2;
  if (has(1, 'l') && has(2, 'l')) return i + 3;
  if (has(1, 'd')) return i + 2;
  return kNpos;
}

size_t match_letters(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  if (i >= n) return kNpos;
  size_t j = i;
  if (letter_prefix(u[i].cp) && i + 1 < n && unicode_is_letter(u[i + 1].cp)) j = i + 1;
  if (j >= n || !unicode_is_letter(u[j].cp)) return kNpos;
  ++j;
  while (j < n && unicode_is_letter(u[j].cp)) ++j;
  return j;
}

size_t match_number(const std::vector<Utf8Cp>& u, size_t i) {
  if (i < u.size() && unicode_is_number(u[i].cp)) return i + 1;
  return kNpos;
}

size_t match_punct(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  size_t j = i;
  if (j < n && u[j].cp == ' ') ++j;
  size_t start = j;
  while (j < n && !unicode_is_whitespace(u[j].cp) && !unicode_is_letter(u[j].cp) &&
         !unicode_is_number(u[j].cp))
    ++j;
  if (j == start) return kNpos;
  while (j < n && (u[j].cp == '\r' || u[j].cp == '\n')) ++j;
  return j;
}

size_t match_newline(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  size_t j = i;
  while (j < n && unicode_is_whitespace(u[j].cp)) ++j;
  if (j == i) return kNpos;
  while (j > i && u[j - 1].cp != '\r' && u[j - 1].cp != '\n') --j;
  return j == i ? kNpos : j;
}

// \s+(?!\S)：吃到檔尾，或在非空白前留下最後一個空白給後面的規則。
size_t match_ws_tail(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  size_t j = i;
  while (j < n && unicode_is_whitespace(u[j].cp)) ++j;
  size_t m = j - i;
  if (m == 0) return kNpos;
  if (j == n) return j;
  if (m >= 2) return j - 1;
  return kNpos;
}

size_t match_ws(const std::vector<Utf8Cp>& u, size_t i) {
  size_t n = u.size();
  size_t j = i;
  while (j < n && unicode_is_whitespace(u[j].cp)) ++j;
  return j == i ? kNpos : j;
}

size_t match_at(const std::vector<Utf8Cp>& u, size_t i) {
  size_t e;
  if ((e = match_contraction(u, i)) != kNpos) return e;
  if ((e = match_letters(u, i)) != kNpos) return e;
  if ((e = match_number(u, i)) != kNpos) return e;
  if ((e = match_punct(u, i)) != kNpos) return e;
  if ((e = match_newline(u, i)) != kNpos) return e;
  if ((e = match_ws_tail(u, i)) != kNpos) return e;
  return match_ws(u, i);
}

struct JVal {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  int64_t num = 0;
  std::string str;
  std::vector<JVal> arr;
  std::vector<std::pair<std::string, JVal>> obj;
  const JVal* find(const char* key) const {
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
};

class TokImpl final : public Tokenizer {
 public:
  TokImpl() { init_bytes(); }

  std::vector<int32_t> encode(const std::string& text) const override;
  std::string decode(const std::vector<int32_t>& ids) const override;
  int32_t token_id(const std::string& token) const override;
  size_t vocab_size() const override { return entries_.size(); }

  void load_file(const std::string& path);

 private:
  friend class Parser;

  struct Entry {
    std::string bytes;
    std::string text;
    bool literal = false;
    bool seen = false;
  };
  struct Added {
    std::string content;
    int32_t id = 0;
  };

  void init_bytes();
  void add_vocab(const std::string& tok, int32_t id);
  void add_added(int32_t id, const std::string& content);
  void add_merge(std::string a, std::string b);
  void take_norm(const JVal& v);
  void take_pretok(const JVal& v);
  void finalize();
  std::string token_bytes(const std::string& tok) const;
  void ensure_id(int32_t id);
  int find_added(const std::string& text, size_t i) const;
  void pretok(const std::string& s, std::vector<int32_t>& out) const;
  void bpe_append(const std::string& piece, std::vector<int32_t>& out) const;
  std::vector<int32_t> bpe_compute(const std::string& bytes) const;
  void merge_syms(std::vector<Sym>& syms) const;

  char32_t byte_cp_[256]{};
  int16_t cp_to_byte_[324]{};
  int32_t byte_id_[256]{};
  bool nfc_ = false;
  bool saw_regex_ = false;
  std::vector<Entry> entries_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::pair<std::string, std::string>> raw_merges_;
  std::unordered_map<uint64_t, MergeInfo> merges_;
  std::vector<Added> added_;
  std::vector<int> added_at_[256];
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

void TokImpl::init_bytes() {
  for (int i = 0; i < 324; ++i) cp_to_byte_[i] = -1;
  bool used[256] = {};
  auto add = [&](int a, int b) {
    for (int i = a; i <= b; ++i) {
      byte_cp_[i] = char32_t(i);
      used[i] = true;
    }
  };
  add(int('!'), int('~'));
  add(0xA1, 0xAC);
  add(0xAE, 0xFF);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (!used[b]) byte_cp_[b] = char32_t(256 + n++);
    cp_to_byte_[byte_cp_[b]] = int16_t(b);
  }
}

void TokImpl::ensure_id(int32_t id) {
  MX_CHECK(id >= 0 && id <= kMaxId, "token id 超出範圍：%d", id);
  if (size_t(id) >= entries_.size()) entries_.resize(size_t(id) + 1);
}

std::string TokImpl::token_bytes(const std::string& tok) const {
  auto cps = utf8_decode(tok);
  std::string o;
  o.reserve(cps.size());
  for (const auto& u : cps) {
    MX_CHECK(u.cp < 324 && cp_to_byte_[u.cp] >= 0, "詞表 token 含非 byte-level 字元 U+%04X",
             unsigned(u.cp));
    o.push_back(char(cp_to_byte_[u.cp]));
  }
  return o;
}

void TokImpl::add_vocab(const std::string& tok, int32_t id) {
  ensure_id(id);
  MX_CHECK(!entries_[id].seen, "重複的 token id %d", id);
  entries_[id].seen = true;
  entries_[id].bytes = token_bytes(tok);
  auto [it, ok] = token_to_id_.emplace(tok, id);
  MX_CHECK(ok, "詞表有重複的 token 字串");
  (void)it;
}

void TokImpl::add_added(int32_t id, const std::string& content) {
  MX_CHECK(!content.empty(), "added token 內容是空的");
  MX_CHECK(token_to_id_.find(content) == token_to_id_.end(), "added token 內容與既有 token 重複");
  ensure_id(id);
  MX_CHECK(!entries_[id].seen, "added token id %d 與既有 id 衝突", id);
  entries_[id].seen = true;
  entries_[id].literal = true;
  entries_[id].text = content;
  token_to_id_.emplace(content, id);
  added_.push_back(Added{content, id});
}

void TokImpl::add_merge(std::string a, std::string b) {
  MX_CHECK(!a.empty() && !b.empty(), "BPE merge 片段是空的");
  raw_merges_.emplace_back(std::move(a), std::move(b));
}

void TokImpl::take_norm(const JVal& v) {
  if (v.kind == JVal::Null) {
    nfc_ = false;
    return;
  }
  MX_CHECK(v.kind == JVal::Obj, "normalizer 格式錯誤");
  const JVal* type = v.find("type");
  MX_CHECK(type && type->kind == JVal::Str, "normalizer 缺少 type");
  if (type->str == "NFC") {
    nfc_ = true;
    return;
  }
  MX_CHECK(type->str == "Sequence", "不支援的 normalizer：%s", type->str.c_str());
  const JVal* arr = v.find("normalizers");
  MX_CHECK(arr && arr->kind == JVal::Arr && !arr->arr.empty(), "Sequence normalizer 格式錯誤");
  for (const auto& step : arr->arr) {
    MX_CHECK(step.kind == JVal::Obj, "normalizer 步驟格式錯誤");
    const JVal* st = step.find("type");
    MX_CHECK(st && st->kind == JVal::Str, "normalizer 步驟缺少 type");
    MX_CHECK(st->str == "NFC", "不支援的 normalizer：%s", st->str.c_str());
    nfc_ = true;
  }
}

void TokImpl::take_pretok(const JVal& v) {
  if (v.kind == JVal::Arr) {
    for (const auto& x : v.arr) take_pretok(x);
    return;
  }
  if (v.kind != JVal::Obj) return;
  const JVal* type = v.find("type");
  if (type && type->kind == JVal::Str && type->str == "Split") {
    const JVal* beh = v.find("behavior");
    MX_CHECK(beh && beh->kind == JVal::Str && beh->str == "Isolated", "不支援的 Split behavior");
    const JVal* inv = v.find("invert");
    MX_CHECK(!inv || (inv->kind == JVal::Bool && !inv->b), "不支援 invert Split");
    const JVal* pat = v.find("pattern");
    MX_CHECK(pat, "Split 缺少 pattern");
    const std::string* re = nullptr;
    if (pat->kind == JVal::Str) re = &pat->str;
    else if (pat->kind == JVal::Obj) {
      const JVal* r = pat->find("Regex");
      MX_CHECK(r && r->kind == JVal::Str, "Split pattern 不是 Regex");
      re = &r->str;
    } else {
      MX_CHECK(false, "Split pattern 格式錯誤");
    }
    MX_CHECK(*re == kQwenRe, "不支援的 pre-tokenizer regex");
    saw_regex_ = true;
  }
  if (type && type->kind == JVal::Str && type->str == "ByteLevel") {
    const JVal* ap = v.find("add_prefix_space");
    MX_CHECK(!ap || ap->kind != JVal::Bool || !ap->b, "尚未支援 ByteLevel add_prefix_space");
    const JVal* ur = v.find("use_regex");
    MX_CHECK(!ur || ur->kind != JVal::Bool || !ur->b, "尚未支援 ByteLevel use_regex");
  }
  for (const auto& kv : v.obj) take_pretok(kv.second);
}

class Parser {
 public:
  Parser(const std::string& s, TokImpl& t) : s_(s), t_(t) {}
  void run();

 private:
  const std::string& s_;
  size_t i_ = 0;
  TokImpl& t_;

  [[noreturn]] void die(const std::string& m) const;
  void skip_ws();
  char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
  void expect(char c);
  bool consume(char c);
  bool consume_word(const char* w);
  std::string parse_string();
  uint32_t parse_hex4();
  int64_t parse_int();
  JVal parse_jval();
  template <class Fn>
  void parse_object(Fn&& on_key);
  void parse_merges();
  void parse_model();
};

void Parser::die(const std::string& m) const {
  throw std::runtime_error("tokenizer.json 位元組 " + std::to_string(i_) + "：" + m);
}

void Parser::skip_ws() {
  while (i_ < s_.size()) {
    char c = s_[i_];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
    ++i_;
  }
}

void Parser::expect(char c) {
  if (!consume(c)) die(std::string("預期 '") + c + "'");
}

bool Parser::consume(char c) {
  if (i_ < s_.size() && s_[i_] == c) {
    ++i_;
    return true;
  }
  return false;
}

bool Parser::consume_word(const char* w) {
  size_t n = std::strlen(w);
  if (s_.compare(i_, n, w) != 0) return false;
  if (i_ + n < s_.size()) {
    unsigned char c = static_cast<unsigned char>(s_[i_ + n]);
    if (std::isalnum(c) || c == '_') return false;
  }
  i_ += n;
  return true;
}

uint32_t Parser::parse_hex4() {
  uint32_t v = 0;
  for (int k = 0; k < 4; ++k) {
    if (i_ >= s_.size()) die("\\u 被截斷");
    unsigned char c = static_cast<unsigned char>(s_[i_++]);
    v <<= 4;
    if (c >= '0' && c <= '9') v += c - '0';
    else if (c >= 'a' && c <= 'f') v += c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') v += c - 'A' + 10;
    else die("\\u 不是十六進位");
  }
  return v;
}

std::string Parser::parse_string() {
  if (!consume('"')) die("預期字串");
  std::string o;
  while (i_ < s_.size()) {
    unsigned char c = static_cast<unsigned char>(s_[i_++]);
    if (c == '"') return o;
    if (c == '\\') {
      if (i_ >= s_.size()) die("字串跳脫被截斷");
      char e = s_[i_++];
      switch (e) {
        case '"':
        case '\\':
        case '/': o.push_back(e); break;
        case 'b': o.push_back('\b'); break;
        case 'f': o.push_back('\f'); break;
        case 'n': o.push_back('\n'); break;
        case 'r': o.push_back('\r'); break;
        case 't': o.push_back('\t'); break;
        case 'u': {
          uint32_t u = parse_hex4();
          if (u >= 0xD800 && u <= 0xDBFF) {
            if (!(i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u')) die("孤立的高代理");
            i_ += 2;
            uint32_t lo = parse_hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) die("代理對的低位不正確");
            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
          } else if (u >= 0xDC00 && u <= 0xDFFF) {
            die("孤立的低代理");
          }
          char32_t cp = u;
          o += utf32_to_utf8(&cp, 1);
          break;
        }
        default: die("未知的字串跳脫");
      }
    } else if (c < 0x20) {
      die("字串含未跳脫的控制字元");
    } else {
      o.push_back(char(c));
    }
  }
  die("字串沒有結束引號");
}

int64_t Parser::parse_int() {
  skip_ws();
  size_t start = i_;
  int sign = 1;
  if (consume('-')) sign = -1;
  if (i_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[i_]))) die("預期整數");
  int64_t v = 0;
  while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) {
    int d = s_[i_] - '0';
    MX_CHECK(v <= (INT64_MAX - d) / 10, "整數溢位（位元組 %zu）", start);
    v = v * 10 + d;
    ++i_;
  }
  if (i_ < s_.size() && (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E')) die("預期整數");
  return sign < 0 ? -v : v;
}

template <class Fn>
void Parser::parse_object(Fn&& on_key) {
  expect('{');
  skip_ws();
  if (consume('}')) return;
  while (true) {
    skip_ws();
    std::string key = parse_string();
    skip_ws();
    expect(':');
    skip_ws();
    on_key(key);
    skip_ws();
    if (consume('}')) return;
    expect(',');
  }
}

JVal Parser::parse_jval() {
  skip_ws();
  JVal v;
  char c = peek();
  if (c == '"') {
    v.kind = JVal::Str;
    v.str = parse_string();
  } else if (c == '{') {
    v.kind = JVal::Obj;
    parse_object([&](const std::string& key) { v.obj.emplace_back(key, parse_jval()); });
  } else if (c == '[') {
    v.kind = JVal::Arr;
    expect('[');
    skip_ws();
    if (!consume(']')) {
      while (true) {
        v.arr.push_back(parse_jval());
        skip_ws();
        if (consume(']')) break;
        expect(',');
      }
    }
  } else if (consume_word("true")) {
    v.kind = JVal::Bool;
    v.b = true;
  } else if (consume_word("false")) {
    v.kind = JVal::Bool;
  } else if (consume_word("null")) {
    v.kind = JVal::Null;
  } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
    v.kind = JVal::Num;
    v.num = parse_int();
  } else {
    die("預期 JSON 值");
  }
  return v;
}

void Parser::parse_merges() {
  expect('[');
  skip_ws();
  if (consume(']')) return;
  while (true) {
    skip_ws();
    if (peek() == '"') {
      std::string m = parse_string();
      size_t sp = m.find(' ');
      if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) die("merge 字串缺少空白分隔");
      t_.add_merge(m.substr(0, sp), m.substr(sp + 1));
    } else if (peek() == '[') {
      expect('[');
      skip_ws();
      std::string a = parse_string();
      skip_ws();
      expect(',');
      skip_ws();
      std::string b = parse_string();
      skip_ws();
      expect(']');
      t_.add_merge(std::move(a), std::move(b));
    } else {
      die("merge 必須是字串或兩個字串的陣列");
    }
    skip_ws();
    if (consume(']')) return;
    expect(',');
  }
}

void Parser::parse_model() {
  bool saw_vocab = false, saw_merges = false;
  parse_object([&](const std::string& key) {
    if (key == "vocab") {
      MX_CHECK(!saw_vocab, "model.vocab 重複");
      saw_vocab = true;
      parse_object([&](const std::string& tok) {
        int64_t id = parse_int();
        MX_CHECK(id >= 0 && id <= kMaxId, "vocab id 超出範圍");
        t_.add_vocab(tok, int32_t(id));
      });
    } else if (key == "merges") {
      MX_CHECK(!saw_merges, "model.merges 重複");
      saw_merges = true;
      parse_merges();
    } else if (key == "type") {
      JVal v = parse_jval();
      MX_CHECK(v.kind == JVal::Str && v.str == "BPE", "只支援 BPE model");
    } else if (key == "ignore_merges" || key == "byte_fallback" || key == "fuse_unk") {
      JVal v = parse_jval();
      MX_CHECK(v.kind == JVal::Bool && !v.b, "尚未支援 model.%s", key.c_str());
    } else if (key == "continuing_subword_prefix" || key == "end_of_word_suffix") {
      JVal v = parse_jval();
      MX_CHECK(v.kind == JVal::Str && v.str.empty(), "尚未支援 model.%s", key.c_str());
    } else {
      parse_jval();
    }
  });
  MX_CHECK(saw_vocab, "tokenizer.json 缺少 model.vocab");
  MX_CHECK(saw_merges, "tokenizer.json 缺少 model.merges");
}

void Parser::run() {
  if (s_.size() >= 3 && static_cast<unsigned char>(s_[0]) == 0xEF &&
      static_cast<unsigned char>(s_[1]) == 0xBB && static_cast<unsigned char>(s_[2]) == 0xBF)
    i_ = 3;
  bool saw_model = false, saw_norm = false, saw_pre = false;
  parse_object([&](const std::string& key) {
    if (key == "added_tokens") {
      JVal arr = parse_jval();
      MX_CHECK(arr.kind == JVal::Arr, "added_tokens 不是陣列");
      for (const auto& item : arr.arr) {
        MX_CHECK(item.kind == JVal::Obj, "added token 不是物件");
        const JVal* id = item.find("id");
        const JVal* content = item.find("content");
        MX_CHECK(id && id->kind == JVal::Num, "added token 缺少 id");
        MX_CHECK(content && content->kind == JVal::Str, "added token 缺少 content");
        MX_CHECK(id->num >= 0 && id->num <= kMaxId, "added token id 超出範圍");
        const char* flags[] = {"normalized", "lstrip", "rstrip", "single_word"};
        for (const char* flag : flags) {
          const JVal* f = item.find(flag);
          MX_CHECK(!f || (f->kind == JVal::Bool && !f->b), "尚未支援 added token 的 %s", flag);
        }
        t_.add_added(int32_t(id->num), content->str);
      }
    } else if (key == "normalizer") {
      MX_CHECK(!saw_norm, "normalizer 重複");
      saw_norm = true;
      t_.take_norm(parse_jval());
    } else if (key == "pre_tokenizer") {
      MX_CHECK(!saw_pre, "pre_tokenizer 重複");
      saw_pre = true;
      t_.take_pretok(parse_jval());
    } else if (key == "model") {
      MX_CHECK(!saw_model, "model 重複");
      saw_model = true;
      parse_model();
    } else {
      parse_jval();
    }
  });
  skip_ws();
  if (i_ != s_.size()) die("JSON 結尾還有內容");
  MX_CHECK(saw_model, "tokenizer.json 缺少 model");
}

void TokImpl::finalize() {
  MX_CHECK(saw_regex_, "tokenizer.json 的 pre_tokenizer 沒有 Qwen2 regex");
  MX_CHECK(!token_to_id_.empty(), "詞表是空的");
  MX_CHECK(!raw_merges_.empty(), "merges 是空的");
  for (int b = 0; b < 256; ++b) {
    char32_t cp = byte_cp_[b];
    std::string key = utf32_to_utf8(&cp, 1);
    auto it = token_to_id_.find(key);
    MX_CHECK(it != token_to_id_.end(), "詞表缺少位元組 %d 的 token", b);
    byte_id_[b] = it->second;
  }
  merges_.reserve(raw_merges_.size());
  for (size_t rank = 0; rank < raw_merges_.size(); ++rank) {
    const std::string& a = raw_merges_[rank].first;
    const std::string& b = raw_merges_[rank].second;
    auto ia = token_to_id_.find(a);
    auto ib = token_to_id_.find(b);
    MX_CHECK(ia != token_to_id_.end() && ib != token_to_id_.end(), "merges[%zu] 的片段不在詞表", rank);
    auto im = token_to_id_.find(a + b);
    MX_CHECK(im != token_to_id_.end(), "merges[%zu] 的合併結果不在詞表", rank);
    uint64_t key = (uint64_t(uint32_t(ia->second)) << 32) | uint32_t(ib->second);
    auto [it, ok] = merges_.emplace(key, MergeInfo{uint32_t(rank), im->second});
    MX_CHECK(ok, "重複的 BPE merge，順位 %zu", rank);
    (void)it;
  }
  raw_merges_.clear();
  raw_merges_.shrink_to_fit();

  std::vector<int> order(added_.size());
  for (int i = 0; i < int(added_.size()); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (added_[a].content.size() != added_[b].content.size())
      return added_[a].content.size() > added_[b].content.size();
    return added_[a].id < added_[b].id;
  });
  for (int i : order) {
    unsigned char f = static_cast<unsigned char>(added_[i].content[0]);
    added_at_[f].push_back(i);
  }
}

void TokImpl::load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  MX_CHECK(static_cast<bool>(in), "無法開啟 tokenizer：%s", path.c_str());
  in.seekg(0, std::ios::end);
  std::streamoff n = in.tellg();
  MX_CHECK(n > 0, "tokenizer 是空的：%s", path.c_str());
  in.seekg(0);
  std::string data(size_t(n), '\0');
  in.read(data.data(), n);
  MX_CHECK(static_cast<bool>(in), "讀取 tokenizer 失敗：%s", path.c_str());
  token_to_id_.reserve(1 << 18);
  raw_merges_.reserve(1 << 18);
  Parser(data, *this).run();
  finalize();
}

int TokImpl::find_added(const std::string& text, size_t i) const {
  if (i >= text.size()) return -1;
  const auto& bucket = added_at_[static_cast<unsigned char>(text[i])];
  for (int idx : bucket) {
    const std::string& c = added_[idx].content;
    if (c.size() > text.size() - i) continue;
    if (std::memcmp(text.data() + i, c.data(), c.size()) == 0) return idx;
  }
  return -1;
}

void TokImpl::merge_syms(std::vector<Sym>& syms) const {
  std::priority_queue<Cand, std::vector<Cand>, CandMin> heap;
  auto consider = [&](int32_t pos) {
    if (pos < 0 || syms[size_t(pos)].len == 0) return;
    int32_t nxt = syms[size_t(pos)].next;
    if (nxt < 0) return;
    uint64_t key = (uint64_t(uint32_t(syms[size_t(pos)].c)) << 32) | uint32_t(syms[size_t(nxt)].c);
    auto it = merges_.find(key);
    if (it == merges_.end()) return;
    heap.push(Cand{it->second.rank, uint32_t(pos), it->second.id});
  };
  for (uint32_t i = 0; i + 1 < syms.size(); ++i) consider(int32_t(i));
  while (!heap.empty()) {
    Cand top = heap.top();
    heap.pop();
    Sym& left = syms[top.pos];
    if (left.len == 0 || left.next < 0) continue;
    int32_t nxt = left.next;
    Sym& right = syms[size_t(nxt)];
    uint64_t key = (uint64_t(uint32_t(left.c)) << 32) | uint32_t(right.c);
    auto it = merges_.find(key);
    if (it == merges_.end() || it->second.id != top.nid) continue;
    left.c = top.nid;
    left.len += right.len;
    left.next = right.next;
    right.len = 0;
    if (right.next >= 0) syms[size_t(right.next)].prev = int32_t(top.pos);
    if (left.prev >= 0) consider(left.prev);
    consider(int32_t(top.pos));
  }
}

std::vector<int32_t> TokImpl::bpe_compute(const std::string& bytes) const {
  MX_CHECK(bytes.size() <= 0x7FFFFFFF, "pre-token 過長");
  std::vector<Sym> syms(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    syms[i].c = byte_id_[static_cast<unsigned char>(bytes[i])];
    syms[i].prev = i == 0 ? -1 : int32_t(i - 1);
    syms[i].next = i + 1 == bytes.size() ? -1 : int32_t(i + 1);
    syms[i].len = 1;
  }
  if (syms.size() >= 2) merge_syms(syms);
  std::vector<int32_t> ids;
  ids.reserve(syms.size());
  for (const auto& s : syms)
    if (s.len) ids.push_back(s.c);
  return ids;
}

void TokImpl::bpe_append(const std::string& piece, std::vector<int32_t>& out) const {
  auto it = cache_.find(piece);
  if (it != cache_.end()) {
    out.insert(out.end(), it->second.begin(), it->second.end());
    return;
  }
  auto ids = bpe_compute(piece);
  out.insert(out.end(), ids.begin(), ids.end());
  cache_.emplace(piece, std::move(ids));
}

void TokImpl::pretok(const std::string& s, std::vector<int32_t>& out) const {
  auto units = utf8_decode(s);
  size_t n = units.size();
  size_t i = 0;
  while (i < n) {
    size_t e = match_at(units, i);
    MX_CHECK(e != kNpos && e > i, "pre-tokenizer 無法切分，位元組 %u", units[i].off);
    size_t b0 = units[i].off;
    size_t b1 = e < n ? units[e].off : s.size();
    bpe_append(s.substr(b0, b1 - b0), out);
    i = e;
  }
}

std::vector<int32_t> TokImpl::encode(const std::string& text) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 4 + 1);
  size_t i = 0;
  while (i < text.size()) {
    int ai = find_added(text, i);
    if (ai >= 0) {
      out.push_back(added_[ai].id);
      i += added_[ai].content.size();
      continue;
    }
    size_t j = i + 1;
    while (j < text.size() && find_added(text, j) < 0) ++j;
    std::string_view seg(text.data() + i, j - i);
    if (nfc_) pretok(unicode_nfc(seg), out);
    else {
      utf8_decode(seg);
      pretok(std::string(seg), out);
    }
    i = j;
  }
  return out;
}

std::string TokImpl::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  std::string bytes;
  auto flush = [&] {
    if (bytes.empty()) return;
    append_utf8_lossy(out, bytes);
    bytes.clear();
  };
  for (int32_t id : ids) {
    MX_CHECK(id >= 0 && size_t(id) < entries_.size() && entries_[id].seen, "未知的 token id %d", id);
    if (entries_[id].literal) {
      flush();
      out += entries_[id].text;
    } else {
      bytes += entries_[id].bytes;
    }
  }
  flush();
  return out;
}

int32_t TokImpl::token_id(const std::string& token) const {
  auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? -1 : it->second;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& tokenizer_json_path) {
  auto t = std::make_unique<TokImpl>();
  t->load_file(tokenizer_json_path);
  return t;
}

}  // namespace mx
