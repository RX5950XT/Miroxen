#include "core/model_file.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mx {
namespace {
std::filesystem::path utf8_path(const std::string& path) {
  return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size()));
}
}  // namespace
class MappedFile {
 public:
  explicit MappedFile(const std::string& path) : path(path) {
    MX_CHECK(!path.empty() && path.find('\0') == std::string::npos, "模型路徑不可為空或包含 NUL");
    try {
#ifdef _WIN32
      auto wide = utf8_path(path).wstring();
      file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      MX_CHECK(file_ != INVALID_HANDLE_VALUE, "無法開啟 %s (Windows %lu)", path.c_str(), GetLastError());
      LARGE_INTEGER length;
      MX_CHECK(GetFileSizeEx(file_, &length), "無法取得檔案大小：%s", path.c_str());
      MX_CHECK(length.QuadPart > 0 && uint64_t(length.QuadPart) <= SIZE_MAX, "檔案大小無效：%s", path.c_str());
      size = size_t(length.QuadPart);
      mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
      MX_CHECK(mapping_, "無法映射 %s (Windows %lu)", path.c_str(), GetLastError());
      data = static_cast<const uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
      MX_CHECK(data, "無法映射 %s (Windows %lu)", path.c_str(), GetLastError());
#else
      file_ = open(path.c_str(), O_RDONLY);
      MX_CHECK(file_ >= 0, "無法開啟 %s：%s", path.c_str(), strerror(errno));
      struct stat st;
      MX_CHECK(fstat(file_, &st) == 0 && st.st_size > 0 && uint64_t(st.st_size) <= SIZE_MAX,
               "檔案大小無效：%s", path.c_str());
      size = size_t(st.st_size);
      void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_, 0);
      MX_CHECK(p != MAP_FAILED, "無法映射 %s：%s", path.c_str(), strerror(errno));
      data = static_cast<const uint8_t*>(p);
#endif
    } catch (...) { release(); throw; }
  }
  ~MappedFile() { release(); }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const std::string path;
  const uint8_t* data = nullptr;
  size_t size = 0;
 private:
  void release() noexcept {
#ifdef _WIN32
    if (data) UnmapViewOfFile(data);
    if (mapping_) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
    if (data) munmap(const_cast<uint8_t*>(data), size);
    if (file_ >= 0) close(file_);
#endif
  }
#ifdef _WIN32
  HANDLE file_ = INVALID_HANDLE_VALUE, mapping_ = nullptr;
#else
  int file_ = -1;
#endif
};

namespace {
struct Cursor {
  const uint8_t* data;
  size_t size, pos = 0;
  const uint8_t* take(uint64_t n) {
    MX_CHECK(n <= size - pos, "模型檔截斷：offset=%llu，需要 %llu bytes", (unsigned long long)pos,
             (unsigned long long)n);
    auto p = data + pos;
    pos += size_t(n);
    return p;
  }
  template<class T> T read() {
    T v;
    auto p = take(sizeof(T));
    std::memcpy(&v, p, sizeof(T));
    if constexpr (std::endian::native != std::endian::little)
      std::reverse(reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + sizeof(T));
    return v;
  }
  std::string string() {
    const auto n = read<uint64_t>();
    const auto p = take(n);
    return std::string(reinterpret_cast<const char*>(p), size_t(n));
  }
};
using Metadata = std::map<std::string, GgufValue>;
using Weights = std::map<std::string, WeightRef>;

GgufValue scalar(Cursor& c, uint32_t t) {
  switch (t) {
    case 0: return uint64_t(c.read<uint8_t>());
    case 1: return int64_t(c.read<int8_t>());
    case 2: return uint64_t(c.read<uint16_t>());
    case 3: return int64_t(c.read<int16_t>());
    case 4: return uint64_t(c.read<uint32_t>());
    case 5: return int64_t(c.read<int32_t>());
    case 6: return double(c.read<float>());
    case 7: { const auto v = c.read<uint8_t>(); MX_CHECK(v <= 1, "GGUF bool 無效"); return bool(v); }
    case 8: return c.string();
    case 10: return c.read<uint64_t>();
    case 11: return c.read<int64_t>();
    case 12: return c.read<double>();
    default: throw std::runtime_error(strf("GGUF metadata 型別無效：%u", t));
  }
}

std::optional<GgufValue> value(Cursor& c, uint32_t t, int depth = 0, bool keep = true) {
  MX_CHECK(depth < 64, "GGUF array 巢狀過深");
  if (t != 9) { auto v = scalar(c, t); return keep ? std::optional<GgufValue>(std::move(v)) : std::nullopt; }
  const auto et = c.read<uint32_t>();
  const auto n = c.read<uint64_t>();
  MX_CHECK(et <= 12 && n <= c.size - c.pos, "GGUF array 型別或長度無效");
  std::vector<int64_t> ints;
  std::vector<double> floats;
  std::vector<std::string> strings;
  for (uint64_t i = 0; i < n; ++i) {
    auto v = value(c, et, depth + 1, keep && et != 9);
    if (!v) continue;
    if (auto p = std::get_if<int64_t>(&*v)) ints.push_back(*p);
    else if (auto p = std::get_if<uint64_t>(&*v)) {
      // 目前共用介面沒有 vector<uint64_t>；不可悄悄截斷合法的大整數。
      MX_CHECK(*p <= INT64_MAX, "GGUF uint64 array 超出 GgufValue int64 範圍");
      ints.push_back(int64_t(*p));
    } else if (auto p = std::get_if<bool>(&*v)) ints.push_back(*p);
    else if (auto p = std::get_if<double>(&*v)) floats.push_back(*p);
    else if (auto p = std::get_if<std::string>(&*v)) strings.push_back(std::move(*p));
  }
  if (!keep || et == 9) return std::nullopt;
  if (et == 8) return strings;
  if (et == 6 || et == 12) return floats;
  return ints;
}

std::string strip(std::string name, const std::string& prefix) {
  if (!prefix.empty()) { if (name.starts_with(prefix)) name.erase(0, prefix.size()); }
  else for (std::string_view p : {"model.diffusion_model.", "diffusion_model."})
    if (name.starts_with(p)) { name.erase(0, p.size()); break; }
  MX_CHECK(!name.empty(), "張量名稱不可為空");
  return name;
}

size_t tensor_bytes(const WeightRef& w, bool gguf) {
  int64_t n = 1;
  for (const auto d : w.shape) {
    MX_CHECK(d >= 0 && (!gguf || d > 0), "張量 %s 維度無效", w.name.c_str());
    MX_CHECK(d == 0 || n <= INT64_MAX / d, "張量 %s 大小溢位", w.name.c_str());
    n *= d;
  }
  const auto block = gtype_block(w.type);
  const auto bytes = gtype_block_bytes(w.type);
  MX_CHECK(n % block == 0 && (!gguf || w.shape.back() % block == 0),
           "張量 %s 未對齊量化區塊", w.name.c_str());
  MX_CHECK(uint64_t(n / block) <= SIZE_MAX / bytes, "張量 %s byte size 溢位", w.name.c_str());
  return size_t(n / block) * bytes;
}

void read_gguf(const MappedFile& f, const std::string& prefix, Weights& out, Metadata& meta) {
  Cursor c{f.data, f.size};
  MX_CHECK(c.read<uint32_t>() == 0x46554747, "GGUF magic 無效");
  const auto version = c.read<uint32_t>();
  MX_CHECK(version == 2 || version == 3, "不支援 GGUF version %u", version);
  const auto nt = c.read<uint64_t>(), nk = c.read<uint64_t>();
  MX_CHECK(nt <= (f.size - c.pos) / 24 && nk <= (f.size - c.pos) / 13, "GGUF header count 無效");
  for (uint64_t i = 0; i < nk; ++i) {
    auto key = c.string();
    MX_CHECK(!meta.contains(key), "GGUF metadata 重複：%s", key.c_str());
    const auto type = c.read<uint32_t>();
    auto v = value(c, type);
    if (v) meta.emplace(std::move(key), std::move(*v));
  }
  uint64_t alignment = 32;
  if (auto it = meta.find("general.alignment"); it != meta.end()) {
    if (auto p = std::get_if<uint64_t>(&it->second)) alignment = *p;
    else if (auto p = std::get_if<int64_t>(&it->second)) { MX_CHECK(*p > 0, "GGUF alignment 無效"); alignment = *p; }
    else throw std::runtime_error("GGUF alignment 必須為整數");
  }
  MX_CHECK(alignment && !(alignment & (alignment - 1)) && alignment <= SIZE_MAX,
           "GGUF alignment 必須為 2 的次方");
  std::vector<std::pair<std::string, uint64_t>> offsets;
  for (uint64_t i = 0; i < nt; ++i) {
    WeightRef w;
    w.name = strip(c.string(), prefix);
    const auto rank = c.read<uint32_t>();
    // ComfyUI 轉出的 Conv3d（Qwen3-VL patch_embed）是 5 維；記憶體排列與 4 維合併版相同
    MX_CHECK(rank >= 1 && rank <= 5, "GGUF 張量 %s rank 無效：%u", w.name.c_str(), rank);
    for (uint32_t j = 0; j < rank; ++j) {
      const auto d = c.read<uint64_t>();
      MX_CHECK(d <= INT64_MAX, "GGUF 維度超出 int64 範圍");
      w.shape.push_back(int64_t(d));
    }
    std::reverse(w.shape.begin(), w.shape.end());
    w.type = GType(c.read<uint32_t>());
    MX_CHECK(unsigned(w.type) < 1000, "GGUF 張量型別無效");
    w.nbytes = tensor_bytes(w, true);
    const auto off = c.read<uint64_t>();
    MX_CHECK(off % alignment == 0, "GGUF 張量 %s offset 未對齊", w.name.c_str());
    offsets.emplace_back(w.name, off);
    MX_CHECK(out.emplace(w.name, std::move(w)).second, "GGUF 張量名稱重複");
  }
  MX_CHECK(c.pos <= SIZE_MAX - (alignment - 1), "GGUF data offset 溢位");
  const auto start = (c.pos + alignment - 1) & ~(alignment - 1);
  MX_CHECK(start <= f.size, "GGUF data offset 超過檔案大小");
  for (const auto& [name, off] : offsets) {
    auto& w = out.at(name);
    MX_CHECK(off <= f.size - start && w.nbytes <= f.size - start - off,
             "GGUF 張量 %s 超過檔案大小", name.c_str());
    w.data = f.data + start + off;
  }
}

// JSON number 保留原字串；shape/offset 不經 double，避免大整數失去精度。
struct Json {
  enum Kind { Object, Array, String, Number, Literal } kind;
  std::string text;
  std::map<std::string, Json> object;
  std::vector<Json> array;
  const Json& at(const char* key) const {
    auto it = object.find(key);
    MX_CHECK(kind == Object && it != object.end(), "safetensors 缺少 %s", key);
    return it->second;
  }
  uint64_t integer() const {
    uint64_t v;
    const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    MX_CHECK(kind == Number && r.ec == std::errc{} && r.ptr == text.data() + text.size(),
             "safetensors 必須使用非負整數：%s", text.c_str());
    return v;
  }
};
class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : s_(text) {}
  Json parse() { auto v = item(0); space(); MX_CHECK(p_ == s_.size(), "JSON 結尾有多餘內容"); return v; }
 private:
  std::string_view s_;
  size_t p_ = 0;
  char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }
  char next() { MX_CHECK(p_ < s_.size(), "JSON 截斷"); return s_[p_++]; }
  void space() { while (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n') ++p_; }
  void expect(char c) { MX_CHECK(next() == c, "JSON 預期字元 %c", c); }
  uint32_t hex4() {
    uint32_t n = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = next();
      const int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                    c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      MX_CHECK(v >= 0, "JSON Unicode escape 無效");
      n = n * 16 + v;
    }
    return n;
  }
  static void utf8(std::string& out, uint32_t u) {
    if (u <= 0x7f) out += char(u);
    else if (u <= 0x7ff) { out += char(0xc0 | u >> 6); out += char(0x80 | (u & 63)); }
    else if (u <= 0xffff) { out += char(0xe0 | u >> 12); out += char(0x80 | (u >> 6 & 63)); out += char(0x80 | (u & 63)); }
    else { out += char(0xf0 | u >> 18); out += char(0x80 | (u >> 12 & 63));
           out += char(0x80 | (u >> 6 & 63)); out += char(0x80 | (u & 63)); }
  }
  std::string string() {
    expect('"');
    std::string out;
    while (peek() != '"') {
      const auto c = uint8_t(next());
      MX_CHECK(c >= 32, "JSON string 控制字元無效");
      if (c != '\\') {
        if (c < 128) out += char(c);
        else raw_utf8(out, c);
        continue;
      }
      switch (next()) {
        case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
        case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
        case 'r': out += '\r'; break; case 't': out += '\t'; break;
        case 'u': {
          auto u = hex4();
          if (u >= 0xd800 && u <= 0xdbff) {
            expect('\\'); expect('u'); const auto low = hex4();
            MX_CHECK(low >= 0xdc00 && low <= 0xdfff, "JSON surrogate pair 無效");
            u = 0x10000 + ((u - 0xd800) << 10) + low - 0xdc00;
          } else MX_CHECK(u < 0xdc00 || u > 0xdfff, "JSON 孤立 surrogate 無效");
          utf8(out, u); break;
        }
        default: throw std::runtime_error("JSON escape 無效");
      }
    }
    expect('"'); return out;
  }
  void raw_utf8(std::string& out, uint8_t c) {
    const int count = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
    MX_CHECK(count > 0, "JSON UTF-8 無效");
    uint32_t u = c & (count == 1 ? 31 : count == 2 ? 15 : 7);
    for (int i = 0; i < count; ++i) {
      const auto b = uint8_t(next()); MX_CHECK((b & 0xc0) == 0x80, "JSON UTF-8 無效"); u = (u << 6) | (b & 63);
    }
    MX_CHECK(u >= (count == 1 ? 0x80u : count == 2 ? 0x800u : 0x10000u) && u <= 0x10ffff &&
             (u < 0xd800 || u > 0xdfff), "JSON UTF-8 無效");
    utf8(out, u);
  }
  Json number() {
    const auto start = p_;
    if (peek() == '-') ++p_;
    if (peek() == '0') ++p_;
    else { MX_CHECK(peek() >= '1' && peek() <= '9', "JSON number 無效"); digits(); }
    if (peek() == '.') { ++p_; MX_CHECK(digit(), "JSON fraction 無效"); digits(); }
    if (peek() == 'e' || peek() == 'E') {
      ++p_; if (peek() == '+' || peek() == '-') ++p_;
      MX_CHECK(digit(), "JSON exponent 無效"); digits();
    }
    Json v{Json::Number}; v.text = s_.substr(start, p_ - start); return v;
  }
  bool digit() const { return peek() >= '0' && peek() <= '9'; }
  void digits() { while (digit()) ++p_; }
  Json item(int depth) {
    MX_CHECK(depth < 64, "JSON 巢狀過深"); space();
    if (peek() == '"') { Json v{Json::String}; v.text = string(); return v; }
    if (peek() != '{' && peek() != '[') {
      for (std::string_view lit : {"true", "false", "null"}) if (s_.substr(p_).starts_with(lit)) {
        p_ += lit.size(); Json v{Json::Literal}; v.text = lit; return v;
      }
      return number();
    }
    const bool obj = next() == '{'; const char end = obj ? '}' : ']';
    Json v{obj ? Json::Object : Json::Array}; space();
    if (peek() == end) { ++p_; return v; }
    for (;;) {
      if (obj) {
        auto key = string(); space(); expect(':'); auto child = item(depth + 1);
        MX_CHECK(v.object.emplace(std::move(key), std::move(child)).second, "JSON key 重複");
      } else v.array.push_back(item(depth + 1));
      space(); const auto c = next(); if (c == end) break;
      MX_CHECK(c == ',', "JSON 分隔符號無效"); space();
    }
    return v;
  }
};

GType dtype(const Json& v) {
  MX_CHECK(v.kind == Json::String, "safetensors dtype 必須為字串");
  const std::pair<std::string_view, GType> types[] = {
    {"F32", GType::F32}, {"F16", GType::F16}, {"BF16", GType::BF16}, {"F8_E4M3", GType::F8_E4M3},
    {"F8_E5M2", GType::F8_E5M2}, {"I8", GType::I8}, {"U8", GType::U8}, {"I16", GType::I16},
    {"I32", GType::I32}, {"I64", GType::I64}, {"F64", GType::F64}, {"BOOL", GType::BOOL}};
  for (const auto& [name, t] : types) if (v.text == name) return t;
  throw std::runtime_error(strf("不支援 safetensors dtype：%s", v.text.c_str()));
}

void read_safe(const MappedFile& f, const std::string& prefix, Weights& out, Metadata& meta) {
  Cursor c{f.data, f.size};
  const auto len = c.read<uint64_t>();
  MX_CHECK(len > 0 && len <= 100000000 && len <= f.size - c.pos, "safetensors header 長度無效");
  const auto p = c.take(len);
  auto root = JsonParser(std::string_view(reinterpret_cast<const char*>(p), size_t(len))).parse();
  MX_CHECK(root.kind == Json::Object, "safetensors header 必須為 object");
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const auto& [name, obj] : root.object) {
    MX_CHECK(obj.kind == Json::Object, "safetensors tensor 必須為 object");
    if (name == "__metadata__") {
      for (const auto& [key, v] : obj.object) {
        MX_CHECK(v.kind == Json::String, "safetensors metadata 必須為字串"); meta.emplace(key, v.text);
      }
      continue;
    }
    WeightRef w; w.name = strip(name, prefix); w.type = dtype(obj.at("dtype"));
    const auto& shape = obj.at("shape");
    MX_CHECK(shape.kind == Json::Array, "safetensors shape 必須為 array");
    for (const auto& dim : shape.array) {
      const auto d = dim.integer(); MX_CHECK(d <= INT64_MAX, "safetensors shape 溢位"); w.shape.push_back(int64_t(d));
    }
    w.nbytes = tensor_bytes(w, false);
    const auto& offsets = obj.at("data_offsets");
    MX_CHECK(offsets.kind == Json::Array && offsets.array.size() == 2, "safetensors data_offsets 無效");
    const auto start = offsets.array[0].integer(), end = offsets.array[1].integer();
    MX_CHECK(start <= end && end <= f.size - c.pos && end - start == w.nbytes,
             "safetensors 張量 %s offset/大小無效", name.c_str());
    w.data = f.data + c.pos + start; ranges.emplace_back(start, end);
    MX_CHECK(out.emplace(w.name, std::move(w)).second, "safetensors 剝除前綴後名稱重複");
  }
  std::sort(ranges.begin(), ranges.end());
  uint64_t end = 0;
  for (const auto& r : ranges) { MX_CHECK(r.first == end, "safetensors 資料重疊或有空洞"); end = r.second; }
  MX_CHECK(end == f.size - c.pos, "safetensors 結尾有未標示資料");
}
}  // namespace

TensorStore::TensorStore() = default;
TensorStore::~TensorStore() = default;
TensorStore::TensorStore(TensorStore&&) noexcept = default;
TensorStore& TensorStore::operator=(TensorStore&&) noexcept = default;

void TensorStore::add_file(const std::string& path, const std::string& prefix) {
  auto f = std::make_unique<MappedFile>(path);
  Weights tensors; Metadata metadata;
  auto ext = utf8_path(path).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return c >= 'A' && c <= 'Z' ? char(c + 32) : char(c); });
  try {
    if (ext == ".gguf") read_gguf(*f, prefix, tensors, metadata);
    else if (ext == ".safetensors") read_safe(*f, prefix, tensors, metadata);
    else throw std::runtime_error("不支援的模型副檔名：" + ext);
  } catch (const std::exception& e) { throw std::runtime_error(path + ": " + e.what()); }
  // 解析全部成功才加入，失敗不留下指向已釋放映射的 WeightRef。
  files_.push_back(std::move(f));
  for (auto& [name, w] : tensors) {
    if (tensors_.contains(name)) MX_DEBUG("張量 %s 被 %s 覆蓋", name.c_str(), path.c_str());
    tensors_.insert_or_assign(name, std::move(w));
  }
  for (auto& [key, v] : metadata) meta_.insert_or_assign(key, std::move(v));
}
const WeightRef* TensorStore::find(const std::string& name) const {
  const auto it = tensors_.find(name); return it == tensors_.end() ? nullptr : &it->second;
}
const WeightRef& TensorStore::get(const std::string& name) const {
  if (auto p = find(name)) return *p;
  std::string files;
  for (const auto& f : files_) { if (!files.empty()) files += ", "; files += f->path; }
  throw std::runtime_error("找不到張量 " + name + "；檔案：[" + files + "]");
}
std::vector<std::string> TensorStore::names() const {
  std::vector<std::string> result;
  for (const auto& [name, w] : tensors_) result.push_back(name);
  return result;
}
const GgufValue* TensorStore::meta(const std::string& key) const {
  const auto it = meta_.find(key); return it == meta_.end() ? nullptr : &it->second;
}
std::string TensorStore::meta_str(const std::string& key, const std::string& def) const {
  if (auto v = meta(key)) if (auto p = std::get_if<std::string>(v)) return *p;
  return def;
}
int64_t TensorStore::meta_int(const std::string& key, int64_t def) const {
  if (auto v = meta(key)) {
    if (auto p = std::get_if<int64_t>(v)) return *p;
    if (auto p = std::get_if<uint64_t>(v)) { MX_CHECK(*p <= INT64_MAX, "metadata %s 超出 int64", key.c_str()); return int64_t(*p); }
    if (auto p = std::get_if<bool>(v)) return *p;
  }
  return def;
}
}  // namespace mx
