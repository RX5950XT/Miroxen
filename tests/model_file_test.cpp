#include "core/model_file.h"
#include "core/dequant.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

using namespace mx;
namespace {
using Bytes = std::vector<uint8_t>;
template<class T> void put(Bytes& b, T v) {
  const auto p = reinterpret_cast<const uint8_t*>(&v); b.insert(b.end(), p, p + sizeof(T));
}
void string(Bytes& b, const std::string& s) { put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }
void scalar(Bytes& b, uint32_t type) {
  switch (type) {
    case 0: put<uint8_t>(b, 250); break; case 1: put<int8_t>(b, -12); break;
    case 2: put<uint16_t>(b, 65000); break; case 3: put<int16_t>(b, -1234); break;
    case 4: put<uint32_t>(b, 4000000000U); break; case 5: put<int32_t>(b, -1234567); break;
    case 6: put<float>(b, 1.25f); break; case 7: put<uint8_t>(b, 1); break;
    case 8: string(b, "繁體中文"); break; case 10: put<uint64_t>(b, UINT64_MAX); break;
    case 11: put<int64_t>(b, INT64_MIN); break; case 12: put<double>(b, -2.5); break;
    default: throw std::runtime_error("test metadata type error");
  }
}
struct GgufFixture { Bytes bytes; size_t offset_pos, shape_pos; };
GgufFixture gguf(uint32_t version, uint32_t align, bool explicit_align) {
  Bytes b; put<uint32_t>(b, 0x46554747); put(b, version); put<uint64_t>(b, 3);
  put<uint64_t>(b, 26 + (explicit_align ? 1 : 0));
  if (explicit_align) { string(b, "general.alignment"); put<uint32_t>(b, 4); put(b, align); }
  for (uint32_t t = 0; t <= 12; ++t) if (t != 9) {
    string(b, "s" + std::to_string(t)); put(b, t); scalar(b, t);
    string(b, "a" + std::to_string(t)); put<uint32_t>(b, 9); put(b, t); put<uint64_t>(b, 2);
    // GgufValue 的 unsigned array 儲存在 vector<int64_t>，使用可表示的值。
    for (int i = 0; i < 2; ++i) { if (t == 10) put<uint64_t>(b, 9000000000ULL); else scalar(b, t); }
  }
  string(b, "nested"); put<uint32_t>(b, 9); put<uint32_t>(b, 9); put<uint64_t>(b, 2);
  put<uint32_t>(b, 5); put<uint64_t>(b, 2); put<int32_t>(b, 3); put<int32_t>(b, 4);
  put<uint32_t>(b, 9); put<uint64_t>(b, 1); put<uint32_t>(b, 8); put<uint64_t>(b, 1); string(b, "nested string");
  string(b, "after_nested"); put<uint32_t>(b, 8); string(b, "parsed");
  string(b, "model.diffusion_model.weight"); put<uint32_t>(b, 2);
  const size_t shape_pos = b.size(); put<uint64_t>(b, 32); put<uint64_t>(b, 2);
  put<uint32_t>(b, unsigned(GType::F32)); const size_t off = b.size(); put<uint64_t>(b, 0);
  string(b, "diffusion_model.quant"); put<uint32_t>(b, 2); put<uint64_t>(b, 256); put<uint64_t>(b, 1);
  put<uint32_t>(b, unsigned(GType::Q4_K)); put<uint64_t>(b, 256);
  const size_t third_offset = (256 + 144 + align - 1) / align * align;
  string(b, "model.keep"); put<uint32_t>(b, 1); put<uint64_t>(b, 1);
  put<uint32_t>(b, unsigned(GType::F32)); put<uint64_t>(b, third_offset);
  b.resize((b.size() + align - 1) / align * align);
  const size_t start = b.size();
  for (int i = 0; i < 64; ++i) put<float>(b, float(i + 1));
  b.resize(start + third_offset);
  b[start + 256] = 0; b[start + 257] = 0x34;  // d = 0.25。
  put<float>(b, -7.f);
  return {std::move(b), off, shape_pos};
}
Bytes safetensors() {
  const char* names[] = {"F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I8", "U8", "I16", "I32", "I64", "F64", "BOOL"};
  const GType types[] = {GType::F32, GType::F16, GType::BF16, GType::F8_E4M3, GType::F8_E5M2,
    GType::I8, GType::U8, GType::I16, GType::I32, GType::I64, GType::F64, GType::BOOL};
  Bytes data;
  std::string json = R"({"__metadata__":{"author":"繁體中文","escaped":"\uD83D\uDE80\n"})";
  for (size_t i = 0; i < std::size(names); ++i) {
    const auto start = data.size(), bytes = 6 * gtype_block_bytes(types[i]);
    data.resize(start + bytes, uint8_t(i + 1));
    if (i == 0) for (int k = 0; k < 6; ++k) { const float v = float(k) + 0.5f; std::memcpy(data.data() + k * 4, &v, 4); }
    const std::string name = i == 0 ? "model.diffusion_model.weight" :
                             i == 1 ? "model.keep" : i == 2 ? "custom.bf16" : "diffusion_model." + std::string(names[i]);
    json += ",\"" + name + "\":{\"dtype\":\"" + names[i] + "\",\"shape\":[2,3],\"data_offsets\":[" +
            std::to_string(start) + "," + std::to_string(start + bytes) + "]}";
  }
  const auto off = data.size(); put<float>(data, 42.f);
  json += ",\"scalar\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + 4) + "]}";
  json += ",\"empty\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[" + std::to_string(off + 4) + "," + std::to_string(off + 4) + "]}}";
  while (json.size() % 8) json += ' ';
  Bytes b; put<uint64_t>(b, json.size()); b.insert(b.end(), json.begin(), json.end()); b.insert(b.end(), data.begin(), data.end());
  return b;
}
std::string path_utf8(const std::filesystem::path& p) {
  const auto u = p.u8string(); return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}
struct Temp {
  std::filesystem::path root;
  std::vector<std::filesystem::path> files;
  Temp() {
    auto name = std::filesystem::path(u8"miroxen-讀取器-");
    name += std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    root = std::filesystem::temp_directory_path() / name;
    MX_CHECK(std::filesystem::create_directory(root), "測試暫存目錄建立失敗");
  }
  ~Temp() {
    for (const auto& p : files) { std::error_code ec; std::filesystem::remove(p, ec); }
    std::error_code ec; std::filesystem::remove(root, ec);
  }
  std::string write(const std::string& name, const Bytes& b) {
    const auto p = root / name; files.push_back(p);
    std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char*>(b.data()), b.size());
    MX_CHECK(bool(f), "測試檔寫入失敗"); return path_utf8(p);
  }
};
void rejects(const std::function<void()>& f, const std::string& text = "") {
  try { f(); } catch (const std::runtime_error& e) {
    MX_CHECK(text.empty() || std::string(e.what()).find(text) != std::string::npos, "例外訊息缺少 %s", text.c_str()); return;
  }
  throw std::runtime_error("無效檔案未丟例外");
}
void check_metadata(const TensorStore& s) {
  const int64_t ints[] = {250, -12, 65000, -1234, 4000000000LL, -1234567};
  for (int t = 0; t < 6; ++t) {
    MX_CHECK(s.meta_int("s" + std::to_string(t)) == ints[t], "scalar 整數錯誤");
    MX_CHECK(std::get<std::vector<int64_t>>(*s.meta("a" + std::to_string(t))) == std::vector<int64_t>({ints[t], ints[t]}), "array 整數錯誤");
  }
  MX_CHECK(std::get<double>(*s.meta("s6")) == 1.25 && std::get<double>(*s.meta("s12")) == -2.5, "浮點 metadata 錯誤");
  MX_CHECK(std::get<std::vector<double>>(*s.meta("a6")) == std::vector<double>({1.25, 1.25}), "f32 array 錯誤");
  MX_CHECK(std::get<std::vector<double>>(*s.meta("a12")) == std::vector<double>({-2.5, -2.5}), "f64 array 錯誤");
  MX_CHECK(s.meta_int("s7") == 1 && std::get<std::vector<int64_t>>(*s.meta("a7"))[1] == 1, "bool metadata 錯誤");
  MX_CHECK(s.meta_str("s8") == "繁體中文" && std::get<std::vector<std::string>>(*s.meta("a8"))[1] == "繁體中文", "string metadata 錯誤");
  MX_CHECK(std::get<uint64_t>(*s.meta("s10")) == UINT64_MAX && s.meta_int("s11") == INT64_MIN, "64-bit metadata 錯誤");
  MX_CHECK(std::get<std::vector<int64_t>>(*s.meta("a10"))[1] == 9000000000LL &&
           std::get<std::vector<int64_t>>(*s.meta("a11"))[1] == INT64_MIN, "64-bit array 錯誤");
  MX_CHECK(!s.meta("nested") && s.meta_str("after_nested") == "parsed", "nested array 未正確跳過");
  MX_CHECK(s.meta_str("missing", "fallback") == "fallback" && s.meta_int("missing", 7) == 7, "metadata 預設值錯誤");
  rejects([&] { s.meta_int("s10"); }, "s10");
}
void check_gguf(Temp& temp) {
  for (uint32_t version : {2, 3}) for (bool custom : {false, true}) {
    auto fixture = gguf(version, custom ? 64 : 32, custom);
    const auto file = temp.write("tiny" + std::to_string(version) + (custom ? "custom" : "") + ".gguf", fixture.bytes);
    TensorStore s; s.add_file(file);
    MX_CHECK(s.size() == 3 && s.has("model.keep") && !s.has("keep"), "GGUF 名稱錯誤");
    const auto& w = s.get("weight");
    MX_CHECK(w.shape == std::vector<int64_t>({2, 32}) && w.type == GType::F32 && w.nbytes == 256, "GGUF shape/type/size 錯誤");
    MX_CHECK(static_cast<const float*>(w.data)[0] == 1 && static_cast<const float*>(w.data)[63] == 64, "GGUF data 錯誤");
    MX_CHECK(s.get("quant").shape == std::vector<int64_t>({1, 256}) && s.get("quant").type == GType::Q4_K && s.get("quant").nbytes == 144, "GGUF quant 錯誤");
    check_metadata(s);
    rejects([&] { s.get("missing_tensor"); }, "missing_tensor"); rejects([&] { s.get("missing_tensor"); }, file);
    TensorStore moved(std::move(s)); TensorStore assigned; assigned = std::move(moved);
    MX_CHECK(assigned.get("weight").data == w.data && assigned.get("weight").numel() == 64, "move 後映射失效");
    TensorStore exact; exact.add_file(file, "model.");
    MX_CHECK(exact.has("diffusion_model.weight") && exact.has("diffusion_model.quant") && exact.has("keep"), "明示 prefix 錯誤");
  }
}
void check_safe(Temp& temp) {
  const auto file = temp.write("tiny.safetensors", safetensors()); TensorStore s; s.add_file(file);
  MX_CHECK(s.size() == 14 && s.has("model.keep") && s.has("custom.bf16") && s.get("weight").shape == std::vector<int64_t>({2, 3}), "safetensors names/shape 錯誤");
  const std::pair<const char*, GType> checks[] = {{"weight", GType::F32}, {"model.keep", GType::F16}, {"custom.bf16", GType::BF16},
    {"F8_E4M3", GType::F8_E4M3}, {"F8_E5M2", GType::F8_E5M2}, {"I8", GType::I8}, {"U8", GType::U8},
    {"I16", GType::I16}, {"I32", GType::I32}, {"I64", GType::I64}, {"F64", GType::F64}, {"BOOL", GType::BOOL}};
  for (size_t i = 0; i < std::size(checks); ++i) {
    const auto& w = s.get(checks[i].first);
    MX_CHECK(w.type == checks[i].second && w.nbytes == 6 * gtype_block_bytes(w.type), "safetensors dtype/size 錯誤");
    if (i) for (size_t j = 0; j < w.nbytes; ++j) MX_CHECK(static_cast<const uint8_t*>(w.data)[j] == i + 1, "safetensors data 錯誤");
  }
  MX_CHECK(static_cast<const float*>(s.get("weight").data)[5] == 5.5f && s.get("scalar").numel() == 1 &&
           s.get("empty").nbytes == 0 && s.get("empty").numel() == 0, "safetensors scalar/empty/data 錯誤");
  MX_CHECK(s.meta_str("author") == "繁體中文" && s.meta_str("escaped") == "\xf0\x9f\x9a\x80\n", "safetensors metadata/escape 錯誤");
  TensorStore exact; exact.add_file(file, "custom.");
  MX_CHECK(exact.has("bf16") && exact.has("model.diffusion_model.weight"), "safetensors prefix 錯誤");
  const auto gg = temp.write("override.gguf", gguf(3, 32, false).bytes);
  TensorStore merged; merged.add_file(gg); merged.add_file(file);
  MX_CHECK(merged.size() == 15 && merged.get("weight").shape == std::vector<int64_t>({2, 3}) && merged.meta_str("after_nested") == "parsed", "多檔覆蓋錯誤");
}
Bytes safe_json(const std::string& s, size_t data = 0) {
  Bytes b; put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); b.resize(b.size() + data); return b;
}
void invalid_files(Temp& temp) {
  std::vector<Bytes> badgg;
  auto good = gguf(3, 32, false);
  badgg.push_back({}); badgg.push_back(Bytes(good.bytes.begin(), good.bytes.begin() + 7));
  auto b = good.bytes; b[0] = 0; badgg.push_back(b);
  b = good.bytes; b[4] = 1; badgg.push_back(b);
  b = good.bytes; const uint64_t huge = UINT64_MAX; std::memcpy(b.data() + good.offset_pos, &huge, 8); badgg.push_back(b);
  b = good.bytes; std::memcpy(b.data() + good.shape_pos, &huge, 8); badgg.push_back(b);
  b = good.bytes; b.pop_back(); badgg.push_back(b);
  b = good.bytes; b[good.offset_pos] = 1; badgg.push_back(b);
  b = good.bytes; b[good.shape_pos] = 0; badgg.push_back(b);
  // 所有 metadata 型別均有支援；uint64 array 的範圍受共用 GgufValue 限制。
  b.clear(); put<uint32_t>(b, 0x46554747); put<uint32_t>(b, 3); put<uint64_t>(b, 0); put<uint64_t>(b, 1);
  string(b, "huge_array"); put<uint32_t>(b, 9); put<uint32_t>(b, 10); put<uint64_t>(b, 1); put<uint64_t>(b, UINT64_MAX);
  b.resize((b.size() + 31) / 32 * 32); badgg.push_back(b);
  int cases = 0;
  for (const auto& bytes : badgg) { const auto p = temp.write("bad" + std::to_string(cases++) + ".gguf", bytes); rejects([&] { TensorStore s; s.add_file(p); }); }
  const std::vector<Bytes> badsafe = {
    Bytes(8, 255), safe_json("[]"), safe_json("{\"a\":{},\"a\":{}}"), safe_json("{\"a\":{}}"),
    safe_json(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})"),
    safe_json(R"({"a":{"dtype":"NOPE","shape":[1],"data_offsets":[0,4]}})", 4),
    safe_json(R"({"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}})", 4),
    safe_json(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", 4),
    safe_json(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}})", 5),
    safe_json(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 4),
    safe_json(R"({"__metadata__":{"x":true}})"), safe_json(R"({"__metadata__":{"x":"\uD800"}})"),
    safe_json("{\"__metadata__\":{\"x\":\"\xc0\x80\"}}"), safe_json("{} garbage"), safe_json("{,}")};
  for (const auto& bytes : badsafe) { const auto p = temp.write("bad" + std::to_string(cases++) + ".safetensors", bytes); rejects([&] { TensorStore s; s.add_file(p); }); }
  TensorStore stable; stable.add_file(temp.write("stable.gguf", good.bytes));
  rejects([&] { stable.add_file(""); });
  rejects([&] { stable.add_file(std::string("bad\0.gguf", 9)); });
  const auto old = stable.get("weight").data;
  rejects([&] { stable.add_file(temp.write("transaction.safetensors", safe_json("{}", 4))); });
  MX_CHECK(stable.size() == 3 && stable.get("weight").data == old && stable.meta_str("after_nested") == "parsed", "解析失敗修改了既有 store");
  std::printf("Malformed files: %d rejected; failed add preserves store PASS\n", cases);
}
void real_model() {
  const std::string path = "D:/Workspace/Personal_Project/Miroxen/models/qwen-image-2.1/qwen_image_2.1-Q8_0.gguf";
  if (!std::filesystem::exists(path)) { std::puts("Real Qwen GGUF absent: SKIP"); return; }
  TensorStore s; s.add_file(path); const auto names = s.names();
  MX_CHECK(!names.empty(), "真實模型沒有張量"); std::printf("Real Qwen GGUF: %zu tensors\n", s.size());
  for (size_t i = 0; i < std::min<size_t>(3, names.size()); ++i) {
    const auto& w = s.get(names[i]); std::printf("  %s %s [", w.name.c_str(), gtype_name(w.type));
    for (size_t j = 0; j < w.shape.size(); ++j) std::printf("%s%lld", j ? "," : "", (long long)w.shape[j]);
    std::printf("] %zu bytes\n", w.nbytes);
  }
  const WeightRef* chosen = nullptr;
  for (const auto& name : names) {
    const auto& w = s.get(name);
    if (w.type == GType::Q8_0 && (!chosen || w.numel() < chosen->numel())) chosen = &w;
  }
  MX_CHECK(chosen, "真實模型沒有適合測試的 Q8_0 張量"); const auto& w = *chosen;
  MX_CUDA(cudaSetDevice(0)); DevBuf src(w.nbytes, 0), dst(w.numel() * 4, 0);
  MX_CUDA(cudaMemcpy(src.get(), w.data, w.nbytes, cudaMemcpyHostToDevice));
  dequant_to_f32(src.get(), w.type, w.numel(), dst.as<float>(), nullptr);
  std::vector<float> f(w.numel()), cpu(w.numel());
  MX_CUDA(cudaMemcpy(f.data(), dst.get(), f.size() * 4, cudaMemcpyDeviceToHost)); dequant_cpu(w.data, w.type, w.numel(), cpu.data());
  MX_CHECK(std::memcmp(f.data(), cpu.data(), f.size() * 4) == 0, "真實張量 GPU/CPU 不一致");
  double sum = 0, sq = 0;
  for (float v : f) { MX_CHECK(std::isfinite(v), "真實張量反量化有非有限值"); sum += v; sq += double(v) * v; }
  const auto mean = sum / f.size();
  std::printf("Real GPU dequant %s (%zu elements): mean=%.9g std=%.9g, exact CPU match PASS\n", w.name.c_str(), f.size(), mean, std::sqrt(std::max(0.0, sq / f.size() - mean * mean)));
}
}  // namespace
int main() {
  try {
    Temp temp; check_gguf(temp); check_safe(temp); invalid_files(temp); real_model();
    std::puts("model_file_test PASS: GGUF v2/v3, metadata 12 scalar+12 array+nested, safetensors 12 dtypes, moves/prefix/merge"); return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "model_file_test FAIL: %s\n", e.what()); return 1; }
}
