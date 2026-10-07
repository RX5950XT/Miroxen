#include "server/server_internal.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "core/common.h"
#include "core/model_file.h"
#include "models/minimax_h3/pipeline.h"
#include "server/image_io.h"

#include "nlohmann/json.hpp"

namespace mx {
namespace {

using json = nlohmann::json;

std::string trim(std::string s) {
  auto sp = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (!s.empty() && sp(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && sp(static_cast<unsigned char>(s.back()))) s.erase(s.end() - 1);
  return s;
}

size_t utf8_len(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) ++n;
  return n;
}

bool ends_with_ci(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  for (size_t i = 0; i < suf.size(); ++i) {
    char a = s[s.size() - suf.size() + i];
    char b = suf[i];
    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
    if (a != b) return false;
  }
  return true;
}

bool lora_name_ok(const std::string& name) {
  if (name.empty() || name.size() > 240 || name == "." || name == "..") return false;
  if (name.find("..") != std::string::npos) return false;
  if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
      name.find(':') != std::string::npos)
    return false;
  for (unsigned char c : name)
    if (c < 0x20 || c == 0x7f) return false;
  return ends_with_ci(name, ".safetensors");
}

std::string resolve_lora(const ServerOptions& opt, const std::string& name) {
  if (!lora_name_ok(name))
    throw ApiError(400, "invalid_request_error", "invalid_lora", "LoRA 名稱不合法");
  std::error_code ec;
  auto base = std::filesystem::weakly_canonical(utf8_path(opt.lora_dir), ec);
  if (ec) throw ApiError(400, "invalid_request_error", "lora_not_found", "找不到 LoRA 目錄");
  auto full = std::filesystem::weakly_canonical(base / utf8_path(name), ec);
  if (ec || full.parent_path() != base)
    throw ApiError(400, "invalid_request_error", "invalid_lora", "LoRA 路徑超出目錄");
  if (!std::filesystem::is_regular_file(full))
    throw ApiError(400, "invalid_request_error", "lora_not_found", "找不到 LoRA");
  return full.string();
}

std::string require_text(const json& j, const char* key, size_t max_chars, bool allow_empty) {
  if (!j.contains(key)) {
    if (allow_empty) return {};
    throw ApiError(400, "invalid_request_error", "missing_field", strf("缺少 %s", key));
  }
  if (!j[key].is_string())
    throw ApiError(400, "invalid_request_error", "invalid_type", strf("%s 必須是字串", key));
  auto s = trim(j[key].get<std::string>());
  if (!allow_empty && s.empty())
    throw ApiError(400, "invalid_request_error", "empty_prompt", strf("%s 不能是空的", key));
  if (utf8_len(s) > max_chars)
    throw ApiError(400, "invalid_request_error", "prompt_too_long", strf("%s 超過 %zu 字", key, max_chars));
  return s;
}

int read_int(const json& j, const char* key, int fallback) {
  if (!j.contains(key)) return fallback;
  if (!j[key].is_number_integer())
    throw ApiError(400, "invalid_request_error", "invalid_type", strf("%s 必須是整數", key));
  return j[key].get<int>();
}

// size 為 "auto" 時 w = h = 0（由呼叫端決定是否允許：編輯依參考圖、影片依關鍵幀或官方畫布）。
void parse_wh(const json& j, int& w, int& h) {
  if (j.contains("size")) {
    if (!j["size"].is_string())
      throw ApiError(400, "invalid_request_error", "invalid_size", "size 必須是字串");
    auto s = j["size"].get<std::string>();
    if (s == "auto") {
      w = h = 0;
      return;
    }
    auto x = s.find('x');
    if (x == std::string::npos) x = s.find('X');
    auto a = x == std::string::npos ? std::string() : s.substr(0, x);
    auto b = x == std::string::npos ? std::string() : s.substr(x + 1);
    auto digits = [](const std::string& t) {
      return !t.empty() && std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    if (!digits(a) || !digits(b))
      throw ApiError(400, "invalid_request_error", "invalid_size", "size 格式應為 寬x高，例如 1024x1024");
    try {
      w = std::stoi(a);
      h = std::stoi(b);
    } catch (...) {
      throw ApiError(400, "invalid_request_error", "invalid_size", "size 無法解析");
    }
    return;
  }
  if (!j.contains("width") || !j.contains("height"))
    throw ApiError(400, "invalid_request_error", "invalid_size", "需要 size 或 width 與 height");
  w = read_int(j, "width", 0);
  h = read_int(j, "height", 0);
}

void check_dims(int w, int h, int max_side) {
  if (w == 0 && h == 0) return;  // auto
  auto bad = [&](int v) { return v < 256 || v > max_side || v % 32 != 0; };
  if (bad(w) || bad(h))
    throw ApiError(400, "invalid_request_error", "invalid_size",
                   strf("寬與高必須是 32 的倍數，且介於 256 與 %d", max_side));
}

std::string job_type(const json& j, const std::string& forced) {
  if (!forced.empty()) {
    if (j.contains("type") && j["type"].is_string() && j["type"].get<std::string>() != forced)
      throw ApiError(400, "invalid_request_error", "invalid_type", "type 與此端點不符");
    return forced;
  }
  if (!j.contains("type") || !j["type"].is_string())
    throw ApiError(400, "invalid_request_error", "missing_field", "缺少 type");
  auto t = j["type"].get<std::string>();
  if (t != "image" && t != "video")
    throw ApiError(400, "invalid_request_error", "invalid_type", "type 必須是 image 或 video");
  return t;
}

std::map<std::string, std::string> check_model(Engine& engine, const json& j, const std::string& type) {
  if (!j.contains("model") || !j["model"].is_string() || trim(j["model"].get<std::string>()).empty())
    throw ApiError(400, "invalid_request_error", "missing_model", "需要 model");
  auto id = trim(j["model"].get<std::string>());
  bool matched = false;
  std::string kind;
  ModelInfo model;
  for (const auto& m : engine.models()) {
    if (m.id != id) continue;
    matched = true;
    kind = m.kind;
    model = m;
    break;
  }
  if (!matched) throw ApiError(400, "invalid_request_error", "model_not_found", "找不到模型");
  if (kind != type) throw ApiError(400, "invalid_request_error", "model_kind", "模型種類與此請求不符");
  // 引擎按需載入模型（第一次請求會回報 "load" 階段），所以不檢查 loaded。
  std::map<std::string, std::string> selected;
  if (!j.contains("model_files")) return selected;
  if (!j["model_files"].is_object())
    throw ApiError(400, "invalid_request_error", "invalid_model_files", "model_files 必須是物件");
  for (const auto& [role, value] : j["model_files"].items()) {
    auto it = model.files.find(role);
    if (!value.is_string() || it == model.files.end() ||
        std::find(it->second.begin(), it->second.end(), value.get<std::string>()) == it->second.end())
      throw ApiError(400, "invalid_request_error", "invalid_model_files", "只能選擇 /v1/models 列出的模型檔");
    selected[role] = value.get<std::string>();
  }
  return selected;
}

std::vector<LoraSpec> read_loras(const ServerOptions& opt, const json& j) {
  std::vector<LoraSpec> out;
  if (!j.contains("loras")) return out;
  if (!j["loras"].is_array())
    throw ApiError(400, "invalid_request_error", "invalid_lora", "loras 必須是陣列");
  for (auto& item : j["loras"]) {
    if (!item.is_object() || !item.contains("name") || !item["name"].is_string())
      throw ApiError(400, "invalid_request_error", "invalid_lora", "LoRA 需要 name");
    float scale = 1.f;
    if (item.contains("scale")) {
      if (!item["scale"].is_number())
        throw ApiError(400, "invalid_request_error", "invalid_lora", "LoRA scale 必須是數字");
      scale = item["scale"].get<float>();
      if (!std::isfinite(scale))
        throw ApiError(400, "invalid_request_error", "invalid_lora", "LoRA scale 不是有限數字");
    }
    out.push_back({resolve_lora(opt, item["name"].get<std::string>()), scale});
  }
  return out;
}

std::string payload_b64(const std::string& in) {
  if (in.rfind("data:", 0) != 0) return in;
  auto comma = in.find(',');
  if (comma == std::string::npos || in.substr(0, comma).find("base64") == std::string::npos)
    throw std::runtime_error("data URL 必須是 base64");
  return in.substr(comma + 1);
}

struct TempGuard {
  std::vector<std::filesystem::path> paths;
  bool keep = false;
  ~TempGuard() {
    if (keep) return;
    for (auto& p : paths) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  }
};

std::filesystem::path save_upload(const ServerOptions& opt, const std::string& encoded, TempGuard& guard) {
  std::vector<uint8_t> raw;
  try {
    raw = b64_decode(payload_b64(encoded));
  } catch (const std::exception& e) {
    throw ApiError(400, "invalid_request_error", "invalid_image", e.what());
  }
  DecodedImage img;
  try {
    img = decode_image_bytes(raw.data(), raw.size());
  } catch (const std::exception& e) {
    throw ApiError(400, "invalid_request_error", "invalid_image", e.what());
  }
  auto png = encode_png(img.pixels.data(), img.width, img.height, img.channels);
  static std::atomic<uint64_t> seq{1};
  auto path = utf8_path(opt.out_dir) / "tmp" / ("ref_" + std::to_string(seq.fetch_add(1)) + ".png");
  guard.paths.push_back(path);
  try {
    write_file(path, png.data(), png.size());
  } catch (const std::exception& e) {
    throw ApiError(500, "server_error", "write_failed", e.what());
  }
  return path;
}

void read_uploads(const ServerOptions& opt, const json& j, const char* key, std::vector<std::string>& paths,
                  TempGuard& guard, size_t cap) {
  if (!j.contains(key)) return;
  if (!j[key].is_array())
    throw ApiError(400, "invalid_request_error", "invalid_image", strf("%s 必須是陣列", key));
  if (j[key].size() > cap)
    throw ApiError(400, "invalid_request_error", "invalid_image", strf("%s 最多 %zu 張", key, cap));
  for (auto& item : j[key]) {
    if (!item.is_string())
      throw ApiError(400, "invalid_request_error", "invalid_image", "圖片必須是 base64 字串");
    paths.push_back(save_upload(opt, item.get<std::string>(), guard).string());
  }
}

// 影片／音訊上傳：不解碼（之後由 ffprobe 檢查），副檔名取自 data URL 的 MIME。
std::filesystem::path save_media(const ServerOptions& opt, const std::string& encoded, TempGuard& guard) {
  std::vector<uint8_t> raw;
  try {
    raw = b64_decode(payload_b64(encoded));
  } catch (const std::exception& e) {
    throw ApiError(400, "invalid_request_error", "invalid_media", e.what());
  }
  if (raw.size() < 64) throw ApiError(400, "invalid_request_error", "invalid_media", "媒體檔太小或是空的");
  std::string ext = ".bin";
  static const std::pair<const char*, const char*> kMime[] = {
      {"video/mp4", ".mp4"}, {"video/webm", ".webm"}, {"video/quicktime", ".mov"}, {"audio/wav", ".wav"},
      {"audio/x-wav", ".wav"}, {"audio/mpeg", ".mp3"}, {"audio/mp4", ".m4a"}, {"audio/ogg", ".ogg"},
      {"audio/flac", ".flac"}};
  for (auto& [mime, e] : kMime)
    if (encoded.rfind(std::string("data:") + mime, 0) == 0) ext = e;
  static std::atomic<uint64_t> seq{1};
  auto path = utf8_path(opt.out_dir) / "tmp" / ("media_" + std::to_string(seq.fetch_add(1)) + ext);
  guard.paths.push_back(path);
  try {
    write_file(path, raw.data(), raw.size());
  } catch (const std::exception& e) {
    throw ApiError(500, "server_error", "write_failed", e.what());
  }
  return path;
}

// Ref2VA 參考：[{ "type": "image"|"video"|"audio", "data": base64 或 data URL }, ...]，依提示詞順序。
void read_references(const ServerOptions& opt, const json& j, std::vector<MediaRef>& out, TempGuard& guard) {
  if (!j.contains("references") || j["references"].is_null()) return;
  if (!j["references"].is_array() || j["references"].size() > 12)
    throw ApiError(400, "invalid_request_error", "invalid_references", "references 必須是最多 12 個元素的陣列");
  for (auto& it : j["references"]) {
    if (!it.is_object() || !it.contains("type") || !it["type"].is_string() || !it.contains("data") ||
        !it["data"].is_string())
      throw ApiError(400, "invalid_request_error", "invalid_references", "每個參考需要字串欄位 type 與 data");
    auto kind = it["type"].get<std::string>();
    if (kind != "image" && kind != "video" && kind != "audio")
      throw ApiError(400, "invalid_request_error", "invalid_references", "參考 type 只能是 image、video 或 audio");
    auto data = it["data"].get<std::string>();
    auto path = kind == "image" ? save_upload(opt, data, guard) : save_media(opt, data, guard);
    out.push_back({kind, path.string()});
  }
  int count[3] = {0, 0, 0};  // image / video / audio
  for (auto& m : out) count[m.kind == "image" ? 0 : m.kind == "video" ? 1 : 2]++;
  if (count[0] > 9 || count[1] > 3 || count[2] > 3)
    throw ApiError(400, "invalid_request_error", "invalid_references", "參考最多 9 張圖、3 段影片、3 段音訊");
  if (!out.empty() && count[0] + count[1] == 0)
    throw ApiError(400, "invalid_request_error", "invalid_references", "音訊參考必須搭配至少一張圖或一段影片");
}

void read_one_upload(const ServerOptions& opt, const json& j, const char* key, std::string& path, TempGuard& guard) {
  if (!j.contains(key) || j[key].is_null()) return;
  if (!j[key].is_string())
    throw ApiError(400, "invalid_request_error", "invalid_image", strf("%s 必須是字串", key));
  auto s = j[key].get<std::string>();
  if (s.empty()) return;
  path = save_upload(opt, s, guard).string();
}

std::string read_format(const json& j) {
  if (!j.contains("response_format")) return "b64_json";
  if (!j["response_format"].is_string())
    throw ApiError(400, "invalid_request_error", "invalid_format", "response_format 必須是字串");
  auto f = j["response_format"].get<std::string>();
  if (f != "b64_json" && f != "url")
    throw ApiError(400, "invalid_request_error", "invalid_format", "response_format 只能是 b64_json 或 url");
  return f;
}

std::string read_compute(const json& j) {
  if (!j.contains("compute")) return "";
  if (!j["compute"].is_string())
    throw ApiError(400, "invalid_request_error", "invalid_compute", "compute 必須是字串");
  auto c = j["compute"].get<std::string>();
  if (c != "bf16" && c != "fp8" && c != "fp4")
    throw ApiError(400, "invalid_request_error", "invalid_compute", "compute 只能是 bf16、fp8 或 fp4");
  return c;
}

void read_cfg(const json& j, float& cfg) {
  if (!j.contains("cfg_scale")) return;
  if (!j["cfg_scale"].is_number())
    throw ApiError(400, "invalid_request_error", "invalid_cfg", "cfg_scale 必須是數字");
  cfg = j["cfg_scale"].get<float>();
  if (!(cfg >= 0.f && cfg <= 20.f))
    throw ApiError(400, "invalid_request_error", "invalid_cfg", "cfg_scale 必須介於 0 與 20");
}

json job_value(const Job& job) {
  json j = {{"id", job.id},
            {"object", job.type == "video" ? "video.job" : "image.job"},
            {"type", job.type},
            {"status", job.status},
            {"created", job.created},
            {"prompt", job.prompt},
            {"progress", {{"stage", job.stage}, {"step", job.step}, {"total", job.total}}}};
  if (job.has_result)
    j["result"] = {{"url", job.url}, {"seed", job.seed}, {"seconds", job.seconds}, {"timings", job.timings}};
  else
    j["result"] = nullptr;
  if (!job.err_message.empty())
    j["error"] = {{"message", job.err_message}, {"type", job.err_type}, {"code", job.err_code}};
  else
    j["error"] = nullptr;
  const bool video = job.type == "video";
  const int frames = video ? h3_align_frames(job.video.num_frames) : 0;
  const auto& loras = video ? job.video.loras : job.image.loras;
  json ls = json::array();
  for (const auto& l : loras) ls.push_back({{"name", path_utf8(utf8_path(l.path).filename())}, {"scale", l.scale}});
  j["parameters"] = {{"width", video ? job.video.width : job.image.width},
                       {"height", video ? job.video.height : job.image.height},
                       {"steps", video ? job.video.steps : int(job.image.sigmas.empty() ? job.image.steps : job.image.sigmas.size())},
                       {"compute", video ? job.video.compute : job.image.compute}, {"loras", ls}};
  if (video) { j["parameters"]["num_frames"] = frames; j["parameters"]["fps"] = 24; j["parameters"]["duration_seconds"] = frames / 24.0; }
  return j;
}

std::shared_ptr<Job> parse_inner(const ServerOptions& opt, Engine& engine, const std::string& body,
                                 const std::string& forced) {
  json j = json::parse(body);
  if (!j.is_object()) throw ApiError(400, "invalid_request_error", "invalid_json", "本文必須是 JSON 物件");
  auto type = job_type(j, forced);
  auto model_files = check_model(engine, j, type);
  if (j.contains("n")) {
    int n = read_int(j, "n", 1);
    if (n != 1) throw ApiError(400, "invalid_request_error", "invalid_n", "n 必須是 1");
  }
  int steps = read_int(j, "steps", type == "video" ? 30 : 40);
  int min_steps = type == "video" ? 2 : 1;
  if (steps < min_steps || steps > 200)
    throw ApiError(400, "invalid_request_error", "invalid_steps", "steps 必須介於 " + std::to_string(min_steps) + " 與 200");
  float cfg = 1.f;
  read_cfg(j, cfg);
  int64_t seed = -1;
  if (j.contains("seed")) {
    if (!j["seed"].is_number_integer())
      throw ApiError(400, "invalid_request_error", "invalid_type", "seed 必須是整數");
    seed = j["seed"].get<int64_t>();
  }
  int w = 0, h = 0;
  parse_wh(j, w, h);
  check_dims(w, h, type == "video" ? 1920 : 2752);
  auto prompt = require_text(j, "prompt", 8000, false);
  auto negative = require_text(j, "negative_prompt", 8000, true);
  auto loras = read_loras(opt, j);
  auto compute = read_compute(j);
  TempGuard guard;
  auto job = std::make_shared<Job>();
  job->type = type;
  job->prompt = prompt;
  job->response_format = read_format(j);
  if (type == "image") {
    job->image.prompt = std::move(prompt);
    job->image.negative_prompt = std::move(negative);
    job->image.width = w;
    job->image.height = h;
    job->image.steps = steps;
    job->image.cfg_scale = cfg;
    job->image.seed = seed;
    job->image.loras = std::move(loras);
    job->image.compute = compute;
    job->image.model_files = model_files;
    if (j.contains("shift_terminal")) {
      if (!j["shift_terminal"].is_boolean())
        throw ApiError(400, "invalid_request_error", "invalid_type", "shift_terminal 必須是布林值");
      job->image.shift_terminal = j["shift_terminal"].get<bool>();
    }
    read_uploads(opt, j, "images", job->image.ref_images, guard, 10);
    if (w == 0 && job->image.ref_images.empty())
      throw ApiError(400, "invalid_request_error", "invalid_size", "size 為 auto 時需要參考圖 images");
    if (j.contains("sigmas")) {
      if (!j["sigmas"].is_array() || j["sigmas"].empty() || j["sigmas"].size() > 200)
        throw ApiError(400, "invalid_request_error", "invalid_sigmas", "sigmas 必須是 1～200 個數字的陣列");
      for (auto& v : j["sigmas"]) {
        if (!v.is_number() || !(v.get<double>() > 0.0 && v.get<double>() <= 1.0))
          throw ApiError(400, "invalid_request_error", "invalid_sigmas", "sigmas 每個值都必須在 (0, 1]");
        job->image.sigmas.push_back(v.get<float>());
      }
    }
  } else {
    int frames = j.contains("num_frames") ? read_int(j, "num_frames", 56) : read_int(j, "frames", 56);
    if (frames < 5 || frames > 361)
      throw ApiError(400, "invalid_request_error", "invalid_frames", "num_frames 必須介於 5 與 361");
    job->video.prompt = std::move(prompt);
    job->video.width = w;
    job->video.height = h;
    job->video.num_frames = frames;
    job->video.steps = steps;
    job->video.seed = seed;
    job->video.loras = std::move(loras);
    job->video.compute = compute;
    job->video.model_files = model_files;
    read_one_upload(opt, j, "first_frame", job->video.first_frame, guard);
    read_one_upload(opt, j, "last_frame", job->video.last_frame, guard);
    read_references(opt, j, job->video.references, guard);
    if (!job->video.references.empty() && (!job->video.first_frame.empty() || !job->video.last_frame.empty()))
      throw ApiError(400, "invalid_request_error", "invalid_references", "references 不能和 first_frame／last_frame 同時使用");
    if (j.contains("video_shift")) {
      if (!j["video_shift"].is_number() || !(j["video_shift"].get<double>() >= 0.5 && j["video_shift"].get<double>() <= 50))
        throw ApiError(400, "invalid_request_error", "invalid_shift", "video_shift 必須是 0.5～50 的數字");
      job->video.video_shift = j["video_shift"].get<float>();
    }
    if (j.contains("ref_image_short_edge")) {
      int e = read_int(j, "ref_image_short_edge", 2048);
      if (e < 256 || e > 4096)
        throw ApiError(400, "invalid_request_error", "invalid_references", "ref_image_short_edge 必須介於 256 與 4096");
      job->video.ref_image_short_edge = e;
    }
    if (j.contains("ref_native_canvas")) {
      if (!j["ref_native_canvas"].is_boolean())
        throw ApiError(400, "invalid_request_error", "invalid_type", "ref_native_canvas 必須是布林值");
      job->video.ref_native_canvas = j["ref_native_canvas"].get<bool>();
    }
  }
  job->temps = std::move(guard.paths);
  guard.keep = true;
  return job;
}

}  // namespace

std::string error_json(const std::string& message, const std::string& type, const std::string& code) {
  json j;
  j["error"] = {{"message", message}, {"type", type}, {"code", code}};
  return j.dump();
}

std::string job_to_json(const Job& job) { return job_value(job).dump(); }

std::string jobs_to_json(const std::vector<std::shared_ptr<Job>>& jobs) {
  json data = json::array();
  for (auto& job : jobs) data.push_back(job_value(*job));
  return json{{"object", "list"}, {"data", std::move(data)}}.dump();
}

std::string image_sync_json(const Job& job) {
  json item = {{"seed", job.seed}};
  if (job.want_b64) item["b64_json"] = b64_encode(job.png.data(), job.png.size());
  else item["url"] = job.url;
  return json{{"created", job.created}, {"data", json::array({std::move(item)})}}.dump();
}

std::string models_json(const std::vector<ModelInfo>& models) {
  json data = json::array();
  for (auto& m : models) {
    data.push_back({{"id", m.id},
                    {"object", "model"},
                    {"owned_by", "miroxen"},
                    {"kind", m.kind},
                    {"loaded", m.loaded},
                    {"detail", m.detail}, {"files", m.files}, {"defaults", m.defaults},
                    {"active_files", m.active_files}, {"default_compute", m.compute}});
  }
  return json{{"object", "list"}, {"data", std::move(data)}}.dump();
}

std::string loras_json(const ServerOptions& opt) {
  json data = json::array();
  std::error_code ec;
  auto dir = utf8_path(opt.lora_dir);
  if (std::filesystem::is_directory(dir, ec)) {
    std::vector<json> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
      if (!ent.is_regular_file()) continue;
      auto name = path_utf8(ent.path().filename());
      if (!lora_name_ok(name)) continue;
      std::error_code fec;
      auto sz = ent.file_size(fec);
      if (fec) continue;
      json row = {{"name", name}, {"size", sz}, {"model", "unknown"}, {"purpose", "unknown"}, {"preset", nullptr}};
      std::string lower = name;
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      if (lower.find("minimax_h3") != std::string::npos) row["model"] = "minimax-h3";
      if (lower.find("qwen") != std::string::npos || lower == "turbo8_lora_step2500.safetensors" || lower == "zzz_chibi_v2_st2500.safetensors") row["model"] = "qwen-image-2.1";
      const bool viggle = name == "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors";
      if (viggle) row["preset"] = {{"steps", 6}, {"sigmas", {1., .9375, .875, .75, .5, .25}}, {"shift_terminal", false}, {"cfg_scale", 1}};
      if (lower == "turbo8_lora_step2500.safetensors") row["preset"] = {{"steps", 8}, {"shift_terminal", false}, {"cfg_scale", 1}};
      if (lower == "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors") row["preset"] = {{"steps", 9}, {"video_shift", 6}};
      if (lower == "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors") row["preset"] = {{"steps", 5}, {"video_shift", 6}};
      if (lower == "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors") row["preset"] = {{"steps", 9}, {"video_shift", 12}};
      if (!row["preset"].is_null()) row["purpose"] = "acceleration";
      if (lower == "zzz_chibi_v2_st2500.safetensors") row["purpose"] = "style";
      row["task"] = lower.find("_ref2v_") != std::string::npos ? "reference" : lower.find("_fl2v_") != std::string::npos ? "first_last" : "any";
      if (row["model"] == "unknown") {
        try {
          TensorStore store;
          store.add_file(path_utf8(ent.path()));
          for (const auto& key : store.names()) {
            if (key.find("transformer_blocks.") != std::string::npos) { row["model"] = "qwen-image-2.1"; break; }
            if (key.starts_with("blocks.") || key.find("token_refiner.") != std::string::npos) { row["model"] = "minimax-h3"; break; }
          }
        } catch (const std::exception&) { row["warning"] = "無法識別模型，使用前請確認檔案相容性"; }
      }
      rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end(), [](const json& a, const json& b) {
      return a["name"].get<std::string>() < b["name"].get<std::string>();
    });
    for (auto& row : rows) data.push_back(std::move(row));
  }
  return json{{"data", std::move(data)}}.dump();
}

std::shared_ptr<Job> parse_request(const ServerOptions& opt, Engine& engine, const std::string& body,
                                   const std::string& forced_type) {
  try {
    return parse_inner(opt, engine, body, forced_type);
  } catch (const ApiError&) {
    throw;
  } catch (const json::exception&) {
    throw ApiError(400, "invalid_request_error", "invalid_json", "本文不是合法的 JSON");
  }
}

}  // namespace mx
