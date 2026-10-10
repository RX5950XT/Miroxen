// 社群 LoRA 安裝：面板上傳或貼 Hugging Face／Civitai 網址，驗證後依張量名稱放進 models/loras/image 或 video。
#include "server/server_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "core/common.h"
#include "core/model_file.h"
#include "nlohmann/json.hpp"

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace mx {
namespace fs = std::filesystem;
using json = nlohmann::json;

std::string lora_model_kind(const fs::path& file) {
  try {
    TensorStore st;
    st.add_file(path_utf8(file));
    const auto names = st.names();
    auto any = [&](auto pred) { return std::any_of(names.begin(), names.end(), pred); };
    if (!any([](const std::string& k) { return k.find("lora") != std::string::npos || k.ends_with(".alpha"); }))
      return "";
    // Qwen：diffusers／Comfy 的 transformer_blocks.N 與 kohya 的 lora_unet_transformer_blocks_N 都含這個字串。
    if (any([](const std::string& k) { return k.find("transformer_blocks") != std::string::npos; })) return "qwen-image-2.1";
    if (any([](const std::string& k) {
          return k.starts_with("blocks.") || k.find(".blocks.") != std::string::npos ||
                 k.find("token_refiner") != std::string::npos || k.find("lora_unet_blocks_") != std::string::npos;
        }))
      return "minimax-h3";
  } catch (const std::exception& e) {
    MX_INFO("LoRA 檔案無法讀取：%s", e.what());
  }
  return "";
}

std::string install_lora_file(const ServerOptions& opt, const fs::path& part, const std::string& name) {
  auto fail = [&](const char* code, const std::string& msg) {
    std::error_code ec;
    fs::remove(part, ec);
    throw ApiError(400, "invalid_request_error", code, msg);
  };
  if (!lora_name_ok(name)) fail("invalid_lora", "LoRA 檔名必須以 .safetensors 結尾，且不能含路徑");
  const auto kind = lora_model_kind(part);
  if (kind.empty()) fail("unsupported_lora", "不是 Qwen-Image-2.1 或 MiniMax-H3 的 LoRA（檔案內找不到對應的層）");
  if (lora_files(opt).count(name)) fail("lora_exists", "已經有同名的 LoRA");
  const auto dir = utf8_path(opt.lora_dir) / (kind == "minimax-h3" ? "video" : "image");
  fs::create_directories(dir);
  const auto size = fs::file_size(part);
  fs::rename(part, dir / utf8_path(name));
  MX_INFO("已安裝 LoRA %s → %s", name.c_str(), path_utf8(dir).c_str());
  return json{{"name", name}, {"model", kind}, {"size", size}}.dump();
}

std::string normalize_lora_url(const std::string& raw) {
  auto bad = [](const std::string& msg) { return ApiError(400, "invalid_request_error", "invalid_url", msg); };
  std::string url = raw;
  while (!url.empty() && std::isspace(static_cast<unsigned char>(url.back()))) url.pop_back();
  while (!url.empty() && std::isspace(static_cast<unsigned char>(url.front()))) url.erase(url.begin());
  if (url.size() > 2048 || !url.starts_with("https://")) throw bad("請貼 https:// 開頭的 Hugging Face 或 Civitai 網址");
  for (unsigned char c : url)
    if (c <= 0x20 || c >= 0x7f || std::string("\"<>\\^`{|}").find(char(c)) != std::string::npos)
      throw bad("網址含有不允許的字元");
  const size_t host_end = url.find('/', 8);
  std::string host = url.substr(8, host_end == std::string::npos ? std::string::npos : host_end - 8);
  std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  const std::string path = host_end == std::string::npos ? "/" : url.substr(host_end);
  const std::string bare = path.substr(0, path.find_first_of("?#"));
  auto digits = [](const std::string& s) { return !s.empty() && s.size() < 20 && std::all_of(s.begin(), s.end(), ::isdigit); };
  if (host == "huggingface.co" || host == "hf.co") {
    std::string p = bare;
    const auto blob = p.find("/blob/");
    if (blob != std::string::npos) p.replace(blob, 6, "/resolve/");
    if (p.find("/resolve/") == std::string::npos || !p.ends_with(".safetensors"))
      throw bad("Hugging Face 請貼單一 .safetensors 檔案的網址（檔案頁面或下載連結）");
    return "https://huggingface.co" + p;
  }
  if (host == "civitai.com" || host == "www.civitai.com") {
    const std::string dl = "/api/download/models/";
    if (bare.starts_with(dl) && digits(bare.substr(dl.size()))) return "https://civitai.com" + bare;
    const auto v = path.find("modelVersionId=");
    if (v != std::string::npos) {
      std::string id = path.substr(v + 15, path.find_first_of("&#", v) == std::string::npos ? std::string::npos
                                                                                            : path.find_first_of("&#", v) - v - 15);
      if (digits(id)) return "https://civitai.com" + dl + id;
    }
    if (bare.starts_with("/models/")) {
      std::string id = bare.substr(8, bare.find('/', 8) == std::string::npos ? std::string::npos : bare.find('/', 8) - 8);
      if (digits(id)) return "https://civitai.com/api/v1/models/" + id;  // 背景下載時再查最新版本
    }
    throw bad("Civitai 請貼模型頁面或下載連結");
  }
  throw bad("只支援 huggingface.co 與 civitai.com 的網址");
}

namespace {

struct Proc {
#ifdef _WIN32
  HANDLE h = nullptr;
#else
  pid_t pid = -1;
#endif
};

// 不經過 shell 直接啟動 curl（網址裡的 & 不會被當成指令）。
Proc spawn_curl(const std::vector<std::string>& args) {
  Proc p;
#ifdef _WIN32
  std::wstring cmd = L"curl.exe";
  for (const auto& a : args) {
    auto w = utf8_path(a).wstring();
    cmd += L" \"" + w + L"\"";  // 參數已驗證不含引號
  }
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
    throw std::runtime_error("無法啟動 curl（Windows 10 以上內建，請確認 curl.exe 在 PATH）");
  CloseHandle(pi.hThread);
  p.h = pi.hProcess;
#else
  std::vector<char*> argv{const_cast<char*>("curl")};
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  if (posix_spawnp(&p.pid, "curl", nullptr, nullptr, argv.data(), environ) != 0)
    throw std::runtime_error("無法啟動 curl");
#endif
  return p;
}

// 回傳 -1 表示還在跑；kill 為真時結束行程。
int poll_proc(Proc& p, bool kill) {
#ifdef _WIN32
  if (kill) TerminateProcess(p.h, 1);
  if (WaitForSingleObject(p.h, kill ? INFINITE : 500) == WAIT_TIMEOUT) return -1;
  DWORD code = 1;
  GetExitCodeProcess(p.h, &code);
  CloseHandle(p.h);
  return int(code);
#else
  if (kill) ::kill(p.pid, SIGTERM);
  int st = 0;
  for (int i = 0; i < 5 || kill; i++) {
    if (waitpid(p.pid, &st, WNOHANG) == p.pid) return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
    usleep(100000);
  }
  return -1;
#endif
}

uintmax_t dir_bytes(const fs::path& dir) {
  uintmax_t n = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec))
    if (e.is_regular_file(ec)) n += e.file_size(ec);
  return n;
}

}  // namespace

struct LoraInstaller::Item {
  uint64_t id = 0;
  std::string url, state = "queued", name, model, error;
  uintmax_t bytes = 0;
  std::atomic<bool> stop{false};
};

LoraInstaller::LoraInstaller(ServerOptions opt) : opt_(std::move(opt)) {}

LoraInstaller::~LoraInstaller() {
  {
    std::lock_guard lk(mu_);
    for (auto& it : items_) it->stop = true;
  }
  for (auto& t : threads_) t.join();
}

static json item_json(const std::string& url, uint64_t id, const std::string& state, uintmax_t bytes,
                      const std::string& name, const std::string& model, const std::string& error) {
  json j = {{"id", id}, {"url", url}, {"state", state}, {"bytes", bytes}};
  if (!name.empty()) j["name"] = name;
  if (!model.empty()) j["model"] = model;
  if (!error.empty()) j["error"] = error;
  return j;
}

std::string LoraInstaller::start(const std::string& url) {
  auto normalized = normalize_lora_url(url);
  std::lock_guard lk(mu_);
  const auto active = std::count_if(items_.begin(), items_.end(), [](auto& it) {
    return it->state == "queued" || it->state == "downloading" || it->state == "installing";
  });
  if (active >= 2) throw ApiError(429, "invalid_request_error", "too_many_installs", "同時最多下載兩個 LoRA，請稍候");
  auto item = std::make_shared<Item>();
  item->id = next_++;
  item->url = normalized;
  items_.push_back(item);
  threads_.emplace_back([this, item] { run(item); });
  return item_json(item->url, item->id, item->state, 0, "", "", "").dump();
}

std::string LoraInstaller::list_json() {
  std::lock_guard lk(mu_);
  json data = json::array();
  for (auto it = items_.rbegin(); it != items_.rend(); ++it) {
    auto& i = **it;
    data.push_back(item_json(i.url, i.id, i.state, i.bytes, i.name, i.model, i.error));
  }
  return json{{"data", std::move(data)}}.dump();
}

void LoraInstaller::run(std::shared_ptr<Item> item) {
  auto set = [&](const char* state, const std::string& error = "") {
    std::lock_guard lk(mu_);
    item->state = state;
    item->error = error;
  };
  const auto dir = utf8_path(opt_.lora_dir) / ".incoming" / ("download-" + std::to_string(item->id) + "-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::error_code ec;
  try {
    fs::create_directories(dir);
    // 回傳 curl 結束碼；取消時回 -1。
    auto curl = [&](std::vector<std::string> args, const std::string& url) {
      std::vector<std::string> base = {"-L", "--fail", "--silent", "--show-error", "--proto", "=https",
                                       "--proto-redir", "=https", "--retry", "2", "--max-filesize", "8589934592"};
      const char* token = url.find("civitai.com") != std::string::npos ? std::getenv("CIVITAI_TOKEN")
                                                                         : std::getenv("HF_TOKEN");
      if (token && *token && std::string(token).find('"') == std::string::npos)
        base.insert(base.end(), {"-H", std::string("Authorization: Bearer ") + token});
      base.insert(base.end(), args.begin(), args.end());
      base.push_back(url);
      Proc p = spawn_curl(base);
      for (;;) {
        int rc = poll_proc(p, item->stop);
        {
          std::lock_guard lk(mu_);
          item->bytes = dir_bytes(dir);
        }
        if (item->stop) return -1;
        if (rc >= 0) return rc;
      }
    };
    set("downloading");
    std::string url = item->url;
    if (url.find("/api/v1/models/") != std::string::npos) {  // Civitai 模型頁：取最新版本的下載連結
      const auto meta = dir / "model.json";
      if (curl({"-o", path_utf8(meta)}, url) != 0) throw std::runtime_error("查不到 Civitai 模型資訊");
      std::ifstream mf(meta, std::ios::binary);
      json j = json::parse(std::string(std::istreambuf_iterator<char>(mf), {}), nullptr, false);
      mf.close();
      fs::remove(meta, ec);
      if (!j.is_object() || !j.contains("modelVersions") || !j["modelVersions"].is_array() || j["modelVersions"].empty() ||
          !j["modelVersions"][0].contains("id") || !j["modelVersions"][0]["id"].is_number_integer())
        throw std::runtime_error("Civitai 模型沒有可下載的版本");
      url = "https://civitai.com/api/download/models/" + std::to_string(j["modelVersions"][0]["id"].get<int64_t>());
    }
    const int rc = curl({"-J", "-O", "--output-dir", path_utf8(dir)}, url);
    if (rc < 0) throw std::runtime_error("已取消");
    if (rc == 22) throw std::runtime_error("下載被拒（HTTP 錯誤）。Civitai 部分模型需登入：設定環境變數 CIVITAI_TOKEN；Hugging Face 私有模型設定 HF_TOKEN");
    if (rc == 63) throw std::runtime_error("檔案超過 8 GB 上限");
    if (rc != 0) throw std::runtime_error("下載失敗（curl 錯誤碼 " + std::to_string(rc) + "）");
    fs::path file;
    for (const auto& e : fs::directory_iterator(dir))
      if (e.is_regular_file()) file = e.path();
    if (file.empty()) throw std::runtime_error("下載完成但找不到檔案");
    set("installing");
    const std::string name = path_utf8(file.filename());
    json row = json::parse(install_lora_file(opt_, file, name));
    {
      std::lock_guard lk(mu_);
      item->name = name;
      item->model = row.value("model", "");
    }
    set("done");
  } catch (const std::exception& e) {
    MX_INFO("LoRA 下載失敗：%s", e.what());
    set("failed", e.what());
  }
  fs::remove_all(dir, ec);
}

}  // namespace mx
