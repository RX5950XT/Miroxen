#include "httplib.h"

#include "stb_image.h"

#include "nlohmann/json.hpp"

#include "server/fake_engine.h"
#include "server/image_io.h"
#include "server/server.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

int g_ok = 0;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

void check(bool cond, const std::string& msg) {
  if (!cond) fail(msg);
  ++g_ok;
}

std::string u8s(const fs::path& p) {
  auto u = p.u8string();
  return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

void touch(const fs::path& p, const std::string& bytes) {
  fs::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  if (!out) fail("無法建立 " + u8s(p));
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) fail("寫入失敗 " + u8s(p));
}

json parse_body(const httplib::Result& res) {
  if (!res) fail(std::string("連線失敗: ") + httplib::to_string(res.error()));
  try {
    return json::parse(res->body);
  } catch (const std::exception&) {
    fail("不是 JSON，HTTP " + std::to_string(res->status) + " " + res->body.substr(0, 240));
  }
}

void expect_code(const httplib::Result& res, int status, const char* code) {
  if (!res) fail(std::string("連線失敗: ") + httplib::to_string(res.error()));
  if (res->status != status) {
    fail("預期 HTTP " + std::to_string(status) + " 實際 " + std::to_string(res->status) + " " +
         res->body.substr(0, 300));
  }
  auto j = parse_body(res);
  check(j.contains("error") && j["error"].is_object(), "缺少 error 物件: " + res->body.substr(0, 200));
  check(j["error"].value("code", "") == code, "錯誤碼應為 " + std::string(code) + " 本文 " + res->body.substr(0, 240));
  check(j["error"].contains("message") && j["error"]["message"].is_string(), "缺少 message");
  check(j["error"].contains("type") && j["error"]["type"].is_string(), "缺少 type");
}

httplib::Result post_json(httplib::Client& cli, const std::string& path, const json& body) {
  return cli.Post(path, body.dump(), "application/json");
}

json image_body() {
  return {{"model", "qwen-image-2.1"}, {"prompt", "一盞燈"}, {"size", "256x256"}, {"steps", 1}, {"n", 1}};
}

json wait_until(httplib::Client& cli, const std::string& id, int ms,
                const std::function<bool(const json&)>& pred) {
  auto start = std::chrono::steady_clock::now();
  json last;
  while (true) {
    auto res = cli.Get("/v1/jobs/" + id);
    if (!res || res->status != 200) {
      fail("查詢工作失敗 " + (res ? std::to_string(res->status) + " " + res->body.substr(0, 200)
                                  : httplib::to_string(res.error())));
    }
    last = json::parse(res->body);
    if (pred(last)) return last;
    if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(ms)) fail("逾時 " + last.dump());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

bool terminal(const json& job) {
  auto s = job.value("status", "");
  return s == "succeeded" || s == "failed" || s == "cancelled";
}

void expect_png(const std::vector<uint8_t>& png, int w, int h) {
  check(png.size() >= 8 && png[0] == 0x89 && png[1] == 'P', "不是 PNG");
  int iw = 0, ih = 0, ic = 0;
  unsigned char* px = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &iw, &ih, &ic, 0);
  struct Free {
    unsigned char* p;
    ~Free() {
      if (p) stbi_image_free(p);
    }
  } guard{px};
  check(px != nullptr, "stb 解不開 PNG");
  check(iw == w && ih == h, "PNG 尺寸應為 " + std::to_string(w) + "x" + std::to_string(h) + " 實際 " +
                                 std::to_string(iw) + "x" + std::to_string(ih));
}

struct TempTree {
  fs::path root;
  explicit TempTree() {
    auto n = std::chrono::steady_clock::now().time_since_epoch().count();
    root = fs::temp_directory_path() / ("mxsrv-" + std::to_string(n));
    fs::create_directories(root / "lora" / "nested");
    fs::create_directories(root / "out");
    touch(root / "lora" / "good.safetensors", "not-a-real-header");
    touch(root / "lora" / "image" / "image.safetensors", "not-a-real-header");
    touch(root / "lora" / "video" / "video.safetensors", "not-a-real-header");
    touch(root / "lora" / "skip.txt", "nope");
    touch(root / "lora" / "nested" / "hidden.safetensors", "nope");
  }
  ~TempTree() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  mx::ServerOptions opt(const std::string& out_name) const {
    mx::ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.web_dir = "web";
    o.out_dir = u8s(root / out_name);
    o.lora_dir = u8s(root / "lora");
    return o;
  }
};

struct Running {
  mx::FakeEngine engine;
  mx::Server server;
  std::unique_ptr<httplib::Client> cli;
  explicit Running(mx::ServerOptions opt) : server(engine, std::move(opt)) {
    server.start();
    cli = std::make_unique<httplib::Client>("127.0.0.1", server.port());
    cli->set_connection_timeout(5, 0);
    cli->set_read_timeout(60, 0);
    cli->set_write_timeout(60, 0);
  }
};

void test_meta_and_static(httplib::Client& cli) {
  auto health = cli.Get("/health");
  check(health && health->status == 200, "health 狀態");
  auto hj = parse_body(health);
  check(hj.value("status", "") == "ok", "health 本文");

  auto models = cli.Get("/v1/models");
  check(models && models->status == 200, "models 狀態");
  auto mj = parse_body(models);
  check(mj.value("object", "") == "list", "models object");
  check(mj["data"].size() == 2, "應有兩個模型");
  bool saw_image = false, saw_video = false;
  for (auto& m : mj["data"]) {
    check(m.value("object", "") == "model", "model object");
    check(m.value("owned_by", "") == "miroxen", "owned_by");
    check(m.value("loaded", false), "假引擎模型應已載入");
    if (m.value("id", "") == "qwen-image-2.1" && m.value("kind", "") == "image") saw_image = true;
    if (m.value("id", "") == "minimax-h3" && m.value("kind", "") == "video") saw_video = true;
  }
  check(saw_image && saw_video, "模型 id 或 kind 不對");

  auto sys = parse_body(cli.Get("/v1/system"));
  check(sys.contains("gpus") && sys["gpus"].is_array() && !sys["gpus"].empty(), "gpus");
  check(sys["gpus"][0].contains("name") && sys["gpus"][0].contains("total") && sys["gpus"][0].contains("free"),
        "gpu 欄位");
  check(sys.contains("queue_length") && sys["queue_length"].is_number_integer(), "queue_length");
  check(sys.contains("host") && sys.contains("process") && sys.contains("api"), "監控缺少主機、程序或 API 資料");
  check(sys["api"]["requests"].is_number_unsigned() && sys["api"]["errors"].is_number_unsigned(), "API 計數型別");
  for (auto& g : sys["gpus"]) check(g.contains("utilization_percent") && g.contains("temperature_c") && g.contains("power_w"), "GPU 監控欄位");

  auto loras = parse_body(cli.Get("/v1/loras"));
  check(loras["data"].size() == 3, "應列出平面目錄、image 與 video 的 LoRA，略過其他子目錄");
  check(loras["data"][0].value("name", "") == "good.safetensors", "LoRA 名稱");
  check(loras["data"][0].value("size", 0) > 0, "LoRA 大小");
  check(loras["data"][0].value("model", "") == "unknown" && loras["data"][0].contains("warning"), "未知 LoRA 不能假裝相容");

  auto index = cli.Get("/");
  check(index && index->status == 200, "首頁狀態");
  check(index->body.find("Miroxen") != std::string::npos, "首頁沒有名稱");
  check(index->body.find("生成") != std::string::npos, "首頁沒有面板文案");
  auto js = cli.Get("/web/app.js");
  check(js && js->status == 200 && js->body.find("EventSource") != std::string::npos, "app.js");
  auto css = cli.Get("/web/app.css");
  check(css && css->status == 200 && css->body.find("--accent") != std::string::npos, "app.css");
  auto missing = cli.Get("/web/no-such-file.js");
  expect_code(missing, 404, "not_found");
  auto slip = cli.Get("/files/..");
  check(slip && slip->status != 200, "/files/.. 不該回 200");
}

void test_model_control(httplib::Client& cli) {
  auto network = parse_body(cli.Get("/v1/network"));
  check(network["lan_enabled"] == false && network["urls"].empty(), "本機綁定不應公布 LAN 網址");
  expect_code(cli.Post("/v1/models/load", "{", "application/json"), 400, "invalid_json");
  expect_code(post_json(cli, "/v1/models/load", {{"model", "missing"}}), 400, "model_not_found");
  expect_code(post_json(cli, "/v1/models/load", {{"model", "qwen-image-2.1"}, {"model_files", {{"dit", "../bad"}}}}), 400, "invalid_model_files");
  expect_code(post_json(cli, "/v1/models/load", {{"model", "qwen-image-2.1"}, {"compute", "bad"}}), 400, "invalid_compute");
  expect_code(cli.Post("/v1/models/unload", "{}", "text/plain"), 415, "unsupported_media_type");
  auto unloaded = parse_body(post_json(cli, "/v1/models/unload", json::object()));
  check(!unloaded["data"][0]["loaded"].get<bool>() && !unloaded["data"][1]["loaded"].get<bool>(), "卸載所有模型");
  auto loaded = parse_body(post_json(cli, "/v1/models/load", {{"model", "qwen-image-2.1"}}));
  check(loaded["data"][0]["loaded"].get<bool>() && !loaded["data"][1]["loaded"].get<bool>(), "手動載入圖片模型");
  loaded = parse_body(post_json(cli, "/v1/models/load", {{"model", "minimax-h3"}}));
  check(!loaded["data"][0]["loaded"].get<bool>() && loaded["data"][1]["loaded"].get<bool>(), "切換影音卸載圖片");
  auto body = image_body(); body["type"] = "image"; body["steps"] = 200;
  auto job = parse_body(post_json(cli, "/v1/jobs", body));
  expect_code(post_json(cli, "/v1/models/unload", json::object()), 409, "model_busy");
  expect_code(post_json(cli, "/v1/models/load", {{"model", "minimax-h3"}}), 409, "model_busy");
  cli.Delete("/v1/jobs/" + job["id"].get<std::string>());
  wait_until(cli, job["id"], 5000, terminal);
  check(post_json(cli, "/v1/models/unload", json::object())->status == 200, "取消完成後可以卸載");
}

void test_validation(httplib::Client& cli) {
  auto files = image_body();
  files["model_files"] = {{"dit", "../test-Q8_0.gguf"}};
  expect_code(post_json(cli, "/v1/images/generations", files), 400, "invalid_model_files");
  files["model_files"] = {{"unsupported", "test-Q8_0.gguf"}};
  expect_code(post_json(cli, "/v1/images/generations", files), 400, "invalid_model_files");
  files["model_files"] = {{"dit", "test-Q4_K.gguf"}};
  auto accepted = post_json(cli, "/v1/images/generations", files);
  check(accepted && accepted->status == 200, "已安裝的模型檔應能送出");
  expect_code(cli.Post("/v1/images/generations", "{}", "application/json"), 400, "missing_model");
  expect_code(cli.Post("/v1/jobs", "{", "application/json"), 400, "invalid_json");
  auto bad_size = image_body();
  bad_size["size"] = "100x100";
  expect_code(post_json(cli, "/v1/images/generations", bad_size), 400, "invalid_size");
  bad_size["size"] = "257x256";
  expect_code(post_json(cli, "/v1/images/generations", bad_size), 400, "invalid_size");
  auto empty = image_body();
  empty["prompt"] = "   ";
  expect_code(post_json(cli, "/v1/images/generations", empty), 400, "empty_prompt");
  auto long_prompt = image_body();
  long_prompt["prompt"] = std::string(8001, 'a');
  expect_code(post_json(cli, "/v1/images/generations", long_prompt), 400, "prompt_too_long");
  auto steps = image_body();
  steps["steps"] = 0;
  expect_code(post_json(cli, "/v1/images/generations", steps), 400, "invalid_steps");
  steps["steps"] = 201;
  expect_code(post_json(cli, "/v1/images/generations", steps), 400, "invalid_steps");
  auto compute = image_body();
  compute["compute"] = "int8";
  expect_code(post_json(cli, "/v1/images/generations", compute), 400, "invalid_compute");
  compute["compute"] = 4;
  expect_code(post_json(cli, "/v1/images/generations", compute), 400, "invalid_compute");
  auto st = image_body();
  st["shift_terminal"] = "no";
  expect_code(post_json(cli, "/v1/images/generations", st), 400, "invalid_type");
  auto sig = image_body();
  sig["sigmas"] = json::array({1.0, 1.5});
  expect_code(post_json(cli, "/v1/images/generations", sig), 400, "invalid_sigmas");
  auto autosize = image_body();
  autosize["size"] = "auto";  // 沒有參考圖時不能 auto
  expect_code(post_json(cli, "/v1/images/generations", autosize), 400, "invalid_size");
  bad_size["size"] = "272x256";  // 16 的倍數但不是 32 的倍數
  expect_code(post_json(cli, "/v1/images/generations", bad_size), 400, "invalid_size");
  auto refs = json{{"type", "video"}, {"model", "minimax-h3"}, {"prompt", "海"}, {"size", "864x480"},
                   {"references", json::array({json{{"type", "gif"}, {"data", "AAAA"}}})}};
  expect_code(post_json(cli, "/v1/videos/generations", refs), 400, "invalid_references");
  refs["references"] = "x";
  expect_code(post_json(cli, "/v1/videos/generations", refs), 400, "invalid_references");
  refs.erase("references");
  refs["video_shift"] = 0;
  expect_code(post_json(cli, "/v1/videos/generations", refs), 400, "invalid_shift");
  auto n = image_body();
  n["n"] = 2;
  expect_code(post_json(cli, "/v1/images/generations", n), 400, "invalid_n");
  auto cfg = image_body();
  cfg["cfg_scale"] = 21;
  expect_code(post_json(cli, "/v1/images/generations", cfg), 400, "invalid_cfg");
  auto kind = image_body();
  kind["model"] = "minimax-h3";
  expect_code(post_json(cli, "/v1/images/generations", kind), 400, "model_kind");
  auto frames = json{{"type", "video"}, {"model", "minimax-h3"}, {"prompt", "海"},
                      {"size", "864x480"}, {"steps", 1}, {"num_frames", 4}};
  expect_code(post_json(cli, "/v1/jobs", frames), 400, "invalid_steps");
  frames["steps"] = 2;
  expect_code(post_json(cli, "/v1/jobs", frames), 400, "invalid_frames");
  frames["num_frames"] = 56;
  frames["size"] = "2048x1152";
  expect_code(post_json(cli, "/v1/videos/generations", frames), 400, "invalid_size");
  frames.erase("num_frames");
  frames["frames"] = 2;
  frames["size"] = "256x256";
  expect_code(post_json(cli, "/v1/jobs", frames), 400, "invalid_frames");

  auto bad_lora = image_body();
  bad_lora["loras"] = json::array({{{"name", "../good.safetensors"}, {"scale", 1}}});
  expect_code(post_json(cli, "/v1/images/generations", bad_lora), 400, "invalid_lora");
  bad_lora["loras"] = json::array({{{"name", "nested/hidden.safetensors"}}});
  expect_code(post_json(cli, "/v1/images/generations", bad_lora), 400, "invalid_lora");
  bad_lora["loras"] = json::array({{{"name", "C:good.safetensors"}}});
  expect_code(post_json(cli, "/v1/images/generations", bad_lora), 400, "invalid_lora");
  bad_lora["loras"] = json::array({{{"name", "missing.safetensors"}}});
  expect_code(post_json(cli, "/v1/images/generations", bad_lora), 400, "lora_not_found");
  for (const char* name : {"image.safetensors", "video.safetensors"}) {
    bad_lora["loras"] = json::array({{{"name", name}}});
    auto accepted = post_json(cli, "/v1/images/generations", bad_lora);
    check(accepted && accepted->status == 200, "分類 LoRA 可用檔名呼叫");
  }
  expect_code(post_json(cli, "/v1/images/generations", {{"model", "qwen-image-2.1"},
                                                        {"prompt", "x"},
                                                        {"size", "256x256"},
                                                        {"images", json::array({"!!!!"})}}),
              400, "invalid_image");
  expect_code(cli.Get("/v1/jobs/job_deadbeef"), 404, "job_not_found");
  expect_code(cli.Delete("/v1/jobs/job_deadbeef"), 404, "job_not_found");
}

void test_image_and_jobs(httplib::Client& cli) {
  std::vector<uint8_t> rgb(16 * 16 * 3, 40);
  auto png = mx::encode_png(rgb.data(), 16, 16, 3);
  auto b64 = mx::b64_encode(png.data(), png.size());
  auto gen = image_body();
  gen["steps"] = 2;
  gen["seed"] = 42;
  gen["cfg_scale"] = 1.5;
  gen["negative_prompt"] = "模糊";
  gen["images"] = json::array({"data:image/png;base64," + b64});
  gen["loras"] = json::array({{{"name", "good.safetensors"}, {"scale", 0.5}}});
  auto res = post_json(cli, "/v1/images/generations", gen);
  check(res && res->status == 200, "圖像生成狀態 " + (res ? std::to_string(res->status) + " " + res->body.substr(0, 240) : ""));
  auto body = parse_body(res);
  check(body.contains("created") && body["data"].size() == 1, "圖像回應形狀");
  check(body["data"][0].value("seed", -1) == 42, "種子應為 42");
  check(body["data"][0].contains("b64_json"), "預設應回 b64_json");
  auto decoded = mx::b64_decode(body["data"][0]["b64_json"].get<std::string>());
  expect_png(decoded, 256, 256);

  auto url_body = image_body();
  url_body["response_format"] = "url";
  url_body["seed"] = 7;
  url_body.erase("n");
  url_body.erase("size");
  url_body["width"] = 256;
  url_body["height"] = 256;
  auto url_res = post_json(cli, "/v1/images/generations", url_body);
  check(url_res && url_res->status == 200, "url 生成失敗");
  auto uj = parse_body(url_res);
  auto url = uj["data"][0].value("url", "");
  check(url.rfind("/files/", 0) == 0 && url.size() > 10 && url.find(".png") != std::string::npos, "url " + url);
  check(!uj["data"][0].contains("b64_json"), "url 模式不該帶 b64");
  check(uj["data"][0].value("seed", -1) == 7, "url 種子");
  auto file = cli.Get(url);
  check(file && file->status == 200, "下載生成檔");
  expect_png(std::vector<uint8_t>(file->body.begin(), file->body.end()), 256, 256);

  auto blocker_req = image_body();
  blocker_req["type"] = "image";
  blocker_req["steps"] = 80;
  blocker_req["prompt"] = "擋住佇列";
  auto blocker = parse_body(post_json(cli, "/v1/jobs", blocker_req));
  check(blocker.value("status", "") == "queued", "建立時應為 queued");
  check(blocker.value("object", "") == "image.job", "image.job");
  auto blocker_id = blocker.value("id", "");
  check(blocker_id.rfind("job_", 0) == 0, "工作 id");
  wait_until(cli, blocker_id, 5000, [](const json& j) { return j.value("status", "") == "running"; });
  auto sys = parse_body(cli.Get("/v1/system"));
  check(sys.value("queue_length", 0) >= 1, "執行中佇列長度");

  auto victim_req = blocker_req;
  victim_req["steps"] = 4;
  victim_req["prompt"] = "會被取消";
  auto victim = parse_body(post_json(cli, "/v1/jobs", victim_req));
  auto victim_id = victim.value("id", "");
  auto queued = parse_body(cli.Get("/v1/jobs/" + victim_id));
  check(queued.value("status", "") == "queued", "第二件應還在排隊，實際 " + queued.value("status", ""));
  auto cancelled = parse_body(cli.Delete("/v1/jobs/" + victim_id));
  check(cancelled.value("status", "") == "cancelled", "排隊中的工作應立刻取消");
  check(cancelled["error"].value("code", "") == "cancelled", "取消錯誤碼");

  auto listed = parse_body(cli.Get("/v1/jobs"));
  check(listed.value("object", "") == "list", "工作清單");
  bool saw_b = false, saw_v = false;
  for (auto& row : listed["data"]) {
    if (row.value("id", "") == blocker_id) saw_b = true;
    if (row.value("id", "") == victim_id) saw_v = true;
  }
  check(saw_b && saw_v, "清單應含兩件工作");
  check(listed["data"].size() >= 2 && listed["data"][0].value("id", "") == victim_id, "清單應新的在前");

  cli.Delete("/v1/jobs/" + blocker_id);
  auto blocker_done = wait_until(cli, blocker_id, 8000, terminal);
  check(blocker_done.value("status", "") == "cancelled", "執行中的工作取消後應為 cancelled，實際 " +
                                                             blocker_done.value("status", ""));

  auto video_req = json{{"model", "minimax-h3"}, {"prompt", "海浪"}, {"size", "256x256"},
                         {"steps", 40}, {"num_frames", 5}};
  auto video = parse_body(post_json(cli, "/v1/videos/generations", video_req));
  check(video.value("object", "") == "video.job", "video.job");
  check(video.value("status", "") == "queued", "影片建立時應為 queued");
  auto video_id = video.value("id", "");
  wait_until(cli, video_id, 5000, [](const json& j) { return j.value("status", "") == "running"; });
  cli.Delete("/v1/jobs/" + video_id);
  auto video_done = wait_until(cli, video_id, 8000, terminal);
  check(video_done.value("status", "") == "cancelled", "影片工作應取消而非依賴 ffmpeg");

  auto sse_req = image_body();
  sse_req["type"] = "image";
  sse_req["steps"] = 12;
  sse_req["prompt"] = "看進度";
  auto sse_job = parse_body(post_json(cli, "/v1/jobs", sse_req));
  auto sse = cli.Get("/v1/jobs/" + sse_job.value("id", "") + "/events");
  check(sse && sse->status == 200, "SSE 狀態");
  check(sse->body.find("data: ") != std::string::npos, "SSE 應有 data 事件");
  check(sse->body.find("denoise") != std::string::npos, "SSE 應有 denoise");
  check(sse->body.find("succeeded") != std::string::npos, "SSE 應有 succeeded");
  auto done = parse_body(cli.Get("/v1/jobs/" + sse_job.value("id", "")));
  check(done.value("status", "") == "succeeded", "SSE 結束後工作應成功");
  check(done["result"].value("url", "").find(".png") != std::string::npos, "結果網址");
  check(done["result"].contains("seed") && done["result"].contains("seconds"), "結果欄位");
  check(done["progress"].contains("stage") && done["progress"].contains("step") && done["progress"].contains("total"),
        "進度欄位");
}

void test_auth_and_limit(const TempTree& tree) {
  auto locked_opt = tree.opt("out-auth");
  locked_opt.api_key = "secret";
  Running locked(locked_opt);
  auto& cli = *locked.cli;
  auto health = cli.Get("/health");
  check(health && health->status == 200, "有金鑰時 health 仍公開");
  auto index = cli.Get("/");
  check(index && index->status == 200 && index->body.find("Miroxen") != std::string::npos, "靜態檔不需金鑰");
  expect_code(cli.Get("/v1/models"), 401, "unauthorized");
  expect_code(cli.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer wrong"}}), 401, "unauthorized");
  expect_code(post_json(cli, "/v1/models/unload", json::object()), 401, "unauthorized");
  expect_code(post_json(cli, "/v1/models/load", {{"model", "qwen-image-2.1"}}), 401, "unauthorized");
  auto ok = cli.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer secret"}});
  check(ok && ok->status == 200, "正確金鑰應通過");
  expect_code(post_json(cli, "/v1/jobs", image_body()), 401, "unauthorized");

  auto open_opt = tree.opt("out-open");
  Running open(open_opt);
  auto models = open.cli->Get("/v1/models");
  check(models && models->status == 200, "沒有金鑰時 /v1 應開放");

  auto big_opt = tree.opt("out-big");
  big_opt.max_body_mb = 1;
  Running big(big_opt);
  expect_code(big.cli->Post("/v1/jobs", "{", "application/json"), 400, "invalid_json");
  std::string payload(1200 * 1024, 'x');
  expect_code(big.cli->Post("/v1/jobs", payload, "application/json"), 413, "payload_too_large");
}

void run_all() {
  TempTree tree;
  Running main_server(tree.opt("out"));
  test_meta_and_static(*main_server.cli);
  test_model_control(*main_server.cli);
  test_validation(*main_server.cli);
  touch(tree.root / "lora" / "image" / "duplicate.safetensors", "nope");
  touch(tree.root / "lora" / "video" / "duplicate.safetensors", "nope");
  expect_code(main_server.cli->Get("/v1/loras"), 400, "ambiguous_lora");
  fs::remove(tree.root / "lora" / "image" / "duplicate.safetensors");
  fs::remove(tree.root / "lora" / "video" / "duplicate.safetensors");
  test_image_and_jobs(*main_server.cli);
  test_auth_and_limit(tree);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) {
    mx::FakeEngine engine;
    mx::ServerOptions opt;
    if (argc >= 3 && argv[2][0] != '\0') opt.port = std::atoi(argv[2]);
    return mx::run_server(engine, opt);
  }
  try {
    run_all();
    std::cout << "server_test: " << g_ok << " checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "server_test FAIL: " << e.what() << "\n";
    return 1;
  }
}
