#pragma once
// 伺服器內部。HTTP 層、工作佇列、請求剖析共用。

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "app/engine.h"
#include "server/server.h"

namespace mx {

struct ApiError : std::runtime_error {
  int status;
  std::string type;
  std::string code;
  ApiError(int status_, std::string type_, std::string code_, std::string message)
      : std::runtime_error(std::move(message)),
        status(status_),
        type(std::move(type_)),
        code(std::move(code_)) {}
};

struct Job {
  std::string id;
  std::string type;  // image | video
  std::string status = "queued";
  std::string stage;
  int step = 0;
  int total = 0;
  std::string url;
  int64_t seed = 0;
  double seconds = 0;
  std::map<std::string, double> timings;
  bool has_result = false;
  bool want_b64 = false;
  std::string response_format = "b64_json";
  std::vector<uint8_t> png;
  std::string err_message, err_type, err_code;
  std::shared_ptr<std::atomic<bool>> cancel = std::make_shared<std::atomic<bool>>(false);
  int64_t created = 0;
  uint64_t seq = 0;
  std::string prompt;
  ImageRequest image;
  VideoRequest video;
  std::vector<std::filesystem::path> temps;
};

inline bool job_terminal(const std::string& status) {
  return status == "succeeded" || status == "failed" || status == "cancelled";
}

inline std::filesystem::path utf8_path(const std::string& s) {
  return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

inline std::string path_utf8(const std::filesystem::path& p) {
  auto u = p.u8string();
  return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

class ServerState {
 public:
  ServerState(Engine& engine, ServerOptions opt);
  ~ServerState();

  Engine& engine;
  ServerOptions opt;
  std::mutex mu;
  std::condition_variable work_cv;
  std::condition_variable job_cv;
  std::deque<std::shared_ptr<Job>> queue;
  std::deque<std::shared_ptr<Job>> all;
  std::unordered_map<std::string, std::shared_ptr<Job>> by_id;
  std::atomic<bool> stopping{false};
  std::thread worker;
  uint64_t next_n = 1;

  // 持有鎖把工作放進佇列，回傳當時（一定是 queued）的 JSON。
  std::string accept(const std::shared_ptr<Job>& job);
  std::string list_json();
  // found 為 false 時回空字串。
  std::string job_json_by_id(const std::string& id, bool& found);
  std::shared_ptr<Job> find(const std::string& id);
  // 找不到回 false。排隊中的工作會直接標成 cancelled。
  bool cancel(const std::string& id);
  int queue_length();
  void control_model(const std::string& id, const std::map<std::string, std::string>& files, const std::string& compute);
  void shutdown();

 private:
  bool changing_model = false;
  void worker_loop();
  void execute(const std::shared_ptr<Job>& job);
  void prune_locked();
};

std::shared_ptr<Job> parse_request(const ServerOptions& opt, Engine& engine, const std::string& body,
                                   const std::string& forced_type);

std::string error_json(const std::string& message, const std::string& type, const std::string& code);
std::string job_to_json(const Job& job);
std::string jobs_to_json(const std::vector<std::shared_ptr<Job>>& jobs);
std::string image_sync_json(const Job& job);
std::string models_json(const std::vector<ModelInfo>& models);
std::string loras_json(const ServerOptions& opt);
std::string system_json(const std::vector<GpuInfo>& gpus, int queue_length, uint64_t requests = 0,
                        uint64_t errors = 0, double uptime = 0, bool authenticated = false);

}  // namespace mx
