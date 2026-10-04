#include "httplib.h"

#include "server/server_internal.h"

#include <chrono>
#include <utility>

#include "core/common.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace mx {
namespace {

constexpr const char* kJson = "application/json; charset=utf-8";

void write_error(httplib::Response& res, int status, const std::string& message, const std::string& type,
                 const std::string& code) {
  res.status = status;
  res.set_content(error_json(message, type, code), kJson);
}

void write_api(httplib::Response& res, const ApiError& e) { write_error(res, e.status, e.what(), e.type, e.code); }

template <class Fn>
void guard(httplib::Response& res, Fn&& fn) {
  try {
    fn();
  } catch (const ApiError& e) {
    write_api(res, e);
  } catch (const std::exception& e) {
    write_error(res, 500, e.what(), "server_error", "internal");
  }
}

bool authorized(const ServerOptions& opt, const httplib::Request& req) {
  if (opt.api_key.empty()) return true;
  const std::string& path = req.path;
  if (path == "/health" || path == "/" || path.rfind("/web/", 0) == 0 || path.rfind("/files/", 0) == 0) return true;
  if (path.rfind("/v1/", 0) != 0) return true;
  return req.get_header_value("Authorization") == "Bearer " + opt.api_key;
}

std::string mime_for(const std::string& name) {
  auto dot = name.rfind('.');
  std::string ext = dot == std::string::npos ? "" : name.substr(dot);
  for (char& c : ext)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  if (ext == ".html") return "text/html; charset=utf-8";
  if (ext == ".css") return "text/css; charset=utf-8";
  if (ext == ".js") return "text/javascript; charset=utf-8";
  if (ext == ".png") return "image/png";
  if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
  if (ext == ".mp4") return "video/mp4";
  if (ext == ".webp") return "image/webp";
  return "application/octet-stream";
}

bool safe_asset_name(const std::string& name) {
  if (name.empty() || name.size() > 200 || name[0] == '.') return false;
  if (name.find("..") != std::string::npos) return false;
  for (unsigned char c : name) {
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
              c == '-';
    if (!ok) return false;
  }
  return true;
}

void serve_named(const std::filesystem::path& dir, const std::string& name, httplib::Response& res) {
  if (!safe_asset_name(name)) {
    write_error(res, 404, "找不到檔案", "not_found", "not_found");
    return;
  }
  std::error_code ec;
  auto path = dir / utf8_path(name);
  if (!std::filesystem::is_regular_file(path, ec)) {
    write_error(res, 404, "找不到檔案", "not_found", "not_found");
    return;
  }
  auto canon_dir = std::filesystem::weakly_canonical(dir, ec);
  auto canon = std::filesystem::weakly_canonical(path, ec);
  if (ec || canon.parent_path() != canon_dir) {
    write_error(res, 404, "找不到檔案", "not_found", "not_found");
    return;
  }
  res.set_file_content(path_utf8(canon), mime_for(name));
}

void wait_job(ServerState& st, const std::shared_ptr<Job>& job, httplib::Response& res) {
  std::unique_lock lk(st.mu);
  st.job_cv.wait(lk, [&] { return st.stopping.load() || job_terminal(job->status); });
  if (!job_terminal(job->status)) {
    write_error(res, 503, "伺服器停止中", "server_error", "shutting_down");
    return;
  }
  if (job->status == "succeeded") {
    res.status = 200;
    res.set_content(image_sync_json(*job), kJson);
    job->png.clear();
    return;
  }
  int status = job->status == "cancelled" ? 499 : 500;
  write_error(res, status, job->err_message.empty() ? "生成失敗" : job->err_message,
              job->err_type.empty() ? "server_error" : job->err_type,
              job->err_code.empty() ? "engine_error" : job->err_code);
}

void serve_events(ServerState& st, const std::shared_ptr<Job>& job, httplib::Response& res) {
  res.set_header("Cache-Control", "no-cache");
  res.set_header("Content-Encoding", "identity");
  res.set_header("X-Accel-Buffering", "no");
  res.set_chunked_content_provider(
      "text/event-stream; charset=utf-8", [&st, job, seen = uint64_t{0}](size_t, httplib::DataSink& sink) mutable {
        if (!sink.is_writable()) return false;
        std::string payload;
        bool done = false;
        {
          std::unique_lock lk(st.mu);
          if (job->seq == seen && !st.stopping.load()) {
            st.job_cv.wait_for(lk, std::chrono::seconds(15),
                               [&] { return st.stopping.load() || job->seq != seen; });
          }
          if (st.stopping.load() && job->seq == seen) return false;
          if (job->seq == seen) {
            payload = ": ping\n\n";
          } else {
            seen = job->seq;
            payload = "data: " + job_to_json(*job) + "\n\n";
            done = job_terminal(job->status);
          }
        }
        if (!sink.write(payload.data(), payload.size())) return false;
        if (done) sink.done();
        return true;
      });
}

}  // namespace

struct Server::Impl {
  Engine& engine;
  ServerOptions opt;
  std::unique_ptr<ServerState> state;
  httplib::Server svr;
  std::thread http;
  int bound_port = 0;
  std::atomic<int> phase{0};
  std::mutex wait_mu;
  std::condition_variable wait_cv;
  bool finished = false;

  Impl(Engine& engine_in, ServerOptions opt_in) : engine(engine_in), opt(std::move(opt_in)) {}

  void signal_finished() {
    std::lock_guard lk(wait_mu);
    finished = true;
    wait_cv.notify_all();
  }
  void install_hooks();
  void route_meta();
  void route_jobs();
  void route_files();
  void routes() {
    install_hooks();
    route_meta();
    route_jobs();
    route_files();
  }
};

void Server::Impl::install_hooks() {
  svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    MX_INFO("%s %s -> %d", req.method.c_str(), req.path.c_str(), res.status);
  });
  svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (!res.body.empty()) return;
    const char* msg = "請求失敗";
    const char* code = "error";
    if (res.status == 413) {
      msg = "請求本文超過大小限制";
      code = "payload_too_large";
    } else if (res.status == 404) {
      msg = "找不到";
      code = "not_found";
    } else if (res.status == 401) {
      msg = "未授權";
      code = "unauthorized";
    }
    res.set_content(error_json(msg, "api_error", code), kJson);
  });
  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const ApiError& e) {
      write_api(res, e);
      return;
    } catch (const std::exception& e) {
      write_error(res, 500, e.what(), "server_error", "internal");
      return;
    }
    write_error(res, 500, "內部錯誤", "server_error", "internal");
  });
  svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
    if (authorized(opt, req)) return httplib::Server::HandlerResponse::Unhandled;
    write_error(res, 401, "需要 Authorization: Bearer 金鑰", "auth_error", "unauthorized");
    return httplib::Server::HandlerResponse::Handled;
  });
  size_t limit = opt.max_body_mb == 0 ? size_t{0} : opt.max_body_mb * 1024ull * 1024ull;
  svr.set_payload_max_length(limit);
  // 同步生成可能要數分鐘，最後才一次寫出較大的 JSON。
  svr.set_read_timeout(120, 0);
  svr.set_write_timeout(120, 0);
}

void Server::Impl::route_meta() {
  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", kJson);
  });
  svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    guard(res, [&] { res.set_content(models_json(engine.models()), kJson); });
  });
  svr.Get("/v1/system", [this](const httplib::Request&, httplib::Response& res) {
    guard(res, [&] { res.set_content(system_json(engine.gpus(), state->queue_length()), kJson); });
  });
  svr.Get("/v1/loras", [this](const httplib::Request&, httplib::Response& res) {
    guard(res, [&] { res.set_content(loras_json(opt), kJson); });
  });
}

void Server::Impl::route_jobs() {
  ServerState* st = state.get();
  svr.Post("/v1/images/generations", [this, st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      auto job = parse_request(opt, engine, req.body, "image");
      job->want_b64 = job->response_format != "url";
      st->accept(job);
      wait_job(*st, job, res);
    });
  });
  svr.Post("/v1/videos/generations", [this, st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      auto job = parse_request(opt, engine, req.body, "video");
      res.set_content(st->accept(job), kJson);
    });
  });
  svr.Post("/v1/jobs", [this, st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      auto job = parse_request(opt, engine, req.body, "");
      res.set_content(st->accept(job), kJson);
    });
  });
  svr.Get("/v1/jobs", [st](const httplib::Request&, httplib::Response& res) {
    guard(res, [&] { res.set_content(st->list_json(), kJson); });
  });
  svr.Get("/v1/jobs/:id/events", [st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      auto job = st->find(req.path_params.at("id"));
      if (!job) {
        write_error(res, 404, "找不到工作", "not_found", "job_not_found");
        return;
      }
      serve_events(*st, job, res);
    });
  });
  svr.Get("/v1/jobs/:id", [st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      bool found = false;
      auto body = st->job_json_by_id(req.path_params.at("id"), found);
      if (!found) write_error(res, 404, "找不到工作", "not_found", "job_not_found");
      else res.set_content(std::move(body), kJson);
    });
  });
  svr.Delete("/v1/jobs/:id", [st](const httplib::Request& req, httplib::Response& res) {
    guard(res, [&] {
      auto id = req.path_params.at("id");
      if (!st->cancel(id)) {
        write_error(res, 404, "找不到工作", "not_found", "job_not_found");
        return;
      }
      bool found = false;
      res.set_content(st->job_json_by_id(id, found), kJson);
    });
  });
}

void Server::Impl::route_files() {
  auto web = utf8_path(opt.web_dir);
  auto out = utf8_path(opt.out_dir);
  svr.Get("/", [web](const httplib::Request&, httplib::Response& res) { serve_named(web, "index.html", res); });
  svr.Get("/web/:name", [web](const httplib::Request& req, httplib::Response& res) {
    serve_named(web, req.path_params.at("name"), res);
  });
  svr.Get("/files/:name", [out](const httplib::Request& req, httplib::Response& res) {
    serve_named(out, req.path_params.at("name"), res);
  });
}

namespace {

Server* g_server = nullptr;

#ifdef _WIN32
BOOL WINAPI on_ctrl(DWORD) {
  if (g_server) g_server->stop();
  return TRUE;
}
#endif

}  // namespace

Server::Server(Engine& engine, ServerOptions opt) : impl_(std::make_unique<Impl>(engine, std::move(opt))) {}

Server::~Server() { stop(); }

void Server::start() {
  if (!impl_ || impl_->phase.load() != 0) throw std::runtime_error("伺服器已啟動");
  impl_->state = std::make_unique<ServerState>(impl_->engine, impl_->opt);
  impl_->routes();
  int bound = -1;
  if (impl_->opt.port == 0) bound = impl_->svr.bind_to_any_port(impl_->opt.host);
  else if (impl_->svr.bind_to_port(impl_->opt.host, impl_->opt.port)) bound = impl_->opt.port;
  if (bound < 0) {
    impl_->state->shutdown();
    impl_->state.reset();
    throw std::runtime_error(strf("無法綁定 %s:%d", impl_->opt.host.c_str(), impl_->opt.port));
  }
  impl_->bound_port = bound;
  impl_->http = std::thread([this] { impl_->svr.listen_after_bind(); });
  impl_->svr.wait_until_ready();
  if (!impl_->svr.is_running()) {
    if (impl_->http.joinable()) impl_->http.join();
    impl_->state->shutdown();
    impl_->state.reset();
    throw std::runtime_error("伺服器沒有進入聆聽狀態");
  }
  impl_->phase.store(1);
}

void Server::stop() {
  if (!impl_) return;
  int expected = 1;
  if (!impl_->phase.compare_exchange_strong(expected, 2)) {
    if (expected == 2) {
      std::unique_lock lk(impl_->wait_mu);
      impl_->wait_cv.wait(lk, [&] { return impl_->finished; });
    }
    return;
  }
  if (impl_->state) impl_->state->shutdown();
  impl_->svr.stop();
  if (impl_->http.joinable()) impl_->http.join();
  impl_->signal_finished();
}

void Server::wait() {
  std::unique_lock lk(impl_->wait_mu);
  impl_->wait_cv.wait(lk, [&] { return impl_->finished; });
}

int Server::port() const { return impl_ ? impl_->bound_port : 0; }

int run_server(Engine& engine, const ServerOptions& opt) {
  try {
    Server server(engine, opt);
    server.start();
    MX_INFO("服務已啟動 http://%s:%d", opt.host.c_str(), server.port());
    g_server = &server;
#ifdef _WIN32
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#endif
    server.wait();
#ifdef _WIN32
    SetConsoleCtrlHandler(on_ctrl, FALSE);
#endif
    g_server = nullptr;
    return 0;
  } catch (const std::exception& e) {
    MX_ERROR("%s", e.what());
    return 1;
  }
}

}  // namespace mx
