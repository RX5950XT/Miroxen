#pragma once
// HTTP API 與網頁面板。引擎只透過 Engine 呼叫，這個標頭不依賴 httplib。

#include <cstddef>
#include <memory>
#include <string>

#include "app/engine.h"

namespace mx {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8188;
  std::string api_key;  // 空字串表示不驗證
  std::string web_dir = "web";
  std::string out_dir = "out";
  std::string lora_dir = "models/loras";
  size_t max_body_mb = 256;  // 0 表示不限制本文大小（參考影片以 base64 上傳，預設放寬到 256 MB）
};

// 阻塞到 Server::stop()（Ctrl+C 會呼叫它）。失敗回 1。
int run_server(Engine& engine, const ServerOptions& opt);

// 測試用。port 0 會綁到系統分配的埠。engine 的壽命必須蓋過 Server。
class Server {
 public:
  Server(Engine& engine, ServerOptions opt);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  void start();
  void stop();
  void wait();
  int port() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mx
