#pragma once
// 真正的引擎：依需求載入各模型管線，序列化執行。
#include <map>
#include <memory>
#include <string>

#include "app/engine.h"

namespace mx {

struct EngineConfig {
  std::string models_dir = "models";
  // Qwen-Image-2.1（相對於 models_dir；空字串用預設）
  std::string qi_dit, qi_te, qi_vae, qi_tok;
  int qi_text_device = -1;  // -1 = 自動
};

std::unique_ptr<Engine> make_engine(const EngineConfig& cfg);
// miroxen serve 子命令（main.cpp 用）。
int cmd_serve(const std::map<std::string, std::string>& kv, const EngineConfig& cfg);

}  // namespace mx
