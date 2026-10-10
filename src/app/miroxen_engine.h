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
  std::string qi_dit, qi_te, qi_vae, qi_tok, qi_mmproj;
  std::string qi_turbo_dit;  // Qwen-Image-2.1-Turbo（8 步蒸餾版），與標準版共用編碼器／VAE
  int qi_text_device = -1;  // -1 = 自動
  // MiniMax-H3
  std::string h3_dit, h3_te, h3_vvae, h3_avae, h3_tok, h3_dit_ref;
  std::string out_dir = "out";
  std::string compute = "fp8";  // DiT 運算精度：bf16 / fp8 / fp4（fp8/fp4 需 Blackwell，否則自動退回 bf16）
};

std::unique_ptr<Engine> make_engine(const EngineConfig& cfg);
// miroxen serve 子命令（main.cpp 用）。
int cmd_serve(const std::map<std::string, std::string>& kv, const EngineConfig& cfg);

}  // namespace mx
