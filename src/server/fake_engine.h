#pragma once
// 測試與 `miroxen serve --fake` 用的假引擎。不載入模型。

#include "app/engine.h"

namespace mx {

class FakeEngine : public Engine {
 public:
  std::vector<ModelInfo> models() override;
  std::vector<GpuInfo> gpus() override;
  void load_model(const std::string& id, const std::map<std::string, std::string>&, const std::string&) override {
    loaded_image_ = id == "qwen-image-2.1"; loaded_turbo_ = id == kQwenTurboId; loaded_video_ = id == "minimax-h3";
  }
  void unload_models() override { loaded_image_ = false; loaded_turbo_ = false; loaded_video_ = false; }
  ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override;
  VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override;
 private:
  std::atomic<bool> loaded_image_{true}, loaded_turbo_{false}, loaded_video_{true};
};

}  // namespace mx
