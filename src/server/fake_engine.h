#pragma once
// 測試與 `miroxen serve --fake` 用的假引擎。不載入模型。

#include "app/engine.h"

namespace mx {

class FakeEngine : public Engine {
 public:
  std::vector<ModelInfo> models() override;
  std::vector<GpuInfo> gpus() override;
  ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override;
  VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override;
};

}  // namespace mx
