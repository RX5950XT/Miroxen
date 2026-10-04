#include "app/miroxen_engine.h"

#include <filesystem>
#include <mutex>

#include "core/common.h"
#include "models/qwen_image/pipeline.h"

namespace mx {
namespace fs = std::filesystem;

namespace {

std::string first_existing(const std::string& dir, std::initializer_list<const char*> names) {
  for (auto n : names) {
    fs::path p = fs::path(dir) / n;
    if (fs::exists(p)) return p.string();
  }
  return (fs::path(dir) / *names.begin()).string();
}

class MiroxenEngine : public Engine {
 public:
  explicit MiroxenEngine(EngineConfig c) : cfg_(std::move(c)) {
    std::string qd = (fs::path(cfg_.models_dir) / "qwen-image-2.1").string();
    if (cfg_.qi_dit.empty())
      cfg_.qi_dit = first_existing(qd, {"qwen_image_2.1-Q8_0.gguf", "qwen-image-2.1-Q8_0.gguf",
                                        "qwen_image_2.1-Q4_K.gguf"});
    if (cfg_.qi_te.empty())
      cfg_.qi_te = first_existing(qd, {"Qwen3VL-8B-Instruct-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q8_0.gguf"});
    if (cfg_.qi_vae.empty()) cfg_.qi_vae = first_existing(qd, {"vae/qwen_image_2.1_vae_bf16.safetensors"});
    if (cfg_.qi_tok.empty()) cfg_.qi_tok = first_existing(qd, {"hf/processor/tokenizer.json"});
  }

  std::vector<ModelInfo> models() override {
    std::lock_guard<std::mutex> lk(m_);
    return {{"qwen-image-2.1", "image", qi_ != nullptr, qi_ ? qi_->describe() : cfg_.qi_dit},
            {"minimax-h3", "video", false, "尚未實作"}};
  }

  std::vector<GpuInfo> gpus() override {
    int n = 0;
    MX_CUDA(cudaGetDeviceCount(&n));
    std::vector<GpuInfo> v;
    for (int i = 0; i < n; i++) {
      cudaDeviceProp p;
      MX_CUDA(cudaGetDeviceProperties(&p, i));
      MX_CUDA(cudaSetDevice(i));
      GpuInfo g;
      g.index = i;
      g.name = p.name;
      MX_CUDA(cudaMemGetInfo(&g.free, &g.total));
      v.push_back(g);
    }
    return v;
  }

  ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override {
    std::lock_guard<std::mutex> lk(m_);
    if (!qi_) {
      if (cb) cb(Progress{"load", 0, 1});
      QwenImageDevices d;
      d.text = cfg_.qi_text_device;
      qi_ = std::make_unique<QwenImagePipeline>(
          QwenImagePaths{cfg_.qi_dit, cfg_.qi_te, cfg_.qi_vae, cfg_.qi_tok}, d);
    }
    return qi_->generate(r, cb, cancel);
  }

  VideoResult generate_video(const VideoRequest&, const ProgressFn&, const std::atomic<bool>*) override {
    throw std::runtime_error("MiniMax-H3 尚未實作");
  }

 private:
  EngineConfig cfg_;
  std::mutex m_;
  std::unique_ptr<QwenImagePipeline> qi_;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const EngineConfig& cfg) {
  return std::make_unique<MiroxenEngine>(cfg);
}

}  // namespace mx
