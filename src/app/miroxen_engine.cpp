#include "app/miroxen_engine.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>

#include "core/common.h"
#include "models/minimax_h3/pipeline.h"
#include "models/qwen_image/pipeline.h"
#include "kernels/blockscale.h"
#include "server/fake_engine.h"
#include "server/server.h"

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

void fill_defaults(EngineConfig& c) {
  std::string qd = (fs::path(c.models_dir) / "qwen-image-2.1").string();
  if (c.qi_dit.empty())
    c.qi_dit = first_existing(qd, {"qwen_image_2.1-Q8_0.gguf", "qwen-image-2.1-Q8_0.gguf", "qwen_image_2.1-Q4_K.gguf"});
  if (c.qi_te.empty())
    c.qi_te = first_existing(qd, {"Qwen3VL-8B-Instruct-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q8_0.gguf"});
  if (c.qi_vae.empty())  // 紋理修正版只改解碼器（編碼器相同），有就優先用
    c.qi_vae = first_existing(qd, {"vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
                                   "vae/qwen_image_2.1_vae_bf16.safetensors"});
  if (c.qi_tok.empty()) c.qi_tok = first_existing(qd, {"hf/processor/tokenizer.json"});
  if (c.qi_mmproj.empty()) c.qi_mmproj = first_existing(qd, {"mmproj-Qwen3VL-8B-Instruct-F16.gguf"});
  std::string hd = (fs::path(c.models_dir) / "minimax-h3").string();
  if (c.h3_dit.empty())
    c.h3_dit = first_existing(hd, {"minimax_h3_fl2va_pruned-Q4_K.gguf", "minimax_h3_fl2va_pruned-Q8_0.gguf",
                                   "minimax_h3_fl2va_pruned-Q6_K.gguf"});
  if (c.h3_dit_ref.empty())
    c.h3_dit_ref = first_existing(hd, {"minimax_h3_ref2va_pruned-Q4_K.gguf", "minimax_h3_ref2va_pruned-Q8_0.gguf",
                                       "minimax_h3_ref2va_pruned-Q6_K.gguf", "minimax_h3_ref2va_pruned-Q4_K_M.gguf"});
  if (c.h3_te.empty()) c.h3_te = first_existing(hd, {"qwen3vl_32b_minimax_h3-Q4_K_M.gguf"});
  if (c.h3_vvae.empty()) c.h3_vvae = first_existing(hd, {"vae/minimax_h3_video_vae_fp16.safetensors"});
  if (c.h3_avae.empty()) c.h3_avae = first_existing(hd, {"vae/minimax_h3_audio_vae_fp32.safetensors"});
  if (c.h3_tok.empty()) c.h3_tok = first_existing(hd, {"hf/tokenizer/tokenizer.json"});
}

class MiroxenEngine : public Engine {
 public:
  explicit MiroxenEngine(EngineConfig c) : cfg_(std::move(c)) { fill_defaults(cfg_); }

  std::vector<ModelInfo> models() override {
    std::lock_guard<std::mutex> lk(m_);
    return {{"qwen-image-2.1", "image", qi_ != nullptr, qi_ ? qi_->describe() : cfg_.qi_dit},
            {"minimax-h3", "video", h3_ != nullptr, h3_ ? h3_->describe() : cfg_.h3_dit}};
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
    const QMode want = parse_qmode(r.compute.empty() ? cfg_.compute : r.compute);
    if (qi_ && qi_compute_ != want) qi_.reset();  // 換運算精度要重新載入（權重格式不同）
    if (!qi_) {
      h3_.reset();  // 兩個模型不同時常駐，釋放 VRAM 與鎖頁記憶體
      if (cb) cb(Progress{"load", 0, 1});
      QwenImageDevices d;
      d.text = cfg_.qi_text_device;
      d.compute = qi_compute_ = want;
      qi_ = std::make_unique<QwenImagePipeline>(QwenImagePaths{cfg_.qi_dit, cfg_.qi_te, cfg_.qi_vae, cfg_.qi_tok, cfg_.qi_mmproj}, d);
    }
    return qi_->generate(r, cb, cancel);
  }

  VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override {
    std::lock_guard<std::mutex> lk(m_);
    const QMode want = parse_qmode(r.compute.empty() ? cfg_.compute : r.compute);
    if (h3_ && h3_compute_ != want) h3_.reset();
    if (!h3_) {
      qi_.reset();
      if (cb) cb(Progress{"load", 0, 1});
      H3Paths p{cfg_.h3_dit, cfg_.h3_te, cfg_.h3_vvae, cfg_.h3_avae, cfg_.h3_tok, cfg_.h3_dit_ref, cfg_.out_dir};
      H3Devices d;
      d.compute = h3_compute_ = want;
      h3_ = std::make_unique<H3Pipeline>(p, d);
    }
    return h3_->generate(r, cb, cancel);
  }

 private:
  EngineConfig cfg_;
  std::mutex m_;
  std::unique_ptr<QwenImagePipeline> qi_;
  std::unique_ptr<H3Pipeline> h3_;
  QMode qi_compute_ = QMode::BF16, h3_compute_ = QMode::BF16;  // 已載入管線所用的精度
};

}  // namespace

std::unique_ptr<Engine> make_engine(const EngineConfig& cfg) {
  return std::make_unique<MiroxenEngine>(cfg);
}

int cmd_serve(const std::map<std::string, std::string>& kv, const EngineConfig& cfg) {
  auto get = [&](const char* k, const std::string& d) {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  };
  ServerOptions opt;
  opt.host = get("--host", opt.host);
  opt.port = std::atoi(get("--port", std::to_string(opt.port)).c_str());
  opt.api_key = get("--api-key", "");
  opt.web_dir = get("--web", opt.web_dir);
  opt.out_dir = get("--out", cfg.out_dir);
  opt.lora_dir = get("--loras", (fs::path(cfg.models_dir) / "loras").string());
  opt.max_body_mb = size_t(std::atoll(get("--max-body", std::to_string(opt.max_body_mb)).c_str()));
  if (kv.count("--fake")) {
    FakeEngine fake;
    return run_server(fake, opt);
  }
  auto engine = make_engine(cfg);
  return run_server(*engine, opt);
}

}  // namespace mx
