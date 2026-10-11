#include "app/miroxen_engine.h"

#include <cstdlib>
#include <algorithm>
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
  std::string qd = (fs::path(c.models_dir) / "image/qwen-image-2.1").string();
  if (c.qi_dit.empty())
    c.qi_dit = first_existing(qd, {"qwen-image-2.1-UC-Q8_0.gguf"});
  if (c.qi_turbo_dit.empty())
    c.qi_turbo_dit = first_existing(qd, {"Qwen-Image-2.1-Turbo-Q8_0.gguf", "Qwen-Image-2.1-Turbo-AD-Q6_K.gguf",
                                         "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf"});
  if (c.qi_te.empty())  // AtomicChat 去拒答版 AD-Q4_K 比一般 Q4_K_M 更接近 BF16，且放得進 8 GB 副卡
    c.qi_te = first_existing(qd, {"Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf",
                                  "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q8_0.gguf"});
  if (c.qi_vae.empty())  // 紋理修正版只改解碼器（編碼器相同），有就優先用
    c.qi_vae = first_existing(qd, {"vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
                                   "vae/qwen_image_2.1_vae_bf16.safetensors"});
  if (c.qi_tok.empty()) c.qi_tok = first_existing(qd, {"hf/processor/tokenizer.json"});
  if (c.qi_mmproj.empty()) c.qi_mmproj = first_existing(qd, {"mmproj-Qwen3VL-8B-Instruct-F16.gguf"});
  std::string hd = (fs::path(c.models_dir) / "video/minimax-h3").string();
  if (c.h3_dit.empty())
    c.h3_dit = first_existing(hd, {"minimax_h3_fl2va_pruned-Q4_K.gguf", "minimax_h3_fl2va_pruned-Q8_0.gguf",
                                   "minimax_h3_fl2va_pruned-Q6_K.gguf"});
  if (c.h3_dit_ref.empty())
    c.h3_dit_ref = first_existing(hd, {"minimax_h3_ref2va_pruned-Q4_K.gguf", "minimax_h3_ref2va_pruned-Q8_0.gguf",
                                       "minimax_h3_ref2va_pruned-Q6_K.gguf", "minimax_h3_ref2va_pruned-Q4_K_M.gguf"});
  if (c.h3_te.empty())  // pottokao 去拒答版 Q3_K_S（含視覺塔）比 unsloth Q4_K_M 小 5.6 GB，實測畫面同級
    c.h3_te = first_existing(hd, {"MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S_vis.gguf",
                                  "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"});
  if (c.h3_vvae.empty()) c.h3_vvae = first_existing(hd, {"vae/minimax_h3_video_vae_fp16.safetensors"});
  if (c.h3_avae.empty()) c.h3_avae = first_existing(hd, {"vae/minimax_h3_audio_vae_fp32.safetensors"});
  if (c.h3_tok.empty()) c.h3_tok = first_existing(hd, {"hf/tokenizer/tokenizer.json"});
}

class MiroxenEngine : public Engine {
 public:
  explicit MiroxenEngine(EngineConfig c) : cfg_(std::move(c)) {
    fill_defaults(cfg_);
    qi_info_ = inventory("qwen-image-2.1", "image", {{"dit", cfg_.qi_dit}, {"text_encoder", cfg_.qi_te},
                          {"vae", cfg_.qi_vae}, {"vision", cfg_.qi_mmproj}});
    qt_info_ = inventory(kQwenTurboId, "image", {{"dit", cfg_.qi_turbo_dit}, {"text_encoder", cfg_.qi_te},
                          {"vae", cfg_.qi_vae}, {"vision", cfg_.qi_mmproj}});
    h3_info_ = inventory("minimax-h3", "video", {{"dit", cfg_.h3_dit}, {"dit_ref", cfg_.h3_dit_ref},
                          {"text_encoder", cfg_.h3_te}, {"video_vae", cfg_.h3_vvae}, {"audio_vae", cfg_.h3_avae}});
  }

  std::vector<ModelInfo> models() override {
    std::lock_guard lk(meta_mu_);  // 不等待生成鎖，取樣中也能查 API 狀態
    return {qi_info_, qt_info_, h3_info_};
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
      char bus[32]{};
      MX_CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), i));
      g.pci_bus = bus;
      MX_CUDA(cudaMemGetInfo(&g.free, &g.total));
      v.push_back(g);
    }
    return v;
  }

  ImageResult generate_image(const ImageRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override {
    std::lock_guard<std::mutex> lk(m_);
    ensure_image(r.model, r.model_files, r.compute, cb);
    return qi_->generate(r, cb, cancel);
  }

  VideoResult generate_video(const VideoRequest& r, const ProgressFn& cb,
                             const std::atomic<bool>* cancel) override {
    std::lock_guard<std::mutex> lk(m_);
    ensure_video(r.model_files, r.compute, cb);
    return h3_->generate(r, [&](const Progress& p) {
      if (p.stage == "encode") publish(true);  // 管線可能切換 FL2VA／Ref2VA，同步實際 DiT 資訊。
      if (cb) cb(p);
    }, cancel);
  }

  void load_model(const std::string& id, const std::map<std::string, std::string>& files, const std::string& compute) override {
    std::lock_guard lk(m_);
    if (id == "qwen-image-2.1" || id == kQwenTurboId) ensure_image(id, files, compute, {});
    else if (id == "minimax-h3") ensure_video(files, compute, {});
    else MX_CHECK(false, "找不到模型");
  }
  void unload_models() override {
    std::lock_guard lk(m_);
    qi_.reset(); h3_.reset(); publish(false);
  }

 private:
  void ensure_image(const std::string& id, const std::map<std::string, std::string>& selected, const std::string& compute,
                    const ProgressFn& cb) {
    MX_CHECK(id == "qwen-image-2.1" || id == kQwenTurboId, "找不到圖片模型 %s", id.c_str());
    const bool turbo = id == kQwenTurboId;
    auto files = select_files(turbo ? qt_info_ : qi_info_, selected);
    const QMode want = parse_qmode(compute.empty() ? cfg_.compute : compute);
    if (qi_ && (qi_compute_ != want || files != qi_active_ || qi_turbo_ != turbo)) { qi_.reset(); publish(false); }
    if (!qi_) {
      h3_.reset();  // 兩個模型不同時常駐，釋放 VRAM 與鎖頁記憶體
      publish(false);
      if (cb) cb(Progress{"load", 0, 1});
      QwenImageDevices d;
      d.text = cfg_.qi_text_device;
      d.compute = qi_compute_ = want;
      qi_ = std::make_unique<QwenImagePipeline>(
          QwenImagePaths{files["dit"], files["text_encoder"], files["vae"], cfg_.qi_tok, files["vision"], turbo}, d);
      qi_active_ = files;
      qi_turbo_ = turbo;
      publish(true);
    }
  }
  void ensure_video(const std::map<std::string, std::string>& selected, const std::string& compute, const ProgressFn& cb) {
    auto files = select_files(h3_info_, selected);
    const QMode want = parse_qmode(compute.empty() ? cfg_.compute : compute);
    if (h3_ && (h3_compute_ != want || files != h3_active_)) { h3_.reset(); publish(false); }
    if (!h3_) {
      qi_.reset();
      publish(false);
      if (cb) cb(Progress{"load", 0, 1});
      H3Paths p{files["dit"], files["text_encoder"], files["video_vae"], files["audio_vae"], cfg_.h3_tok, files["dit_ref"], cfg_.out_dir};
      H3Devices d;
      d.compute = h3_compute_ = want;
      h3_ = std::make_unique<H3Pipeline>(p, d);
      h3_active_ = files;
      publish(true);
    }
  }

  ModelInfo inventory(const std::string& id, const std::string& kind,
                      const std::map<std::string, std::string>& defaults) {
    ModelInfo info{id, kind, false, "尚未載入"};
    info.compute = cfg_.compute;
    for (const auto& [role, path] : defaults) {
      const fs::path base(path);
      info.defaults[role] = base.filename().string();
      auto& list = info.files[role];
      list.push_back(base.filename().string());
      paths_[id + "/" + role] = base.parent_path().string();
      defaults_[id + "/" + role] = path;
      if (!fs::is_directory(base.parent_path())) continue;
      for (const auto& ent : fs::directory_iterator(base.parent_path())) {
        if (!ent.is_regular_file() || ent.path() == base) continue;
        auto name = ent.path().filename().string();
        bool match = ent.path().extension() == base.extension();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
        const bool turbo_file = lower.starts_with("qwen-image-2.1-turbo-") && lower.find("abliterated") == std::string::npos;
        if (role == "dit") match &= kind == "image" ? (id == kQwenTurboId ? turbo_file
                                                                          : (lower.starts_with("qwen_image_2.1-") || lower.starts_with("qwen-image-2.1-")) && lower.find("turbo") == std::string::npos)
                                                    : name.starts_with("minimax_h3_fl2va_");
        if (role == "dit_ref") match &= name.starts_with("minimax_h3_ref2va_");
        if (role == "text_encoder") match &= kind == "image" ? name.starts_with("Qwen3VL-8B-Instruct-") || name.starts_with("Qwen-Image-2.1-Turbo-Abliterated-Uncensored-")
                                                             : name.starts_with("qwen3vl_32b_minimax_h3-") || (name.starts_with("MiniMax-H3-TextEncoder-") && name.ends_with("_vis.gguf"));
        if (role == "vision") match &= name.starts_with("mmproj-Qwen3VL-8B-Instruct-");
        if (role == "video_vae") match &= name.starts_with("minimax_h3_video_vae_");
        if (role == "audio_vae") match &= name.starts_with("minimax_h3_audio_vae_");
        if (match) list.push_back(name);
      }
      std::sort(list.begin(), list.end());
    }
    return info;
  }

  std::map<std::string, std::string> select_files(const ModelInfo& info,
                           const std::map<std::string, std::string>& selected) {
    std::map<std::string, std::string> result;
    for (const auto& [role, choices] : info.files) {
      auto it = selected.find(role);
      if (it == selected.end()) { result[role] = defaults_.at(info.id + "/" + role); continue; }
      MX_CHECK(std::find(choices.begin(), choices.end(), it->second) != choices.end(), "模型檔不在已安裝清單");
      auto base = fs::weakly_canonical(paths_.at(info.id + "/" + role));
      auto path = fs::weakly_canonical(base / it->second);
      MX_CHECK(path.parent_path() == base && fs::is_regular_file(path), "模型檔不存在或路徑無效");
      result[role] = path.string();
    }
    return result;
  }

  void publish(bool ready) {
    std::lock_guard lk(meta_mu_);
    for (ModelInfo* info : {&qi_info_, &qt_info_}) {
      info->loaded = ready && qi_ != nullptr && qi_turbo_ == (info == &qt_info_);
      info->detail = info->loaded ? qi_->describe() : "尚未載入";
      info->compute = cfg_.compute;
      info->active_files = info->loaded ? qi_active_ : std::map<std::string, std::string>{};
    }
    h3_info_.loaded = ready && h3_ != nullptr;
    h3_info_.detail = h3_info_.loaded ? h3_->describe() : "尚未載入";
    h3_info_.active_files = h3_info_.loaded ? h3_active_ : std::map<std::string, std::string>{};
  }

  EngineConfig cfg_;
  std::mutex meta_mu_;
  ModelInfo qi_info_, qt_info_, h3_info_;
  std::map<std::string, std::string> paths_, defaults_, qi_active_, h3_active_;
  std::mutex m_;
  std::unique_ptr<QwenImagePipeline> qi_;
  std::unique_ptr<H3Pipeline> h3_;
  QMode qi_compute_ = QMode::BF16, h3_compute_ = QMode::BF16;  // 已載入管線所用的精度
  bool qi_turbo_ = false;  // 已載入的圖片管線是 Turbo
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
