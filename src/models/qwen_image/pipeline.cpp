#include "models/qwen_image/pipeline.h"

#include <chrono>
#include <cmath>
#include <random>

#include "server/image_io.h"
#include "kernels/ops.h"

namespace mx {

static const char* kSysPrompt = "Comprehend and analyze the provided prompt.";

std::vector<float> qwen_image_sigmas(int steps, int64_t image_tokens, bool shift_terminal,
                                     const std::vector<float>& nodes) {
  if (!nodes.empty()) steps = int(nodes.size());
  MX_CHECK(steps >= 1, "steps 必須 >= 1");
  // scheduler_config：base_image_seq_len 256、max_image_seq_len 8192、base_shift 0.5、max_shift 0.9
  const double m = (0.9 - 0.5) / (8192.0 - 256.0), b = 0.5 - m * 256.0;
  const double mu = double(image_tokens) * m + b, emu = std::exp(mu);
  std::vector<double> s(steps);
  for (int i = 0; i < steps; i++) {
    double lin = !nodes.empty() ? double(nodes[i]) : steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);
    MX_CHECK(lin > 0.0 && lin <= 1.0, "sigma 節點必須在 (0, 1]");
    s[i] = emu / (emu + (1.0 / lin - 1.0));
  }
  const double sf = shift_terminal ? (1.0 - s.back()) / (1.0 - 0.02) : 1.0;  // shift_terminal 0.02
  std::vector<float> out(steps + 1, 0.f);
  for (int i = 0; i < steps; i++) out[i] = float(1.0 - (1.0 - s[i]) / sf);
  return out;
}

void qwen_image_dims(double area, double ratio, int& w, int& h) {
  double width = std::sqrt(area * ratio), height = width / ratio;
  w = int(std::nearbyint(width / 32.0)) * 32;  // Python round()：四捨六入五成雙
  h = int(std::nearbyint(height / 32.0)) * 32;
  w = std::max(w, 32), h = std::max(h, 32);
}

static int pick_text_device(int requested, size_t te_bytes) {
  if (requested >= 0) return requested;
  int n = 0;
  MX_CUDA(cudaGetDeviceCount(&n));
  if (n < 2) return 0;
  size_t fr = 0, tot = 0;
  MX_CUDA(cudaSetDevice(1));
  MX_CUDA(cudaMemGetInfo(&fr, &tot));
  return fr > te_bytes + (size_t(1) << 30) ? 1 : 0;
}

QwenImagePipeline::QwenImagePipeline(const QwenImagePaths& p, QwenImageDevices d)
    : paths_(p), dev_(d) {
  tok_ = Tokenizer::load(p.tokenizer);
  te_st_.add_file(p.text_encoder, "\x01");
  dit_st_.add_file(p.dit);
  vae_st_.add_file(p.vae);
  size_t te_bytes = 0;
  for (auto& n : te_st_.names()) te_bytes += te_st_.get(n).nbytes;
  dev_.text = pick_text_device(d.text, te_bytes);

  te_ = std::make_unique<Qwen3TextEncoder>(te_st_, dev_.text);
  te_ctx_ = std::make_unique<Ctx>(dev_.text, size_t(512) << 20, te_->scratch_bytes() * 2);
  dit_ = std::make_unique<QwenImageDit>(dit_st_, dev_.dit, d.compute);
  vae_ = std::make_unique<QwenImageVae>(vae_st_, dev_.vae);

  // 最大解析度 2752×1536（16512 token）+ 2048 文字 token 的啟用值上限；編輯時依參考圖再放大。
  ensure_dit_ctx(16512 + 2048);
  vae_ctx_ = dev_.vae == dev_.dit
                 ? nullptr
                 : std::make_unique<Ctx>(dev_.vae, vae_->decode_act_bytes(96, 172), 0);

  std::string sys = std::string("<|im_start|>system\n") + kSysPrompt + "<|im_end|>\n";
  drop_idx_ = int(tok_->encode(sys).size());
  MX_INFO("Qwen-Image 管線就緒：DiT@GPU%d、文字編碼器@GPU%d、VAE@GPU%d", dev_.dit, dev_.text,
          dev_.vae);
}

QwenImagePipeline::~QwenImagePipeline() = default;

void QwenImagePipeline::ensure_dit_ctx(int64_t rows) {
  const auto& c = dit_->cfg();
  size_t act = size_t(rows) * (c.dim * 4 + 5 * c.dim * 2 + 3 * c.ffn * 2) + (size_t(64) << 20);
  if (dit_->compute() != QMode::BF16) act += size_t(rows) * c.ffn * 9 / 8;  // 啟用值量化暫存
  if (dit_ctx_ && dit_ctx_->act.capacity() >= act) return;
  dit_ctx_.reset();
  dit_ctx_ = std::make_unique<Ctx>(dev_.dit, act, dit_->scratch_bytes());
}

std::string QwenImagePipeline::describe() const {
  return strf("DiT %s/%s @GPU%d (%.1f GiB) · TE %d 層 @GPU%d (%.1f GiB) · VAE @GPU%d",
              gtype_name(dit_st_.get("transformer_blocks.0.attn.to_q.weight").type), qmode_name(dit_->compute()), dev_.dit,
              dit_->weight_bytes() / 1073741824.0, te_->layers(), dev_.text,
              te_->weight_bytes() / 1073741824.0, dev_.vae);
}

std::vector<bf16> QwenImagePipeline::encode_prompt(const std::string& prompt, int64_t& T,
                                                   const std::vector<Rgba>& refs, std::vector<int64_t>* slots) {
  // 編輯範本：每張圖前面加 "<imageN>" 標籤與視覺區塊（第二張起以空白分隔），prompt 緊接在最後一個區塊後。
  std::vector<VisionFeatures> feats;
  std::string vision;
  if (!refs.empty()) {
    MX_CHECK(!paths_.mmproj.empty(), "影像編輯需要 Qwen3-VL 視覺塔（mmproj）");
    if (!vis_) {
      vis_st_.add_file(paths_.mmproj, "\x01");
      vis_ = std::make_unique<Qwen3VisionEncoder>(vis_st_, dev_.text);
    }
    for (size_t i = 0; i < refs.size(); i++) {
      const Rgba& r = refs[i];
      size_t need = vis_->act_bytes(r.h, r.w);
      if (!vis_ctx_ || vis_ctx_->act.capacity() < need)
        vis_ctx_ = std::make_unique<Ctx>(dev_.text, need, vis_->scratch_bytes());
      auto rgb = flatten_white(r);  // checkpoint 訓練時視覺塔看的是白底合成
      feats.push_back(vis_->encode(*vis_ctx_, rgb.data(), r.h, r.w));
      vision += strf("%s<image%zu><|vision_start|>", i ? " " : "", i + 1);
      for (int64_t k = 0; k < feats.back().tokens(); k++) vision += "<|image_pad|>";
      vision += "<|vision_end|>";
    }
  }
  std::string text = std::string("<|im_start|>system\n") + kSysPrompt + "<|im_end|>\n<|im_start|>user\n" + vision +
                     (prompt.empty() ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n";
  auto ids = tok_->encode(text);
  MX_CHECK(int(ids.size()) > drop_idx_, "prompt 編碼後長度異常");
  MX_CHECK(ids.size() <= 4096 + size_t(vision.size() / 13), "prompt 太長（%zu token）", ids.size());
  const int64_t n = int64_t(ids.size()), dim = te_->dim();
  if (te_ctx_->act.capacity() < te_->act_bytes(n)) {
    te_ctx_.reset();
    te_ctx_ = std::make_unique<Ctx>(dev_.text, te_->act_bytes(n), te_->scratch_bytes() * 2);
  }
  Ctx& c = *te_ctx_;
  ArenaScope sc(c.act);
  bf16* h = c.act.alloc_n<bf16>(n * dim);
  if (feats.empty()) {
    te_->encode(c, ids, h);
  } else {
    const int32_t pad = tok_->token_id("<|image_pad|>");
    std::vector<const VisionFeatures*> fp;
    for (auto& f : feats) fp.push_back(&f);
    TextVisionInput tv = build_text_vision(ids, {pad}, fp);
    te_->encode(c, ids, h, &tv);
    if (slots) {
      slots->clear();
      for (int64_t i = 0; i < n; i++)
        if (ids[i] == pad && (i == 0 || ids[i - 1] != pad)) slots->push_back(i - drop_idx_);
    }
  }
  T = n - drop_idx_;
  std::vector<bf16> out(size_t(T) * dim);
  MX_CUDA(cudaMemcpy(out.data(), h + drop_idx_ * dim, out.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
  return out;
}

void QwenImagePipeline::apply_loras(const std::vector<LoraSpec>& loras) {
  bool same = loras.size() == active_loras_.size();
  for (size_t i = 0; same && i < loras.size(); i++)
    same = loras[i].path == active_loras_[i].path && loras[i].scale == active_loras_[i].scale;
  if (same) return;
  auto ls = dit_->linears();
  clear_loras(ls);
  active_loras_.clear();
  auto names = dit_->linear_names();
  auto t0 = std::chrono::steady_clock::now();
  try {
    for (auto& l : loras) {
      auto st = apply_lora(l.path, l.scale, dev_.dit, names,
                           [&](const std::string& n, LoraTarget& t) { return dit_->resolve_lora(n, t); });
      MX_CHECK(st.applied > 0, "LoRA %s 沒有任何層對應到 Qwen-Image-2.1", l.path.c_str());
      active_loras_.push_back(l);
    }
    dit_->merge_loras(*dit_ctx_);
  } catch (...) {
    clear_loras(ls);  // 還原成原始權重，避免殘留一半的 LoRA
    active_loras_.clear();
    dit_->merge_loras(*dit_ctx_);
    throw;
  }
  if (dit_->compute() != QMode::BF16)
    MX_INFO("LoRA 載入並併入 %s 權重：%.1f 秒", qmode_name(dit_->compute()),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

ImageResult QwenImagePipeline::generate(const ImageRequest& r, const ProgressFn& cb,
                                        const std::atomic<bool>* cancel) {
  auto t0 = std::chrono::steady_clock::now();
  MX_CHECK(r.ref_images.size() <= 10, "參考圖最多 10 張");
  auto report = [&](const char* st, int i, int n) {
    if (cb) cb(Progress{st, i, n});
    if (cancel && cancel->load()) throw std::runtime_error("已取消");
  };
  apply_loras(r.loras);
  report("encode", 0, 1);
  // 參考圖：各自縮放到約 1024² 像素（保持比例、32 的倍數），同一份同時給視覺塔（白底）與 VAE（RGBA）。
  std::vector<Rgba> refs;
  for (auto& path : r.ref_images) {
    Rgba img = load_rgba(path);
    int w, h;
    qwen_image_dims(1024.0 * 1024.0, double(img.w) / img.h, w, h);
    refs.push_back(resize_lanczos(img, w, h));
  }
  int out_w = r.width, out_h = r.height;
  if (!refs.empty() && (out_w <= 0 || out_h <= 0))
    qwen_image_dims(1024.0 * 1024.0, double(refs.back().w) / refs.back().h, out_w, out_h);
  MX_CHECK(out_w > 0 && out_h > 0 && out_w % 32 == 0 && out_h % 32 == 0, "寬高必須是 32 的倍數");
  const bool cfg = r.cfg_scale > 1.f && !r.negative_prompt.empty();
  int64_t T = 0, Tn = 0;
  std::vector<int64_t> slots, nslots;
  auto txt = encode_prompt(r.prompt, T, refs, &slots);
  std::vector<bf16> ntxt;
  if (cfg) ntxt = encode_prompt(r.negative_prompt, Tn, refs, &nslots);

  const int64_t H = out_h / 16, W = out_w / 16, N = H * W, C = dit_->cfg().in_ch;
  // 參考圖 latent：VAE 取分佈平均，(z − mean) / std，排成 [H·W, 64]。
  std::vector<std::vector<float>> cond_lat;
  int64_t cond_rows = 0;
  for (auto& ref : refs) {
    Ctx& vc = vae_ctx_ ? *vae_ctx_ : *dit_ctx_;
    const int64_t lh = ref.h / 16, lw = ref.w / 16, n = lh * lw;
    std::vector<float> z(n * C), rows(n * C);
    {
      ArenaScope sc(vc.act);
      float* dz = vc.act.alloc_n<float>(n * C);
      vae_->encode(vc, ref.px.data(), ref.h, ref.w, dz);
      MX_CUDA(cudaMemcpy(z.data(), dz, z.size() * 4, cudaMemcpyDeviceToHost));
    }
    const float *mean = QwenImageVae::latents_mean(), *stdv = QwenImageVae::latents_std();
    for (int64_t i = 0; i < n; i++)
      for (int64_t ch = 0; ch < C; ch++) rows[i * C + ch] = (z[ch * n + i] - mean[ch]) / stdv[ch];
    cond_lat.push_back(std::move(rows));
    cond_rows += n;
  }
  ensure_dit_ctx(T + cond_rows + N);
  Ctx& c = *dit_ctx_;
  c.use();
  std::vector<DevBuf> cond_dev;
  auto conds_for = [&](const std::vector<int64_t>& sl) {
    std::vector<QwenImageCond> v;
    for (size_t i = 0; i < refs.size(); i++) {
      if (cond_dev.size() <= i) {
        cond_dev.emplace_back(cond_lat[i].size() * 4, c.device);
        MX_CUDA(cudaMemcpy(cond_dev[i].get(), cond_lat[i].data(), cond_lat[i].size() * 4, cudaMemcpyHostToDevice));
      }
      v.push_back({sl[i], refs[i].h / 16, refs[i].w / 16, cond_dev[i].as<float>()});
    }
    return v;
  };
  QwenImageSeq pos, neg;
  {
    DevBuf d(txt.size() * sizeof(bf16), c.device);
    MX_CUDA(cudaMemcpy(d.get(), txt.data(), txt.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    dit_->prepare(c, pos, d.as<bf16>(), T, H, W, conds_for(slots));
    if (cfg) {
      DevBuf dn(ntxt.size() * sizeof(bf16), c.device);
      MX_CUDA(cudaMemcpy(dn.get(), ntxt.data(), ntxt.size() * sizeof(bf16), cudaMemcpyHostToDevice));
      dit_->prepare(c, neg, dn.as<bf16>(), Tn, H, W, conds_for(nslots));
    }
  }

  int64_t seed = r.seed >= 0 ? r.seed : int64_t(std::random_device{}() & 0x7fffffff);
  std::vector<float> noise(N * C);
  {
    std::mt19937_64 rng{uint64_t(seed)};
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : noise) v = nd(rng);
  }
  DevBuf lat(noise.size() * 4, c.device), vel(noise.size() * 4, c.device), nvel;
  if (cfg) nvel = DevBuf(noise.size() * 4, c.device);
  MX_CUDA(cudaMemcpy(lat.get(), noise.data(), noise.size() * 4, cudaMemcpyHostToDevice));

  auto sig = qwen_image_sigmas(r.steps, N, r.shift_terminal, r.sigmas);
  const int steps = int(sig.size()) - 1;
  for (int i = 0; i < steps; i++) {
    report("denoise", i, steps);
    dit_->forward(c, pos, lat.as<float>(), sig[i], vel.as<float>());
    if (cfg) {
      dit_->forward(c, neg, lat.as<float>(), sig[i], nvel.as<float>());
      // v = neg + s·(pos − neg)
      axpby_f32(vel.as<float>(), nvel.as<float>(), r.cfg_scale, 1.f - r.cfg_scale, N * C, c.stream);
    }
    axpy_f32(lat.as<float>(), vel.as<float>(), sig[i + 1] - sig[i], N * C, c.stream);
  }
  report("decode", 0, 1);
  // [N, C] → [C, H, W] 並反正規化。
  std::vector<float> host(N * C), z(N * C);
  MX_CUDA(cudaMemcpy(host.data(), lat.get(), host.size() * 4, cudaMemcpyDeviceToHost));
  const float *mean = QwenImageVae::latents_mean(), *stdv = QwenImageVae::latents_std();
  for (int64_t n = 0; n < N; n++)
    for (int64_t ch = 0; ch < C; ch++) z[ch * N + n] = host[n * C + ch] * stdv[ch] + mean[ch];
  Ctx& vc = vae_ctx_ ? *vae_ctx_ : c;
  vc.use();
  std::vector<uint8_t> rgba;
  {
    DevBuf dz(z.size() * 4, vc.device);
    MX_CUDA(cudaMemcpy(dz.get(), z.data(), z.size() * 4, cudaMemcpyHostToDevice));
    vae_->decode(vc, dz.as<float>(), H, W, rgba);
  }
  ImageResult res;
  res.width = int(W * 16);
  res.height = int(H * 16);
  res.png = encode_png(rgba.data(), res.width, res.height, 4);
  res.seed = seed;
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report("done", 1, 1);
  return res;
}

}  // namespace mx
