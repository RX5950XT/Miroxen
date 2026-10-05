#include "models/minimax_h3/pipeline.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

#include "server/image_io.h"
#include "server/media_io.h"
#include "kernels/ops.h"

namespace mx {
namespace fs = std::filesystem;

namespace {
constexpr int kAudioChannels = 2;

int pick_second(int requested) {
  if (requested >= 0) return requested;
  int n = 0;
  MX_CUDA(cudaGetDeviceCount(&n));
  return n > 1 ? 1 : 0;
}

void write_wav(const std::string& path, const std::vector<float>& pcm, int rate) {
  std::vector<uint8_t> d;
  auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) d.push_back(uint8_t(v >> (8 * i))); };
  auto u16 = [&](uint16_t v) { d.push_back(uint8_t(v)); d.push_back(uint8_t(v >> 8)); };
  uint32_t bytes = uint32_t(pcm.size() * 2);
  d.insert(d.end(), {'R', 'I', 'F', 'F'});
  u32(36 + bytes);
  d.insert(d.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  u32(16), u16(1), u16(2), u32(rate), u32(rate * 4), u16(4), u16(16);
  d.insert(d.end(), {'d', 'a', 't', 'a'});
  u32(bytes);
  for (float s : pcm) {
    int v = int(std::lround(std::clamp(s, -1.f, 1.f) * 32767.f));
    u16(uint16_t(int16_t(v)));
  }
  write_file(path, d.data(), d.size());
}
}  // namespace

H3Pipeline::H3Pipeline(const H3Paths& p, H3Devices d) : paths_(p), dev_(d) {
  dev_.text = pick_second(d.text);
  dev_.audio_vae = pick_second(d.audio_vae);
  tok_ = Tokenizer::load(p.tokenizer);
  te_st_.add_file(p.text_encoder, "\x01");
  dit_st_.add_file(p.dit);
  vvae_st_.add_file(p.video_vae);
  avae_st_.add_file(p.audio_vae);
  te_ = std::make_unique<Qwen3TextEncoder>(te_st_, dev_.text, /*stream=*/true);
  te_ctx_ = std::make_unique<Ctx>(dev_.text, size_t(512) << 20, te_->scratch_bytes());
  dit_ = std::make_unique<H3Dit>(dit_st_, dev_.dit, 0, d.compute);
  vvae_ = std::make_unique<H3VideoVae>(vvae_st_, dev_.video_vae, d.compute);
  avae_ = std::make_unique<H3AudioVae>(avae_st_, dev_.audio_vae);
  avae_ctx_ = std::make_unique<Ctx>(dev_.audio_vae, size_t(1) << 30, 0);
  MX_INFO("MiniMax-H3 管線就緒：DiT@GPU%d（串流）、文字編碼器@GPU%d（串流）、影片 VAE@GPU%d、音訊 VAE@GPU%d",
          dev_.dit, dev_.text, dev_.video_vae, dev_.audio_vae);
}

H3Pipeline::~H3Pipeline() = default;

std::string H3Pipeline::describe() const {
  return strf("DiT %s/%s @GPU%d（串流）· TE %d 層 @GPU%d（串流）",
              gtype_name(dit_st_.get("blocks.0.attn.qkv_proj.weight").type), qmode_name(dit_->compute()), dev_.dit,
              te_->layers(), dev_.text);
}

void H3Pipeline::ensure_dit(bool ref) {
  if (ref == ref_active_ && dit_) return;
  if (ref) {
    MX_CHECK(!paths_.dit_ref.empty(), "Ref2VA 需要 Ref2VA checkpoint（例如 minimax_h3_ref2va_pruned-Q4_K.gguf）");
    if (dit_ref_st_.size() == 0) dit_ref_st_.add_file(paths_.dit_ref);
  }
  QMode q = dit_ ? dit_->compute() : dev_.compute;
  dit_.reset();  // 兩個 DiT 不同時載入
  active_loras_.clear();
  dit_ = std::make_unique<H3Dit>(ref ? dit_ref_st_ : dit_st_, dev_.dit, 0, q);
  ref_active_ = ref;
}

void H3Pipeline::apply_loras(const std::vector<LoraSpec>& loras) {
  bool same = loras.size() == active_loras_.size();
  for (size_t i = 0; same && i < loras.size(); i++)
    same = loras[i].path == active_loras_[i].path && loras[i].scale == active_loras_[i].scale;
  if (same) return;
  auto ls = dit_->linears();
  clear_loras(ls);
  active_loras_.clear();
  auto names = dit_->linear_names();
  for (auto& l : loras) {
    auto st = apply_lora(l.path, l.scale, dev_.dit, names,
                         [&](const std::string& n, LoraTarget& t) { return dit_->resolve_lora(n, t); });
    MX_CHECK(st.applied > 0, "LoRA %s 沒有任何層對應到 MiniMax-H3", l.path.c_str());
    active_loras_.push_back(l);
  }
  dit_->invalidate_streams();
}

VisionFeatures H3Pipeline::vision(const uint8_t* rgb0, const uint8_t* rgb1, int h, int w) {
  if (!vis_) vis_ = std::make_unique<Qwen3VisionEncoder>(te_st_, dev_.text);
  size_t need = vis_->act_bytes(h, w);
  if (!vis_ctx_ || vis_ctx_->act.capacity() < need) {
    vis_ctx_.reset();
    vis_ctx_ = std::make_unique<Ctx>(dev_.text, need, vis_->scratch_bytes());
  }
  return vis_->encode_pair(*vis_ctx_, rgb0, rgb1, h, w);
}

// 呈現方式：pieces 依序（標籤文字、視覺區塊），最後是 prompt 原文；無聊天範本、無特殊 token，各段分開 tokenize，
// 取第 50 層輸出。
std::vector<bf16> H3Pipeline::encode_prompt(const std::vector<Piece>& pieces, const std::string& prompt,
                                            std::vector<H3Tag>& tags) {
  std::vector<int32_t> ids;
  tags.clear();
  const int32_t ipad = tok_->token_id("<|image_pad|>"), vpad = tok_->token_id("<|video_pad|>"),
                vs = tok_->token_id("<|vision_start|>"), ve = tok_->token_id("<|vision_end|>");
  std::vector<const VisionFeatures*> feats;
  auto text = [&](const std::string& s) {
    auto t = tok_->encode(s);
    ids.insert(ids.end(), t.begin(), t.end());
    tags.insert(tags.end(), t.size(), H3_TEXT);
  };
  for (auto& p : pieces) {
    if (!p.text.empty()) text(p.text);
    if (!p.vision) continue;
    ids.push_back(vs);
    ids.insert(ids.end(), size_t(p.vision->tokens()), p.video ? vpad : ipad);
    ids.push_back(ve);
    tags.insert(tags.end(), size_t(p.vision->tokens()) + 2, H3_VIDEO);
    feats.push_back(p.vision);
  }
  text(prompt.empty() ? " " : prompt);
  const int64_t T = int64_t(ids.size()), td = te_->dim();
  if (te_ctx_->act.capacity() < te_->act_bytes(T)) {
    te_ctx_.reset();
    te_ctx_ = std::make_unique<Ctx>(dev_.text, te_->act_bytes(T), te_->scratch_bytes());
  }
  std::vector<bf16> txt(T * td);
  ArenaScope sc(te_ctx_->act);
  bf16* h = te_ctx_->act.alloc_n<bf16>(T * td);
  if (feats.empty()) {
    te_->encode(*te_ctx_, ids, h);
  } else {
    TextVisionInput tv = build_text_vision(ids, {ipad, vpad}, feats);
    te_->encode(*te_ctx_, ids, h, &tv);
  }
  MX_CUDA(cudaMemcpy(txt.data(), h, txt.size() * 2, cudaMemcpyDeviceToHost));
  return txt;
}

namespace {

// [24, nf, H, W] latent → [nf·(H/2)·(W/2), 96] 的列（1×2×2 patch）。
void patchify(const std::vector<float>& lat, int64_t nf, int64_t H, int64_t W, float* dst) {
  for (int64_t f = 0; f < nf; f++)
    for (int64_t y = 0; y < H / 2; y++)
      for (int64_t x = 0; x < W / 2; x++) {
        int64_t row = (f * (H / 2) + y) * (W / 2) + x;
        for (int64_t c = 0; c < 24; c++)
          for (int ph = 0; ph < 2; ph++)
            for (int pw = 0; pw < 2; pw++)
              dst[row * 96 + c * 4 + ph * 2 + pw] = lat[((c * nf + f) * H + 2 * y + ph) * W + 2 * x + pw];
      }
}

// 視覺條件 latent 停在 t = 0.999（官方 keyframe_noise_aug）：x ← 0.999·x + 0.001·雜訊。
void noise_aug(std::vector<float>& z, std::mt19937_64& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : z) v = 0.999f * v + 0.001f * nd(rng);
}

// Python 的 "{:.1f}"（剛好落在 .x5 的值四捨六入五成雙），用在影片區塊的 "<0.2 seconds>" 標籤。
std::string fmt_tenths(double v) {
  double t = v * 10.0, f = std::floor(t);
  int64_t n = int64_t(f);
  if (t - f > 0.5 || (t - f == 0.5 && (n & 1))) n++;
  return strf("%lld.%lld", (long long)(n / 10), (long long)(n % 10));
}

// Qwen3-VL 影片前處理的 smart_resize：所有幀的總像素超過上限時等比縮小（factor 32）。
void video_smart_resize(int frames, int h, int w, int& oh, int& ow) {
  const double max_px = 25165824.0, min_px = 4096.0, f = 32.0;
  oh = int(std::nearbyint(h / f) * f), ow = int(std::nearbyint(w / f) * f);
  const int tb = int(std::nearbyint(frames / 2.0)) * 2;
  if (double(tb) * oh * ow > max_px) {
    double beta = std::sqrt(double(frames) * h * w / max_px);
    oh = std::max(32, int(std::floor(h / beta / f)) * 32), ow = std::max(32, int(std::floor(w / beta / f)) * 32);
  } else if (double(tb) * oh * ow < min_px) {
    double beta = std::sqrt(min_px / (double(frames) * h * w));
    oh = int(std::ceil(h * beta / f)) * 32, ow = int(std::ceil(w * beta / f)) * 32;
  }
}

// 音訊 VAE 編碼 → 聲道優先的列 [2A, 32]（分佈平均、正規化，不加雜訊）。
std::vector<float> encode_audio_rows(H3AudioVae& vae, Ctx& c, const std::vector<float>& pcm, int64_t& A) {
  const int64_t samples = int64_t(pcm.size() / 2);
  A = (samples + 799) / 800;
  DevBuf dz(size_t(2) * 32 * A * 4, c.device);
  int64_t got = 0;
  vae.encode(c, pcm.data(), samples, dz.as<float>(), got);
  MX_CHECK(got == A, "音訊 latent 長度不符");
  std::vector<float> z(size_t(2) * 32 * A), rows(z.size());
  MX_CUDA(cudaMemcpy(z.data(), dz.get(), z.size() * 4, cudaMemcpyDeviceToHost));
  for (int64_t ch = 0; ch < 2; ch++)
    for (int64_t a = 0; a < A; a++)
      for (int64_t k = 0; k < 32; k++) rows[(ch * A + a) * 32 + k] = z[(ch * 32 + k) * A + a];
  return rows;
}

}  // namespace

// FL2VA：0～2 張關鍵幀（首幀、尾幀）。沒指定畫布時依第一張的比例用官方畫布（短邊 768、面積上限 768×1344）。
H3Pipeline::Prepared H3Pipeline::prepare_fl2va(const VideoRequest& r, const Report& report) {
  std::vector<Rgba> keys;
  std::vector<std::string> anchors;
  if (!r.first_frame.empty()) keys.push_back(load_rgba(r.first_frame)), anchors.push_back("first");
  if (!r.last_frame.empty()) keys.push_back(load_rgba(r.last_frame)), anchors.push_back("last");
  int cw = r.width, ch = r.height;
  if (cw <= 0 || ch <= 0) {
    if (keys.empty())
      h3_canvas(16, 9, 768, 768 * 1344, cw, ch);
    else
      h3_canvas(keys[0].w, keys[0].h, 768, 768 * 1344, cw, ch);
  }
  MX_CHECK(cw % 32 == 0 && ch % 32 == 0, "寬高必須是 32 的倍數");
  // 第一張拉伸到畫布；第二張（跟隨者）等比放大蓋滿後置中裁切。
  std::vector<std::vector<uint8_t>> rgbs;
  for (size_t i = 0; i < keys.size(); i++)
    rgbs.push_back(drop_alpha(i == 0 ? resize_lanczos(keys[i], cw, ch) : cover_crop(keys[i], cw, ch)));
  Prepared p;
  p.frames = h3_align_frames(std::max(5, r.num_frames));
  const int64_t F_lat = h3_latent_frames(p.frames), H = ch / 16, W = cw / 16, A = h3_audio_latents(p.frames);

  report("encode", 0, 1);
  std::vector<VisionFeatures> feats;
  for (auto& rgb : rgbs) feats.push_back(vision(rgb.data(), rgb.data(), ch, cw));
  std::vector<Piece> pieces;
  for (size_t i = 0; i < feats.size(); i++) pieces.push_back({strf("<Picture %zu>: ", i + 1), &feats[i]});
  std::vector<H3Tag> tags;
  p.txt = encode_prompt(pieces, r.prompt, tags);
  p.L = h3_build_layout(int64_t(tags.size()), F_lat, H, W, A, anchors, tags);

  // 條件幀：VAE 編碼（分佈平均、正規化）＋雜訊增強；雜訊先於影片／音訊雜訊抽取（官方順序）。
  p.seed = r.seed >= 0 ? r.seed : int64_t(std::random_device{}() & 0x7fffffff);
  p.rng.seed(uint64_t(p.seed));
  if (!rgbs.empty()) {
    Ctx vc(dev_.video_vae, vvae_->decode_act_bytes(H, W), 0);
    DevBuf dz(size_t(24) * H * W * 4, vc.device);
    const int64_t rpf = (H / 2) * (W / 2);
    p.vref.resize(rgbs.size() * rpf * 96);
    for (size_t i = 0; i < rgbs.size(); i++) {
      vvae_->encode_image(vc, rgbs[i].data(), ch, cw, dz.as<float>());
      std::vector<float> z(size_t(24) * H * W);
      MX_CUDA(cudaMemcpy(z.data(), dz.get(), z.size() * 4, cudaMemcpyDeviceToHost));
      noise_aug(z, p.rng);
      patchify(z, 1, H, W, p.vref.data() + int64_t(i) * rpf * 96);
    }
  }
  return p;
}

// Ref2VA：參考依請求順序；畫布不隨參考改變（預設官方 16:9、短邊 768）。
H3Pipeline::Prepared H3Pipeline::prepare_ref2va(const VideoRequest& r, const Report& report) {
  MX_CHECK(r.first_frame.empty() && r.last_frame.empty(), "Ref2VA 不能同時給首尾幀");
  int nimg = 0, nvid = 0, naud = 0;
  for (auto& m : r.references) {
    MX_CHECK(m.kind == "image" || m.kind == "video" || m.kind == "audio", "參考種類必須是 image／video／audio");
    (m.kind == "image" ? nimg : m.kind == "video" ? nvid : naud)++;
  }
  MX_CHECK(nimg <= 9 && nvid <= 3 && naud <= 3 && nimg + nvid + naud <= 12, "參考最多 9 張圖、3 段影片、3 段音訊，共 12 個");
  MX_CHECK(nimg + nvid > 0, "音訊參考必須搭配至少一張圖或一段影片");
  int cw = r.width, ch = r.height;
  if (cw <= 0 || ch <= 0) h3_canvas(16, 9, 768, 768 * 1344, cw, ch);
  MX_CHECK(cw % 32 == 0 && ch % 32 == 0, "寬高必須是 32 的倍數");
  Prepared p;
  p.frames = h3_align_frames(std::max(5, r.num_frames));
  const int64_t F_lat = h3_latent_frames(p.frames), H = ch / 16, W = cw / 16, A = h3_audio_latents(p.frames);
  const double seconds = p.frames / 24.0;
  p.seed = r.seed >= 0 ? r.seed : int64_t(std::random_device{}() & 0x7fffffff);
  p.rng.seed(uint64_t(p.seed));

  report("encode", 0, 1);
  const auto tp = std::chrono::steady_clock::now();
  auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count(); };
  // 1. 讀媒體、算視覺特徵與 latent（影片列加雜訊增強，音訊列乾淨）。
  struct Ref {
    H3RefGeom geom;
    std::vector<VisionFeatures> feats;  // 圖：1 個；影片：每兩幀一個時間 patch
    std::vector<std::string> stamps;    // 影片每個 patch 的時間標籤
  };
  std::vector<Ref> refs;
  std::vector<std::vector<float>> vlat, alat;  // 依序
  std::unique_ptr<Ctx> vc;
  auto vae_ctx = [&]() -> Ctx& {
    if (!vc) vc = std::make_unique<Ctx>(dev_.video_vae, size_t(3) << 30, 0);
    return *vc;
  };
  for (auto& m : r.references) {
    Ref ref;
    if (m.kind == "image") {
      Rgba img = load_rgba(m.path);
      MX_CHECK(img.w <= 4 * img.h && img.h <= 4 * img.w, "參考圖長寬比必須在 1:4 到 4:1（%s）", m.path.c_str());
      // 縮放到自己的短邊（官方 2048，可放大、無面積上限），兩軸取最接近的 32 倍數。
      const double s = double(r.ref_image_short_edge) / std::min(img.w, img.h);
      const int ih = std::max(32, int(std::nearbyint(img.h * s / 32)) * 32);
      const int iw = std::max(32, int(std::nearbyint(img.w * s / 32)) * 32);
      auto rgb = drop_alpha(resize_lanczos(img, iw, ih));
      ref.feats.push_back(vision(rgb.data(), rgb.data(), ih, iw));
      ref.geom = {H3RefGeom::IMAGE, 1, ih / 16, iw / 16, 0};
      Ctx& c = vae_ctx();
      DevBuf dz(size_t(24) * (ih / 16) * (iw / 16) * 4, c.device);
      vvae_->encode_image(c, rgb.data(), ih, iw, dz.as<float>());
      std::vector<float> z(size_t(24) * (ih / 16) * (iw / 16));
      MX_CUDA(cudaMemcpy(z.data(), dz.get(), z.size() * 4, cudaMemcpyDeviceToHost));
      vlat.push_back(std::move(z));
    } else if (m.kind == "video") {
      MediaInfo mi = probe_media(m.path);
      MX_CHECK(mi.has_video && mi.width > 0 && mi.height > 0, "%s 沒有影像軌", m.path.c_str());
      // 參考影片放到它自己比例的畫布：官方是短邊 768；預設不超過輸出畫布（序列短很多，DiT 快數倍）。
      int vw, vh;
      if (r.ref_native_canvas)
        h3_canvas(mi.width, mi.height, 768, 768 * 1344, vw, vh);
      else
        h3_canvas(mi.width, mi.height, std::min({768, cw, ch}), std::min<int64_t>(768 * 1344, int64_t(cw) * ch), vw, vh);
      int nf = 0;
      auto frames = decode_video_rgb(m.path, vw, vh, 24.0, p.frames, nf);
      const size_t fsz = size_t(vw) * vh * 3;
      // 給文字編碼器：每秒 2 幀（每 12 幀取 1），補到偶數，每兩幀一個時間 patch，標籤是兩幀時間的平均。
      std::vector<int> idx;
      for (double cur = 0; std::nearbyint(cur) < nf; cur += 12.0)
        if (idx.empty() || int(std::nearbyint(cur)) > idx.back()) idx.push_back(int(std::nearbyint(cur)));
      MX_CHECK(idx.size() >= 2, "參考影片太短（至少 13 幀，約 0.55 秒）：%s", m.path.c_str());
      std::vector<double> ts;
      for (size_t i = 0; i < idx.size(); i++) ts.push_back(double(i) / 2.0);
      if (idx.size() % 2) idx.push_back(idx.back()), ts.push_back(ts.back());
      int sh, sw;
      video_smart_resize(int(idx.size()), vh, vw, sh, sw);
      for (size_t i = 0; i < idx.size(); i += 2) {
        auto pick = [&](int k) {
          Rgba f;
          f.w = vw, f.h = vh, f.px.resize(size_t(vw) * vh * 4);
          for (size_t q = 0; q < size_t(vw) * vh; q++) {
            std::copy_n(&frames[size_t(k) * fsz + q * 3], 3, &f.px[q * 4]);
            f.px[q * 4 + 3] = 255;
          }
          return drop_alpha(sh == vh && sw == vw ? f : resize_lanczos(f, sw, sh));
        };
        auto a = pick(idx[i]), b = pick(idx[i + 1]);
        ref.feats.push_back(vision(a.data(), b.data(), sh, sw));
        ref.stamps.push_back("<" + fmt_tenths((ts[i] + ts[i + 1]) / 2.0) + " seconds>");
      }
      // 給 VAE：往下取 17n+5 幀（不足時就用全部，編碼時補最後一幀）。
      const int nv = std::min(nf, std::max(1, (nf - 5) / 17) * 17 + 5);
      const int64_t fl = H3VideoVae::encoded_frames(nv);
      Ctx& c = vae_ctx();
      DevBuf dz(size_t(24) * fl * (vh / 16) * (vw / 16) * 4, c.device);
      vvae_->encode_video(c, frames.data(), nv, vh, vw, dz.as<float>());
      std::vector<float> z(size_t(24) * fl * (vh / 16) * (vw / 16));
      MX_CUDA(cudaMemcpy(z.data(), dz.get(), z.size() * 4, cudaMemcpyDeviceToHost));
      vlat.push_back(std::move(z));
      ref.geom = {H3RefGeom::VIDEO, fl, vh / 16, vw / 16, 0};
      if (mi.has_audio) {
        auto pcm = decode_audio(m.path, H3AudioVae::kSampleRate, seconds);
        alat.push_back(encode_audio_rows(*avae_, *avae_ctx_, pcm, ref.geom.audio));
      }
    } else {
      auto pcm = decode_audio(m.path, H3AudioVae::kSampleRate, seconds);
      ref.geom = {H3RefGeom::AUDIO, 0, 0, 0, 0};
      alat.push_back(encode_audio_rows(*avae_, *avae_ctx_, pcm, ref.geom.audio));
    }
    refs.push_back(std::move(ref));
  }
  vc.reset();
  MX_INFO("Ref2VA 參考編碼（解碼媒體、視覺塔、VAE）：%.1f 秒", since());

  // 2. 呈現：每個參考一個標籤（依模態各自編號；有音軌的影片先標 <Audio j>），影片每個時間 patch 前有時間標籤。
  std::vector<Piece> pieces;
  int ci = 0, cv = 0, ca = 0;
  for (size_t i = 0; i < refs.size(); i++) {
    const auto& m = r.references[i];
    Ref& ref = refs[i];
    if (ref.geom.audio > 0) pieces.push_back({strf("<Audio %d>: ", ++ca)});
    if (m.kind == "image") pieces.push_back({strf("<Picture %d>: ", ++ci), &ref.feats[0]});
    if (m.kind == "video") {
      pieces.push_back({strf("<Video %d>: ", ++cv)});
      for (size_t b = 0; b < ref.feats.size(); b++) pieces.push_back({ref.stamps[b], &ref.feats[b], true});
    }
  }
  std::vector<H3Tag> tags;
  p.txt = encode_prompt(pieces, r.prompt, tags);
  MX_INFO("Ref2VA 文字編碼：%zu token，累計 %.1f 秒", tags.size(), since());
  std::vector<H3RefGeom> geoms;
  for (auto& ref : refs) geoms.push_back(ref.geom);
  p.L = h3_build_ref2va_layout(int64_t(tags.size()), tags, geoms, F_lat, H, W, A);

  // 3. 參考列：視覺 latent 加雜訊增強後 patchify（依序，雜訊先於目標雜訊抽取），音訊列直接串接。
  for (size_t i = 0, vi = 0; i < refs.size(); i++) {
    const H3RefGeom& g = refs[i].geom;
    if (g.kind == H3RefGeom::AUDIO) continue;
    noise_aug(vlat[vi], p.rng);
    const size_t off = p.vref.size();
    p.vref.resize(off + size_t(g.frames) * (g.h / 2) * (g.w / 2) * 96);
    patchify(vlat[vi], g.frames, g.h, g.w, p.vref.data() + off);
    vi++;
  }
  for (auto& a : alat) p.aref.insert(p.aref.end(), a.begin(), a.end());
  MX_CHECK(int64_t(p.vref.size()) == p.L.seq.vref_len() * 96 && int64_t(p.aref.size()) == p.L.seq.aref_len() * 32,
           "參考列數與版面不符");
  return p;
}

VideoResult H3Pipeline::generate(const VideoRequest& r, const ProgressFn& cb, const std::atomic<bool>* cancel) {
  auto t0 = std::chrono::steady_clock::now();
  Report report = [&](const char* st, int i, int n) {
    if (cb) cb(Progress{st, i, n});
    if (cancel && cancel->load()) throw std::runtime_error("已取消");
  };
  const bool ref = !r.references.empty();
  if (ref != ref_active_) report("load", 0, 1);
  ensure_dit(ref);
  apply_loras(r.loras);
  Prepared p = ref ? prepare_ref2va(r, report) : prepare_fl2va(r, report);
  MX_INFO("H3 輸入準備（媒體／視覺塔／VAE 編碼／文字編碼）：%.1f 秒，序列 %lld 列（文字 %lld）",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (long long)p.L.seq.S,
          (long long)p.L.T);
  VideoResult res = sample_and_decode(r, p, report);
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report("done", 1, 1);
  return res;
}

VideoResult H3Pipeline::sample_and_decode(const VideoRequest& r, Prepared& p, const Report& report) {
  H3Layout& L = p.L;
  H3Seq& seq = L.seq;
  const int64_t T = L.T, F_lat = L.F_lat, H = L.H, W = L.W, A = L.A;
  const int64_t C = 24, Cv = dit_->video_in(), Ca = dit_->audio_in();
  const int64_t Nr = seq.vref_len(), Nar = seq.aref_len(), Nv = seq.video_len, Na = seq.audio_len;
  // 目標雜訊：先影片、後音訊（接在參考雜訊之後）。
  std::vector<float> vrows((Nr + Nv) * Cv), arows((Nar + Na) * Ca);
  std::copy(p.vref.begin(), p.vref.end(), vrows.begin());
  std::copy(p.aref.begin(), p.aref.end(), arows.begin());
  {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> lat(C * F_lat * H * W);
    for (auto& v : lat) v = nd(p.rng);
    patchify(lat, F_lat, H, W, vrows.data() + Nr * Cv);
    for (int64_t i = Nar * Ca; i < int64_t(arows.size()); i++) arows[i] = nd(p.rng);
  }

  // DiT 取樣（影片 shift 預設 12、音訊 3，共用步數）。timesteps：0 影片、1 音訊、2 視覺參考、3 音訊參考。
  const auto ts = std::chrono::steady_clock::now();
  Ctx c(dev_.dit, dit_->act_bytes(seq.S), dit_->scratch_bytes());
  {
    DevBuf dt(p.txt.size() * 2, c.device);
    MX_CUDA(cudaMemcpy(dt.get(), p.txt.data(), p.txt.size() * 2, cudaMemcpyHostToDevice));
    dit_->prepare(c, seq, L.positions, dt.as<bf16>());
  }
  DevBuf dv(vrows.size() * 4, c.device), da(arows.size() * 4, c.device);
  DevBuf vv(Nv * Cv * 4, c.device), va(Na * Ca * 4, c.device);
  MX_CUDA(cudaMemcpy(dv.get(), vrows.data(), vrows.size() * 4, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(da.get(), arows.data(), arows.size() * 4, cudaMemcpyHostToDevice));
  auto sv = h3_sigmas(r.steps, r.video_shift), sa = h3_sigmas(r.steps, 3.f);
  const int n = int(std::min(sv.size(), sa.size())) - 1;
  // 除錯：MX_H3_DUMP=目錄 時存下文字嵌入、初始雜訊、第一步與最終輸出（給 tools/ref/h3_dit_ref.py 比對）。
  const char* dump = std::getenv("MX_H3_DUMP");
  auto dump_dev = [&](const char* name, const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    MX_CUDA(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    write_file((fs::path(dump) / name).string(), h.data(), h.size());
  };
  if (dump) {
    fs::create_directories(dump);
    write_file((fs::path(dump) / "txt.bin").string(), reinterpret_cast<const uint8_t*>(p.txt.data()), p.txt.size() * 2);
    dump_dev("video0.bin", dv.get(), dv.size());
    dump_dev("audio0.bin", da.get(), da.size());
    std::string meta = strf("{\"T\":%lld,\"F_lat\":%lld,\"H\":%lld,\"W\":%lld,\"A\":%lld,\"steps\":%d}",
                            (long long)T, (long long)F_lat, (long long)H, (long long)W, (long long)A, r.steps);
    write_file((fs::path(dump) / "meta.json").string(), reinterpret_cast<const uint8_t*>(meta.data()), meta.size());
  }
  for (int i = 0; i < n; i++) {
    report("denoise", i, n);
    float tv = 1.f - sv[i], ta = 1.f - sa[i];
    dit_->forward(c, seq, dv.as<float>(), da.as<float>(), {tv, ta, std::max(tv, 0.999f), 1.f}, vv.as<float>(),
                  va.as<float>());
    if (dump && i == 0) {
      dump_dev("v0_video.bin", vv.get(), vv.size());
      dump_dev("v0_audio.bin", va.get(), va.size());
    }
    if (dump && std::getenv("MX_H3_DUMP_ALL")) {  // 每一步的輸入 latent 與速度
      dump_dev(strf("x%d_video.bin", i).c_str(), dv.get(), dv.size());
      dump_dev(strf("x%d_audio.bin", i).c_str(), da.get(), da.size());
      dump_dev(strf("v%d_video.bin", i).c_str(), vv.get(), vv.size());
      dump_dev(strf("v%d_audio.bin", i).c_str(), va.get(), va.size());
    }
    // 只更新生成列；參考／條件列整個取樣過程都保持原樣。
    axpy_f32(dv.as<float>() + Nr * Cv, vv.as<float>(), sv[i] - sv[i + 1], Nv * Cv, c.stream);
    axpy_f32(da.as<float>() + Nar * Ca, va.as<float>(), sa[i] - sa[i + 1], Na * Ca, c.stream);
  }
  MX_CUDA(cudaMemcpy(vrows.data(), dv.get(), vrows.size() * 4, cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(arows.data(), da.get(), arows.size() * 4, cudaMemcpyDeviceToHost));
  MX_INFO("H3 DiT 取樣：%d 次前向 %.1f 秒", n, std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
  if (dump) {
    dump_dev("video_final.bin", dv.get(), dv.size());
    dump_dev("audio_final.bin", da.get(), da.size());
  }

  // 音訊解碼 → WAV。rows：聲道優先 [2A, 32] → [2, 32, A]。
  report("decode_audio", 0, 1);
  const auto ta0 = std::chrono::steady_clock::now();
  fs::create_directories(paths_.out_dir);
  std::string stem = (fs::path(paths_.out_dir) / strf("h3_%lld", (long long)p.seed)).string();
  {
    std::vector<float> az(kAudioChannels * Ca * A);
    const float* ar = arows.data() + Nar * Ca;
    for (int64_t ch = 0; ch < kAudioChannels; ch++)
      for (int64_t a = 0; a < A; a++)
        for (int64_t k = 0; k < Ca; k++) az[(ch * Ca + k) * A + a] = ar[(ch * A + a) * Ca + k];
    Ctx& ac = *avae_ctx_;
    ac.use();
    DevBuf dz(az.size() * 4, ac.device);
    MX_CUDA(cudaMemcpy(dz.get(), az.data(), az.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> pcm;
    avae_->decode(ac, dz.as<float>(), A, pcm);
    write_wav(stem + ".wav", pcm, H3AudioVae::kSampleRate);
  }
  const auto td = std::chrono::steady_clock::now();
  MX_INFO("H3 音訊解碼：%.1f 秒", std::chrono::duration<double>(td - ta0).count());

  // 影片解碼：邊解邊送 ffmpeg（rawvideo rgb24 + wav → H.264/AAC）。
  report("decode_video", 0, 1);
  std::vector<float> z(C * F_lat * H * W);
  for (int64_t f = 0; f < F_lat; f++)
    for (int64_t y = 0; y < H / 2; y++)
      for (int64_t x = 0; x < W / 2; x++) {
        int64_t row = (f * (H / 2) + y) * (W / 2) + x;
        for (int64_t ch = 0; ch < C; ch++)
          for (int ph = 0; ph < 2; ph++)
            for (int pw = 0; pw < 2; pw++)
              z[((ch * F_lat + f) * H + 2 * y + ph) * W + 2 * x + pw] = vrows[(Nr + row) * Cv + ch * 4 + ph * 2 + pw];
      }
  std::string mp4 = stem + ".mp4";
  std::string cmd = strf("ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %lldx%lld -r 24 -i - -i \"%s.wav\" "
                         "-c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -af apad -shortest \"%s\"",
                         (long long)(W * 16), (long long)(H * 16), stem.c_str(), mp4.c_str());
#ifdef _WIN32
  FILE* ff = _popen(cmd.c_str(), "wb");
#else
  FILE* ff = popen(cmd.c_str(), "w");
#endif
  MX_CHECK(ff, "無法啟動 ffmpeg（請確認已安裝並在 PATH 中）");
  int written = 0;
  try {
    Ctx vc(dev_.video_vae, vvae_->decode_act_bytes(H, W), 0);
    DevBuf dz(z.size() * 4, vc.device);
    MX_CUDA(cudaMemcpy(dz.get(), z.data(), z.size() * 4, cudaMemcpyHostToDevice));
    vvae_->decode(vc, dz.as<float>(), F_lat, H, W, [&](const uint8_t* rgb, int nf) {
      size_t bytes = size_t(nf) * (W * 16) * (H * 16) * 3;
      MX_CHECK(std::fwrite(rgb, 1, bytes, ff) == bytes, "寫入 ffmpeg 失敗");
      written += nf;
      report("decode_video", std::min(written, p.frames), p.frames);
    });
  } catch (...) {
#ifdef _WIN32
    _pclose(ff);
#else
    pclose(ff);
#endif
    throw;
  }
#ifdef _WIN32
  int rc = _pclose(ff);
#else
  int rc = pclose(ff);
#endif
  MX_CHECK(rc == 0, "ffmpeg 結束代碼 %d", rc);
  MX_INFO("H3 影片解碼＋編碼 MP4：%.1f 秒",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count());
  VideoResult res;
  res.mp4_path = mp4;
  res.seed = p.seed;
  return res;
}

}  // namespace mx
