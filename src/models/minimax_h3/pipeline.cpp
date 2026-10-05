#include "models/minimax_h3/pipeline.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

#include "server/image_io.h"
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
  vvae_ = std::make_unique<H3VideoVae>(vvae_st_, dev_.video_vae);
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
}

// 呈現方式（FL2VA）：每張關鍵幀前面是 "<Picture i>: " 標籤與視覺區塊（區塊列標成影片），之後是 prompt 原文；
// 無聊天範本、無特殊 token，取第 50 層輸出。
std::vector<bf16> H3Pipeline::encode_prompt(const std::string& prompt, const std::vector<Rgba>& keys,
                                            std::vector<H3Tag>& tags) {
  std::vector<int32_t> ids;
  tags.clear();
  std::vector<VisionFeatures> feats;
  const int32_t pad = tok_->token_id("<|image_pad|>"), vs = tok_->token_id("<|vision_start|>"),
                ve = tok_->token_id("<|vision_end|>");
  for (size_t i = 0; i < keys.size(); i++) {
    if (!vis_) vis_ = std::make_unique<Qwen3VisionEncoder>(te_st_, dev_.text);
    const Rgba& k = keys[i];
    size_t need = vis_->act_bytes(k.h, k.w);
    if (!vis_ctx_ || vis_ctx_->act.capacity() < need)
      vis_ctx_ = std::make_unique<Ctx>(dev_.text, need, vis_->scratch_bytes());
    auto rgb = drop_alpha(k);
    feats.push_back(vis_->encode(*vis_ctx_, rgb.data(), k.h, k.w));
    auto label = tok_->encode(strf("<Picture %zu>: ", i + 1));
    ids.insert(ids.end(), label.begin(), label.end());
    tags.insert(tags.end(), label.size(), H3_TEXT);
    ids.push_back(vs);
    ids.insert(ids.end(), size_t(feats.back().tokens()), pad);
    ids.push_back(ve);
    tags.insert(tags.end(), size_t(feats.back().tokens()) + 2, H3_VIDEO);
  }
  auto p = tok_->encode(prompt.empty() ? " " : prompt);
  ids.insert(ids.end(), p.begin(), p.end());
  tags.insert(tags.end(), p.size(), H3_TEXT);
  const int64_t T = int64_t(ids.size()), td = te_->dim();
  std::vector<bf16> txt(T * td);
  ArenaScope sc(te_ctx_->act);
  bf16* h = te_ctx_->act.alloc_n<bf16>(T * td);
  if (feats.empty()) {
    te_->encode(*te_ctx_, ids, h);
  } else {
    std::vector<const VisionFeatures*> fp;
    for (auto& f : feats) fp.push_back(&f);
    TextVisionInput tv = build_text_vision(ids, pad, fp);
    te_->encode(*te_ctx_, ids, h, &tv);
  }
  MX_CUDA(cudaMemcpy(txt.data(), h, txt.size() * 2, cudaMemcpyDeviceToHost));
  return txt;
}

VideoResult H3Pipeline::generate(const VideoRequest& r, const ProgressFn& cb, const std::atomic<bool>* cancel) {
  auto t0 = std::chrono::steady_clock::now();
  auto report = [&](const char* st, int i, int n) {
    if (cb) cb(Progress{st, i, n});
    if (cancel && cancel->load()) throw std::runtime_error("已取消");
  };
  apply_loras(r.loras);
  // 關鍵幀：0～2 張（首幀、尾幀）。沒指定畫布時依第一張的比例用官方畫布（短邊 768、面積上限 768×1344）。
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
  for (size_t i = 0; i < keys.size(); i++)
    keys[i] = i == 0 ? resize_lanczos(keys[i], cw, ch) : cover_crop(keys[i], cw, ch);
  const int frames = h3_align_frames(std::max(5, r.num_frames));
  const int64_t F_lat = h3_latent_frames(frames), H = ch / 16, W = cw / 16;
  const int64_t A = h3_audio_latents(frames);

  // 1. 文字（＋關鍵幀視覺特徵）。
  report("encode", 0, 1);
  std::vector<H3Tag> tags;
  std::vector<bf16> txt = encode_prompt(r.prompt, keys, tags);
  const int64_t T = int64_t(tags.size());

  // 2. 版面與雜訊（條件幀的雜訊先抽、再影片、後音訊，同官方順序）。
  H3Layout L = h3_build_layout(T, F_lat, H, W, A, anchors, tags);
  H3Seq& seq = L.seq;
  const int64_t C = 24, Cv = dit_->video_in(), Ca = dit_->audio_in();
  const int64_t Nc = seq.cond_len, Nv = seq.video_len, Na = seq.audio_len;
  int64_t seed = r.seed >= 0 ? r.seed : int64_t(std::random_device{}() & 0x7fffffff);
  std::vector<float> vrows((Nc + Nv) * Cv), arows(Na * Ca);
  auto patchify = [&](const std::vector<float>& lat, int64_t nf, float* dst) {  // [24, nf, H, W] → [nf·H/2·W/2, 96]
    for (int64_t f = 0; f < nf; f++)
      for (int64_t y = 0; y < H / 2; y++)
        for (int64_t x = 0; x < W / 2; x++) {
          int64_t row = (f * (H / 2) + y) * (W / 2) + x;
          for (int64_t c = 0; c < C; c++)
            for (int ph = 0; ph < 2; ph++)
              for (int pw = 0; pw < 2; pw++)
                dst[row * Cv + c * 4 + ph * 2 + pw] = lat[((c * nf + f) * H + 2 * y + ph) * W + 2 * x + pw];
        }
  };
  {
    std::mt19937_64 rng{uint64_t(seed)};
    std::normal_distribution<float> nd(0.f, 1.f);
    // 條件幀：VAE 編碼（分佈平均、正規化），再混入一點雜訊停在 t = 0.999（官方 keyframe_noise_aug）。
    if (!keys.empty()) {
      Ctx vc(dev_.video_vae, vvae_->decode_act_bytes(H, W), 0);
      DevBuf dz(size_t(C) * H * W * 4, vc.device);
      for (size_t i = 0; i < keys.size(); i++) {
        auto rgb = drop_alpha(keys[i]);
        vvae_->encode_image(vc, rgb.data(), ch, cw, dz.as<float>());
        std::vector<float> z(size_t(C) * H * W);
        MX_CUDA(cudaMemcpy(z.data(), dz.get(), z.size() * 4, cudaMemcpyDeviceToHost));
        for (auto& v : z) v = 0.999f * v + 0.001f * nd(rng);
        patchify(z, 1, vrows.data() + int64_t(i) * (H / 2) * (W / 2) * Cv);
      }
    }
    std::vector<float> lat(C * F_lat * H * W);
    for (auto& v : lat) v = nd(rng);
    patchify(lat, F_lat, vrows.data() + Nc * Cv);
    for (auto& v : arows) v = nd(rng);
  }

  // 3. DiT 取樣（影片 shift 12、音訊 shift 3，共用步數）。
  Ctx c(dev_.dit, dit_->act_bytes(seq.S), dit_->scratch_bytes());
  {
    DevBuf dt(txt.size() * 2, c.device);
    MX_CUDA(cudaMemcpy(dt.get(), txt.data(), txt.size() * 2, cudaMemcpyHostToDevice));
    dit_->prepare(c, seq, L.positions, dt.as<bf16>());
  }
  DevBuf dv(vrows.size() * 4, c.device), da(arows.size() * 4, c.device);
  DevBuf vv(vrows.size() * 4, c.device), va(arows.size() * 4, c.device);
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
    write_file((fs::path(dump) / "txt.bin").string(), reinterpret_cast<const uint8_t*>(txt.data()), txt.size() * 2);
    dump_dev("video0.bin", dv.get(), dv.size());
    dump_dev("audio0.bin", da.get(), da.size());
    std::string meta = strf("{\"T\":%lld,\"F_lat\":%lld,\"H\":%lld,\"W\":%lld,\"A\":%lld,\"steps\":%d}",
                            (long long)T, (long long)F_lat, (long long)H, (long long)W, (long long)A, r.steps);
    write_file((fs::path(dump) / "meta.json").string(), reinterpret_cast<const uint8_t*>(meta.data()), meta.size());
  }
  for (int i = 0; i < n; i++) {
    report("denoise", i, n);
    float tv = 1.f - sv[i], ta = 1.f - sa[i];
    dit_->forward(c, seq, dv.as<float>(), da.as<float>(), {tv, ta, std::max(tv, 0.999f)}, vv.as<float>(),
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
    axpy_f32(dv.as<float>() + Nc * Cv, vv.as<float>(), sv[i] - sv[i + 1], Nv * Cv, c.stream);
    axpy_f32(da.as<float>(), va.as<float>(), sa[i] - sa[i + 1], Na * Ca, c.stream);
  }
  MX_CUDA(cudaMemcpy(vrows.data(), dv.get(), vrows.size() * 4, cudaMemcpyDeviceToHost));
  MX_CUDA(cudaMemcpy(arows.data(), da.get(), arows.size() * 4, cudaMemcpyDeviceToHost));
  if (dump) {
    dump_dev("video_final.bin", dv.get(), dv.size());
    dump_dev("audio_final.bin", da.get(), da.size());
  }

  // 4. 音訊解碼 → WAV。rows：聲道優先 [2A, 32] → [2, 32, A]。
  report("decode_audio", 0, 1);
  fs::create_directories(paths_.out_dir);
  std::string stem = (fs::path(paths_.out_dir) / strf("h3_%lld", (long long)seed)).string();
  {
    std::vector<float> az(kAudioChannels * Ca * A);
    for (int64_t ch = 0; ch < kAudioChannels; ch++)
      for (int64_t a = 0; a < A; a++)
        for (int64_t k = 0; k < Ca; k++) az[(ch * Ca + k) * A + a] = arows[(ch * A + a) * Ca + k];
    Ctx& ac = *avae_ctx_;
    ac.use();
    DevBuf dz(az.size() * 4, ac.device);
    MX_CUDA(cudaMemcpy(dz.get(), az.data(), az.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> pcm;
    avae_->decode(ac, dz.as<float>(), A, pcm);
    write_wav(stem + ".wav", pcm, H3AudioVae::kSampleRate);
  }

  // 5. 影片解碼：邊解邊送 ffmpeg（rawvideo rgb24 + wav → H.264/AAC）。
  report("decode_video", 0, 1);
  std::vector<float> z(C * F_lat * H * W);
  for (int64_t f = 0; f < F_lat; f++)
    for (int64_t y = 0; y < H / 2; y++)
      for (int64_t x = 0; x < W / 2; x++) {
        int64_t row = (f * (H / 2) + y) * (W / 2) + x;
        for (int64_t ch = 0; ch < C; ch++)
          for (int ph = 0; ph < 2; ph++)
            for (int pw = 0; pw < 2; pw++)
              z[((ch * F_lat + f) * H + 2 * y + ph) * W + 2 * x + pw] = vrows[(Nc + row) * Cv + ch * 4 + ph * 2 + pw];
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
      report("decode_video", written, frames);
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
  VideoResult res;
  res.mp4_path = mp4;
  res.seed = seed;
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report("done", 1, 1);
  return res;
}

}  // namespace mx
