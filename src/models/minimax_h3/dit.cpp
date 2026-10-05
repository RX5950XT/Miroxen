#include "models/minimax_h3/dit.h"

#include <algorithm>
#include <cmath>

#include "core/dequant.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

namespace mx {

namespace {
constexpr int64_t kChunk = 8192;  // qkv / MLP 分段列數，限制暫存大小

std::vector<float> host_f32(const WeightRef& r) {
  std::vector<float> v(r.numel());
  dequant_cpu(r.data, r.type, r.numel(), v.data());
  return v;
}
}  // namespace

std::vector<float> h3_sigmas(int steps, float shift) {
  MX_CHECK(steps >= 2, "MiniMax-H3 需要至少 2 步");
  std::vector<float> s;
  for (int i = 0; i < steps; i++) {
    float b = 1.f - float(i) / float(steps - 1);
    float v = shift * b / (1.f + (shift - 1.f) * b);
    if (s.empty() || v != s.back()) s.push_back(v);
  }
  return s;
}

H3Dit::H3Dit(const TensorStore& st, int device, int max_layers, QMode compute) : dw_(st, device) {
  if (!qmode_supported(compute, device)) {
    MX_INFO("GPU%d 不支援 %s 運算，DiT 改用 bf16", device, qmode_name(compute));
    compute = QMode::BF16;
  }
  compute_ = compute;
  MX_CHECK(st.has("blocks.0.attn.qkv_proj.weight"), "不是 MiniMax-H3 DiT 權重（缺 blocks.0.attn.qkv_proj）");
  MX_CHECK(st.has("adaln_t_table"),
           "目前只支援 pruned（adaln_t_table）版本的 MiniMax-H3 GGUF，例如 unsloth minimax_h3_fl2va_pruned-*.gguf");
  const WeightRef& tb = st.get("adaln_t_table");
  MX_CHECK(tb.shape.size() == 2, "adaln_t_table 形狀錯誤");
  grid_ = tb.shape[0];
  basis_ = tb.shape[1];
  table_ = host_f32(tb);
  inv_freq_ = host_f32(st.get("rope.inv_freq"));
  rot_ = int(inv_freq_.size()) * 3 * 2;

  video_in_ = make_linear(dw_, "video_patch_proj");
  audio_in_ = make_linear(dw_, "audio_patch_proj");
  cond_proj_ = make_linear(dw_, "condition_proj");
  final_adaln_ = make_linear(dw_, "final_layer.adaln_proj.linear");
  video_out_ = make_linear(dw_, "final_layer.video_out");
  audio_out_ = make_linear(dw_, "final_layer.audio_out");
  final_norm_ = dw_.f32("final_layer.norm.weight");
  refiner_norm_ = dw_.f32("token_refiner.final_norm.weight");
  dim_ = video_in_.out;

  auto load = [&](const std::string& p, bool adaln) {
    Block b;
    b.qkv = make_linear_streamed(dw_, p + "attn.qkv_proj");
    b.out = make_linear_streamed(dw_, p + "attn.out_proj");
    b.fc1 = make_linear_streamed(dw_, p + "mlp.fc1");
    b.fc2 = make_linear_streamed(dw_, p + "mlp.fc2");
    if (adaln) b.adaln = make_linear(dw_, p + "adaln_proj.linear");
    b.norm1 = dw_.f32(p + "norm1.weight");
    b.norm2 = dw_.f32(p + "norm2.weight");
    b.q_norm = dw_.f32(p + "attn.q_norm.weight");
    b.k_norm = dw_.f32(p + "attn.k_norm.weight");
    return b;
  };
  int nr = 0, nb = 0;
  while (st.has("token_refiner.blocks." + std::to_string(nr) + ".attn.qkv_proj.weight")) nr++;
  while (st.has("blocks." + std::to_string(nb) + ".attn.qkv_proj.weight")) nb++;
  if (max_layers > 0) nb = std::min(nb, max_layers);
  refiner_.reserve(nr);
  blocks_.reserve(nb);
  for (int i = 0; i < nr; i++) refiner_.push_back(load("token_refiner.blocks." + std::to_string(i) + ".", false));
  for (int i = 0; i < nb; i++) blocks_.push_back(load("blocks." + std::to_string(i) + ".", true));
  inner_ = blocks_[0].qkv.out / 3;
  heads_ = int(inner_ / head_dim_);
  ffn_ = blocks_[0].fc2.in;
  MX_CHECK(blocks_[0].adaln.out == 18 * dim_ && blocks_[0].adaln.in == basis_, "adaln 形狀不符");

  st_refiner_ = std::make_unique<BlockStreamer>(device, compute_);
  for (auto& b : refiner_) st_refiner_->add_block(st, {&b.qkv, &b.out, &b.fc1, &b.fc2});
  st_refiner_->finalize();
  st_blocks_ = std::make_unique<BlockStreamer>(device, compute_);
  for (auto& b : blocks_) st_blocks_->add_block(st, {&b.qkv, &b.out, &b.fc1, &b.fc2});
  st_blocks_->finalize();
  MX_INFO("MiniMax-H3 DiT：%d 層 + %d 層 refiner、dim %lld、注意力 %d×%d、FFN %lld、%s、AdaLN 曲線表 %lld×%lld",
          nb, nr, (long long)dim_, heads_, head_dim_, (long long)ffn_,
          strf("%s/%s", gtype_name(blocks_[0].qkv.w.type), qmode_name(compute_)).c_str(),
          (long long)grid_, (long long)basis_);
}

size_t H3Dit::scratch_bytes() const {
  size_t m = 0;
  for (auto* v : {&refiner_, &blocks_})
    for (auto& b : *v)
      for (const Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2, &b.adaln})
        if (l->in) m = std::max(m, linear_scratch_bytes(*l));
  for (const Linear* l : {&video_in_, &audio_in_, &cond_proj_, &final_adaln_, &video_out_, &audio_out_})
    m = std::max(m, linear_scratch_bytes(*l));
  return m;
}

size_t H3Dit::act_bytes(int64_t S) const {
  size_t per_row = size_t(dim_) * (4 + 2) + size_t(inner_) * 2 * 4;
  size_t chunk = size_t(kChunk) * std::max(3 * inner_, 3 * ffn_) * 2;
  size_t mods = size_t(4) * 18 * dim_ * 4;
  size_t quant = compute_ == QMode::BF16 ? 0 : size_t(std::max<int64_t>(kChunk, S)) * std::max(ffn_, inner_) * 9 / 8;
  return per_row * S + chunk + mods + quant + (size_t(64) << 20);
}

std::vector<Linear*> H3Dit::linears() {
  std::vector<Linear*> v;
  for (auto* bs : {&refiner_, &blocks_})
    for (auto& b : *bs)
      for (Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2}) v.push_back(l);
  return v;
}

std::vector<std::string> H3Dit::linear_names() {
  std::vector<std::string> v;
  for (auto* l : linears()) v.push_back(l->name);
  return v;
}

// LoRA：原始 MiniMax 命名（blocks.N.attn.qkv_proj ...）與 diffusers 命名
// （transformer_blocks.N.attn.to_q / ff.net.0.proj ...）都對應到同一組層。
bool H3Dit::resolve_lora(const std::string& name, LoraTarget& t) {
  auto parse = [&](const std::string& pre, std::vector<Block>& bs, std::string& rest) -> Block* {
    if (name.rfind(pre, 0) != 0) return nullptr;
    size_t dot = name.find('.', pre.size());
    if (dot == std::string::npos) return nullptr;
    int i = std::atoi(name.substr(pre.size(), dot - pre.size()).c_str());
    if (i < 0 || i >= int(bs.size())) return nullptr;
    rest = name.substr(dot + 1);
    return &bs[i];
  };
  std::string r;
  Block* b = parse("blocks.", blocks_, r);
  if (!b) b = parse("transformer_blocks.", blocks_, r);
  if (!b) b = parse("token_refiner.blocks.", refiner_, r);
  if (!b) b = parse("token_refiner.refiner_blocks.", refiner_, r);
  if (!b) return false;
  if (r == "attn.qkv_proj") return t = {&b->qkv, 0, -1}, true;
  if (r == "attn.to_q") return t = {&b->qkv, 0, inner_}, true;
  if (r == "attn.to_k") return t = {&b->qkv, inner_, inner_}, true;
  if (r == "attn.to_v") return t = {&b->qkv, 2 * inner_, inner_}, true;
  if (r == "attn.out_proj" || r == "attn.to_out.0") return t = {&b->out, 0, -1}, true;
  if (r == "mlp.fc1" || r == "ff.net.0.proj") return t = {&b->fc1, 0, -1}, true;
  if (r == "mlp.fc2" || r == "ff.net.2") return t = {&b->fc2, 0, -1}, true;
  return false;
}

void H3Dit::time_basis(Ctx& c, const std::vector<float>& ts, float* basis) {
  std::vector<float> h(ts.size() * basis_);
  for (size_t i = 0; i < ts.size(); i++) {
    float pos = std::clamp(ts[i], 0.f, 1.f) * float(grid_ - 1);
    int64_t lo = std::min<int64_t>(int64_t(std::floor(pos)), grid_ - 2);
    float fr = pos - float(lo);
    for (int64_t j = 0; j < basis_; j++) {
      float a = table_[lo * basis_ + j], b = table_[(lo + 1) * basis_ + j];
      h[i * basis_ + j] = a + (b - a) * fr;
    }
  }
  MX_CUDA(cudaMemcpyAsync(basis, h.data(), h.size() * 4, cudaMemcpyHostToDevice, c.stream));
  c.sync();
}

// 一層：AdaLN(RMSNorm) → 注意力 → 閘控殘差 → AdaLN(RMSNorm) → SwiGLU → 閘控殘差。
// mods 為空時是 token refiner（無調變、無 RoPE）。
void H3Dit::attn_mlp(Ctx& c, const Block& b, float* x, int64_t S, const float* cs, const float* sn,
                     const std::vector<H3Segment>& segs, const float* mods, int64_t mod_stride) {
  const int64_t D = dim_, I = inner_, F = ffn_;
  auto m = [&](const H3Segment& g, int k) -> RowMod {
    if (!mods) return {};
    const float* p = mods + g.t_idx * mod_stride + int64_t(g.tag) * 6 * D + k * D;
    return {p, p, 0};
  };
  ArenaScope sc(c.act);
  bf16* h = c.act.alloc_n<bf16>(S * D);
  bf16* q = c.act.alloc_n<bf16>(S * I);
  bf16* k = c.act.alloc_n<bf16>(S * I);
  bf16* v = c.act.alloc_n<bf16>(S * I);
  bf16* o = c.act.alloc_n<bf16>(S * I);
  for (auto& g : segs)
    rmsnorm_mod(x + g.start * D, g.len, D, eps_, b.norm1, m(g, 1), m(g, 0), h + g.start * D, c.stream);
  const int rot = cs ? rot_ : 0;
  const RopeMode mode = cs ? ROPE_HALF : ROPE_NONE;
  for (int64_t r0 = 0; r0 < S; r0 += kChunk) {
    ArenaScope s2(c.act);
    int64_t n = std::min(kChunk, S - r0);
    bf16* t = c.act.alloc_n<bf16>(n * 3 * I);
    linear(c, b.qkv, h + r0 * D, n, t);
    const float* cr = cs ? cs + r0 * (rot_ / 2) : nullptr;
    const float* sr = sn ? sn + r0 * (rot_ / 2) : nullptr;
    head_norm_rope(t, n, 3 * I, 0, heads_, head_dim_, b.q_norm, eps_, cr, sr, rot, mode, q + r0 * I, c.stream);
    head_norm_rope(t, n, 3 * I, I, heads_, head_dim_, b.k_norm, eps_, cr, sr, rot, mode, k + r0 * I, c.stream);
    head_norm_rope(t, n, 3 * I, 2 * I, heads_, head_dim_, nullptr, 0.f, nullptr, nullptr, 0, ROPE_NONE,
                   v + r0 * I, c.stream);
  }
  AttnParams ap;
  ap.Sq = ap.Skv = int(S);
  ap.Hq = ap.Hkv = heads_;
  ap.D = head_dim_;
  flash_attn(q, k, v, o, ap, c.stream);
  linear(c, b.out, o, S, h);
  for (auto& g : segs) gated_residual(x + g.start * D, h + g.start * D, g.len, D, m(g, 2), 0, c.stream);

  for (auto& g : segs)
    rmsnorm_mod(x + g.start * D, g.len, D, eps_, b.norm2, m(g, 4), m(g, 3), h + g.start * D, c.stream);
  bf16* y = q;  // q 已用完；[S, I] ≥ [S, D]
  for (int64_t r0 = 0; r0 < S; r0 += kChunk) {
    ArenaScope s2(c.act);
    int64_t n = std::min(kChunk, S - r0);
    bf16* gu = c.act.alloc_n<bf16>(n * 2 * F);
    bf16* mid = c.act.alloc_n<bf16>(n * F);
    linear(c, b.fc1, h + r0 * D, n, gu);
    silu_mul(gu, n, F, mid, c.stream);
    linear(c, b.fc2, mid, n, y + r0 * D);
  }
  for (auto& g : segs) gated_residual(x + g.start * D, y + g.start * D, g.len, D, m(g, 5), 0, c.stream);
}

void H3Dit::prepare(Ctx& c, H3Seq& seq, const std::vector<float>& positions, const bf16* txt) {
  c.use();
  const int64_t S = seq.S, T = seq.text_len, D = dim_;
  MX_CHECK(int64_t(positions.size()) == S * 3, "positions 大小不符");
  const int64_t nf = int64_t(inv_freq_.size()), half = rot_ / 2;
  std::vector<float> hc(S * half), hs(S * half);
  for (int64_t s = 0; s < S; s++)
    for (int a = 0; a < 3; a++)
      for (int64_t j = 0; j < nf; j++) {
        float ang = positions[s * 3 + a] * inv_freq_[j];
        hc[s * half + a * nf + j] = std::cos(ang);
        hs[s * half + a * nf + j] = std::sin(ang);
      }
  seq.cos = DevBuf(hc.size() * 4, c.device);
  seq.sin = DevBuf(hs.size() * 4, c.device);
  MX_CUDA(cudaMemcpy(seq.cos.get(), hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(seq.sin.get(), hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));

  // 文字流：condition_proj → token refiner → final_norm。與 timestep 無關，整次生成只算一次。
  ArenaScope sc(c.act);
  bf16* tb = c.act.alloc_n<bf16>(T * D);
  float* tx = c.act.alloc_n<float>(T * D);
  linear(c, cond_proj_, txt, T, tb);
  bf16_to_f32(tb, tx, T * D, c.stream);
  std::vector<H3Segment> one = {{0, T, 0, H3_TEXT}};
  for (int i = 0; i < int(refiner_.size()); i++) {
    st_refiner_->acquire(c, i);
    attn_mlp(c, refiner_[i], tx, T, nullptr, nullptr, one, nullptr, 0);
  }
  rmsnorm_mod(tx, T, D, eps_, refiner_norm_, {}, {}, tb, c.stream);
  seq.text = DevBuf(size_t(T) * D * 4, c.device);
  bf16_to_f32(tb, seq.text.as<float>(), T * D, c.stream);
  c.sync();
}

void H3Dit::forward(Ctx& c, H3Seq& seq, const float* video, const float* audio,
                    const std::vector<float>& timesteps, float* v_video, float* v_audio) {
  c.use();
  const int64_t S = seq.S, D = dim_;
  const int64_t nt = int64_t(timesteps.size());
  ArenaScope sc(c.act);
  float* basis = c.act.alloc_n<float>(nt * basis_);
  time_basis(c, timesteps, basis);
  float* x = c.act.alloc_n<float>(S * D);
  MX_CUDA(cudaMemcpyAsync(x + seq.text_start * D, seq.text.get(), size_t(seq.text_len) * D * 4,
                          cudaMemcpyDeviceToDevice, c.stream));
  linear_f32(c, video_in_, video, seq.cond_len, x + seq.cond_start * D);
  linear_f32(c, video_in_, video + seq.cond_len * video_in_.in, seq.video_len, x + seq.video_start * D);
  linear_f32(c, audio_in_, audio, seq.audio_len, x + seq.audio_start * D);

  const int64_t ms = 18 * D;
  float* mods = c.act.alloc_n<float>(nt * ms);
  const float* cs = seq.cos.as<float>();
  const float* sn = seq.sin.as<float>();
  for (int i = 0; i < int(blocks_.size()); i++) {
    st_blocks_->acquire(c, i);
    linear_f32(c, blocks_[i].adaln, basis, nt, mods);  // pruned：曲線基底直接投影，不過 silu
    attn_mlp(c, blocks_[i], x, S, cs, sn, seq.segs, mods, ms);
  }

  // 輸出頭：final RMSNorm + (shift, scale) 只依 timestep；只算目標影片列與音訊列。
  float* fm = c.act.alloc_n<float>(nt * 2 * D);
  linear_f32(c, final_adaln_, basis, nt, fm);
  auto head = [&](int64_t start, int64_t len, const Linear& out, float* dst) {
    const H3Segment* g = nullptr;
    for (auto& s : seq.segs)
      if (start >= s.start && start < s.start + s.len) g = &s;
    MX_CHECK(g && start + len <= g->start + g->len, "輸出列跨越多個段");
    const float* row = fm + g->t_idx * 2 * D;
    ArenaScope s2(c.act);
    bf16* hb = c.act.alloc_n<bf16>(len * D);
    float* hf = c.act.alloc_n<float>(len * D);
    rmsnorm_mod(x + start * D, len, D, eps_, final_norm_, {row + D, row + D, 0}, {row, row, 0}, hb, c.stream);
    bf16_to_f32(hb, hf, len * D, c.stream);
    linear_f32(c, out, hf, len, dst);
  };
  head(seq.video_start, seq.video_len, video_out_, v_video);
  head(seq.audio_start, seq.audio_len, audio_out_, v_audio);
}

}  // namespace mx
