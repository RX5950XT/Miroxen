#include "models/qwen_image/dit.h"

#include <algorithm>
#include <cmath>

#include "core/dequant.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

namespace mx {

static float round_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

QwenImageDit::QwenImageDit(const TensorStore& st, int device, QMode compute, bool stream) : dw_(st, device) {
  attn8_ = compute != QMode::BF16 && flash_attn_fp8_supported(device);
  if (!qmode_supported(compute, device)) {
    MX_INFO("GPU%d 不支援 %s 運算，DiT 改用 bf16", device, qmode_name(compute));
    compute = QMode::BF16;
  }
  compute_ = compute;
  int n = 0;
  while (st.has("transformer_blocks." + std::to_string(n) + ".attn.to_q.weight")) n++;
  cfg_.layers = n;
  MX_CHECK(n > 0, "不是 Qwen-Image-2.1 權重：找不到 transformer_blocks.0");
  img_in_ = make_linear(dw_, "img_in");
  txt_in1_ = make_linear(dw_, "txt_in.in_layer");
  txt_in2_ = make_linear(dw_, "txt_in.out_layer");
  t1_ = make_linear(dw_, "time_text_embed.timestep_embedder.linear_1");
  t2_ = make_linear(dw_, "time_text_embed.timestep_embedder.linear_2");
  modulation_ = make_linear(dw_, "modulation.1");
  norm_out_ = make_linear(dw_, "norm_out.linear");
  proj_out_ = make_linear(dw_, "proj_out");
  cfg_.dim = img_in_.out;
  cfg_.in_ch = img_in_.in;
  cfg_.ctx_dim = txt_in1_.in;
  cfg_.heads = int(cfg_.dim / cfg_.head_dim);
  MX_CHECK(modulation_.out == 4 * cfg_.dim, "modulation 形狀不符");

  const WeightRef& tn = st.get("txt_in.text_norm.weight");
  std::vector<float> w(tn.numel());
  dequant_cpu(tn.data, tn.type, tn.numel(), w.data());
  for (auto& x : w) x += 1.f;  // checkpoint 存的是 scale-1
  txt_norm_w_ = DevBuf(w.size() * 4, device);
  MX_CUDA(cudaMemcpy(txt_norm_w_.get(), w.data(), w.size() * 4, cudaMemcpyHostToDevice));

  blocks_.reserve(n);  // 串流器保存 Linear 位址，不能重新配置
  for (int i = 0; i < n; i++) {
    std::string p = "transformer_blocks." + std::to_string(i) + ".";
    auto lin = [&](const std::string& name) { return stream ? make_linear_streamed(dw_, p + name) : make_linear(dw_, p + name); };
    Block b;
    b.q = lin("attn.to_q");
    b.k = lin("attn.to_k");
    b.v = lin("attn.to_v");
    b.o = lin("attn.to_out.0");
    if (st.has(p + "img_mlp.gate_up.weight")) b.gate_up = lin("img_mlp.gate_up");
    else if (stream) b.gate_up = make_linear_streamed_concat(dw_, p + "img_mlp.gate_up", p + "img_mlp.gate_layer", p + "img_mlp.proj");
    else b.gate_up = make_linear_concat(dw_, p + "img_mlp.gate_up", p + "img_mlp.gate_layer", p + "img_mlp.proj");
    b.out = lin("img_mlp.out");
    b.norm_q = dw_.f32(p + "attn.norm_q.weight");
    b.norm_k = dw_.f32(p + "attn.norm_k.weight");
    if (!stream)
      for (Linear* l : {&b.q, &b.k, &b.v, &b.o, &b.gate_up, &b.out}) dw_.quantize(*l, compute);
    blocks_.push_back(b);
  }
  if (stream) {
    streamer_ = std::make_unique<BlockStreamer>(device, compute_);
    for (auto& b : blocks_) streamer_->add_block(st, {&b.q, &b.k, &b.v, &b.o, &b.gate_up, &b.out});
    streamer_->finalize();
  }
  cfg_.ffn = blocks_[0].out.in;
  MX_INFO("Qwen-Image DiT：%d 層、dim %lld、權重 %.2f GiB（%s，運算 %s%s）", n, (long long)cfg_.dim,
          weight_bytes() / 1073741824.0, gtype_name(blocks_[0].q.w.type), qmode_name(compute_), stream ? "，逐層串流" : "");
}

size_t QwenImageDit::scratch_bytes() const {
  size_t m = 0;
  for (auto* l : const_cast<QwenImageDit*>(this)->linears()) m = std::max(m, linear_scratch_bytes(*l));
  return m;
}

std::vector<Linear*> QwenImageDit::linears() {
  std::vector<Linear*> v = {&img_in_, &txt_in1_, &txt_in2_, &t1_, &t2_, &modulation_, &norm_out_, &proj_out_};
  for (auto& b : blocks_)
    for (Linear* l : {&b.q, &b.k, &b.v, &b.o, &b.gate_up, &b.out}) v.push_back(l);
  return v;
}

std::vector<std::string> QwenImageDit::linear_names() {
  std::vector<std::string> v;
  for (auto* l : linears()) v.push_back(l->name);
  for (int i = 0; i < cfg_.layers; i++) {
    std::string p = "transformer_blocks." + std::to_string(i) + ".img_mlp.";
    v.push_back(p + "gate_layer");
    v.push_back(p + "proj");
  }
  return v;
}

bool QwenImageDit::resolve_lora(const std::string& name, LoraTarget& t) {
  for (Linear* l : {&img_in_, &txt_in1_, &txt_in2_, &proj_out_})
    if (name == l->name) return t.linear = l, true;
  for (Linear* l : {&t1_, &t2_, &modulation_, &norm_out_})
    if (name == l->name) return t.linear = l, t.f32 = true, true;
  const std::string tb = "transformer_blocks.";
  if (name.rfind(tb, 0) != 0) return false;
  size_t dot = name.find('.', tb.size());
  if (dot == std::string::npos) return false;
  int i = std::atoi(name.substr(tb.size(), dot - tb.size()).c_str());
  if (i < 0 || i >= cfg_.layers) return false;
  std::string r = name.substr(dot + 1);
  Block& b = blocks_[i];
  if (r == "attn.to_q") return t.linear = &b.q, true;
  if (r == "attn.to_k") return t.linear = &b.k, true;
  if (r == "attn.to_v") return t.linear = &b.v, true;
  if (r == "attn.to_out.0" || r == "attn.to_out") return t.linear = &b.o, true;
  if (r == "img_mlp.out") return t.linear = &b.out, true;
  if (r == "img_mlp.gate_up") return t.linear = &b.gate_up, true;
  if (r == "img_mlp.gate_layer") return t = {&b.gate_up, 0, cfg_.ffn}, true;
  if (r == "img_mlp.proj") return t = {&b.gate_up, cfg_.ffn, cfg_.ffn}, true;
  return false;
}

void QwenImageDit::prepare(Ctx& c, QwenImageSeq& seq, const bf16* txt, int64_t T, int64_t H, int64_t W,
                           const std::vector<QwenImageCond>& conds) {
  c.use();
  seq = QwenImageSeq{};
  seq.H = H, seq.W = W;
  // prefix 版面：條件圖的 <|image_pad|> 列（每列 2×2 個 latent）換成整張圖的 latent 段。
  struct Run {
    int64_t src, len;  // 文字段：txt 中的起點；圖片段：conds 索引
    bool text;
  };
  std::vector<Run> runs;
  int64_t cur = 0;
  for (size_t i = 0; i < conds.size(); i++) {
    const auto& cd = conds[i];
    MX_CHECK(cd.H % 2 == 0 && cd.W % 2 == 0 && cd.slot0 >= cur, "條件圖版面錯誤");
    MX_CHECK(cd.slot0 + cd.H * cd.W / 4 <= T, "條件圖佔位超出文字序列");
    if (cd.slot0 > cur) runs.push_back({cur, cd.slot0 - cur, true});
    runs.push_back({int64_t(i), cd.H * cd.W, false});
    cur = cd.slot0 + cd.H * cd.W / 4;
  }
  if (T > cur) runs.push_back({cur, T - cur, true});
  int64_t P = 0;
  for (auto& r : runs) {
    seq.segs.push_back({P, r.len, r.text});
    P += r.len;
  }
  seq.T = P;
  const int64_t S = seq.S(), dim = cfg_.dim;
  // RoPE：frame 16 / h 56 / w 56 個通道 → 8 / 28 / 28 組複數對。文字三軸同步遞增；
  // 每張圖（條件圖與目標圖）frame 固定在當下位置、h/w 以 0 為中心，之後位置前進 max(h, w)。
  const int axes[3] = {16, 56, 56};
  std::vector<float> invf;
  for (int a = 0; a < 3; a++)
    for (int j = 0; j < axes[a] / 2; j++)
      invf.push_back(1.0f / std::pow(10000.0f, float(2 * j) / float(axes[a])));
  const int pairs = int(invf.size());
  MX_CHECK(pairs * 2 == cfg_.head_dim, "RoPE 維度不符");
  std::vector<float> hc(S * pairs), hs(S * pairs);
  auto put = [&](int64_t s, int64_t f, int64_t h, int64_t w) {
    for (int p = 0; p < pairs; p++) {
      int64_t pos = p < 8 ? f : p < 36 ? h : w;
      float ang = float(pos) * invf[p];
      hc[s * pairs + p] = std::cos(ang);
      hs[s * pairs + p] = std::sin(ang);
    }
  };
  int64_t position = 0, row = 0;
  auto image = [&](int64_t ih, int64_t iw) {
    for (int64_t r = 0; r < ih; r++)
      for (int64_t col = 0; col < iw; col++) put(row++, position, r - (ih - ih / 2), col - (iw - iw / 2));
    position += std::max(ih, iw);
  };
  for (auto& r : runs) {
    if (r.text)
      for (int64_t k = 0; k < r.len; k++, position++) put(row++, position, position, position);
    else
      image(conds[r.src].H, conds[r.src].W);
  }
  image(H, W);
  seq.cos = DevBuf(hc.size() * 4, c.device);
  seq.sin = DevBuf(hs.size() * 4, c.device);
  MX_CUDA(cudaMemcpy(seq.cos.get(), hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(seq.sin.get(), hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));

  // txt_in：zero-centered RMSNorm → Linear → GELU(tanh) → Linear（整條文字序列一起算，佔位列之後丟掉）。
  ArenaScope sc(c.act);
  bf16* a = c.act.alloc_n<bf16>(T * std::max(cfg_.ctx_dim, dim));
  bf16* b = c.act.alloc_n<bf16>(T * dim);
  rmsnorm_bf16(txt, T, cfg_.ctx_dim, cfg_.eps, txt_norm_w_.as<float>(), a, c.stream);
  linear(c, txt_in1_, a, T, b);
  gelu_tanh_inplace(b, T * dim, c.stream);
  linear(c, txt_in2_, b, T, a);
  seq.txt = DevBuf(size_t(P) * dim * 4, c.device);
  float* dst = seq.txt.as<float>();
  for (size_t i = 0; i < runs.size(); i++) {
    float* out = dst + seq.segs[i].start * dim;
    if (runs[i].text) {
      bf16_to_f32(a + runs[i].src * dim, out, runs[i].len * dim, c.stream);
    } else {
      const auto& cd = conds[runs[i].src];
      ArenaScope s2(c.act);
      const int64_t n = cd.H * cd.W;
      bf16* lb = c.act.alloc_n<bf16>(n * cfg_.in_ch);
      bf16* hb = c.act.alloc_n<bf16>(n * dim);
      f32_to_bf16(cd.latents, lb, n * cfg_.in_ch, c.stream);
      linear(c, img_in_, lb, n, hb);
      bf16_to_f32(hb, out, n * dim, c.stream);
    }
  }
  for (int i = 0; i < cfg_.layers; i++) {
    seq.kcache.emplace_back(size_t(P) * dim * sizeof(bf16), c.device);
    seq.vcache.emplace_back(size_t(P) * dim * sizeof(bf16), c.device);
  }
  c.sync();
}

// mod：[2, 4·dim]（第 0 列＝目標 t、第 1 列＝t=0）；out_scale：[2, dim]。
void QwenImageDit::time_mod(Ctx& c, float t, float* mod, float* out_scale) {
  const int half = 128;
  std::vector<float> emb(2 * 256);
  float ts[2] = {round_bf16(t) * 1000.f, 0.f};  // diffusers 先把 t 轉成 bf16
  for (int r = 0; r < 2; r++)
    for (int i = 0; i < half; i++) {
      float f = std::exp(-std::log(10000.f) * float(i) / float(half));
      float arg = ts[r] * f;
      emb[r * 256 + i] = round_bf16(std::cos(arg));
      emb[r * 256 + half + i] = round_bf16(std::sin(arg));
    }
  ArenaScope sc(c.act);
  float* e = c.act.alloc_n<float>(512);
  float* h = c.act.alloc_n<float>(2 * cfg_.dim);
  float* temb = c.act.alloc_n<float>(2 * cfg_.dim);
  MX_CUDA(cudaMemcpyAsync(e, emb.data(), 512 * 4, cudaMemcpyHostToDevice, c.stream));
  linear_f32(c, t1_, e, 2, h);
  silu_inplace_f32(h, 2 * cfg_.dim, c.stream);
  linear_f32(c, t2_, h, 2, temb);
  silu_inplace_f32(temb, 2 * cfg_.dim, c.stream);
  linear_f32(c, modulation_, temb, 2, mod);
  linear_f32(c, norm_out_, temb, 2, out_scale);
  c.sync();  // emb 是區域變數
}

void QwenImageDit::run_block(Ctx& c, QwenImageSeq& seq, int li, float* x, int64_t rows,
                             bool prefill, const float* mod) {
  const Block& b = blocks_[li];
  const int64_t dim = cfg_.dim, T = seq.T, S = seq.S();
  const int H = cfg_.heads, D = cfg_.head_dim;
  const int64_t row0 = prefill ? 0 : T;  // 這批列在完整序列中的起點
  auto rm = [&](int64_t off) {
    const float* tgt = mod + off;
    const float* pre = mod + 4 * dim + off;
    return prefill ? RowMod{tgt, pre, T} : RowMod{tgt, tgt, 0};
  };
  ArenaScope sc(c.act);
  bf16* h = c.act.alloc_n<bf16>(rows * dim);
  bf16* tmp = c.act.alloc_n<bf16>(rows * dim);
  bf16* q = c.act.alloc_n<bf16>(rows * dim);
  bf16* k = c.act.alloc_n<bf16>(S * dim);
  bf16* v = c.act.alloc_n<bf16>(S * dim);
  const float* cs = seq.cos.as<float>() + row0 * (D / 2);
  const float* sn = seq.sin.as<float>() + row0 * (D / 2);

  layernorm_mod(x, rows, dim, cfg_.eps, rm(0), {}, h, c.stream);
  {
    ArenaScope s2(c.act);
    bf16* kt = c.act.alloc_n<bf16>(rows * dim);
    linear_multi(c, h, rows, {{&b.q, tmp}, {&b.k, kt}, {&b.v, v + row0 * dim}});
    head_norm_rope(tmp, rows, dim, 0, H, D, b.norm_q, cfg_.eps, cs, sn, D, ROPE_INTERLEAVED, q, c.stream);
    head_norm_rope(kt, rows, dim, 0, H, D, b.norm_k, cfg_.eps, cs, sn, D, ROPE_INTERLEAVED, k + row0 * dim,
                   c.stream);
  }
  const size_t pre_bytes = size_t(T) * dim * sizeof(bf16);
  if (prefill) {
    MX_CUDA(cudaMemcpyAsync(seq.kcache[li].get(), k, pre_bytes, cudaMemcpyDeviceToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(seq.vcache[li].get(), v, pre_bytes, cudaMemcpyDeviceToDevice, c.stream));
  } else {
    MX_CUDA(cudaMemcpyAsync(k, seq.kcache[li].get(), pre_bytes, cudaMemcpyDeviceToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(v, seq.vcache[li].get(), pre_bytes, cudaMemcpyDeviceToDevice, c.stream));
  }
  AttnParams ap;
  ap.Hq = ap.Hkv = H;
  ap.D = D;
  if (prefill)  // prefix 各段看自己之前的所有列；文字段內因果、圖片段內雙向
    for (auto& g : seq.segs) {
      ap.Sq = int(g.len);
      ap.Skv = int(g.start + g.len);
      ap.causal = g.text ? 1 : 0;
      ap.q_offset = int(g.start);
      flash_attn(q + g.start * dim, k, v, tmp + g.start * dim, ap, c.stream);
    }
  ap.q_offset = 0;
  ap.Sq = int(seq.N());  // 目標圖：看完整序列
  ap.Skv = int(S);
  ap.causal = 0;
  const int64_t qoff = prefill ? T : 0;
  if (attn8_) {  // 低精度模式下注意力也用 INT8/FP8 tensor core
    ArenaScope s3(c.act);
    flash_attn_fp8(q + qoff * dim, k, v, tmp + qoff * dim, ap, c.act.alloc(flash_attn_fp8_workspace(ap)), c.stream);
  } else {
    flash_attn(q + qoff * dim, k, v, tmp + qoff * dim, ap, c.stream);
  }
  linear(c, b.o, tmp, rows, h);
  gated_residual(x, h, rows, dim, rm(dim), 1, c.stream);

  layernorm_mod(x, rows, dim, cfg_.eps, rm(2 * dim), {}, h, c.stream);
  const int64_t F = cfg_.ffn;
  bf16* gu = c.act.alloc_n<bf16>(rows * 2 * F);
  linear(c, b.gate_up, h, rows, gu);
  bf16* m = c.act.alloc_n<bf16>(rows * F);
  silu_mul(gu, rows, F, m, c.stream);
  linear(c, b.out, m, rows, h);
  gated_residual(x, h, rows, dim, rm(3 * dim), 1, c.stream);
}

void QwenImageDit::forward(Ctx& c, QwenImageSeq& seq, const float* latents, float t, float* v) {
  c.use();
  const int64_t dim = cfg_.dim, T = seq.T, N = seq.N();
  const bool prefill = !seq.cached;
  const int64_t rows = prefill ? seq.S() : N;
  ArenaScope sc(c.act);
  float* mod = c.act.alloc_n<float>(8 * dim);
  float* oscale = c.act.alloc_n<float>(2 * dim);
  time_mod(c, t, mod, oscale);

  float* x = c.act.alloc_n<float>(rows * dim);
  float* ximg = prefill ? x + T * dim : x;
  if (prefill)
    MX_CUDA(cudaMemcpyAsync(x, seq.txt.get(), size_t(T) * dim * 4, cudaMemcpyDeviceToDevice, c.stream));
  {
    ArenaScope s2(c.act);
    bf16* lb = c.act.alloc_n<bf16>(N * cfg_.in_ch);
    bf16* hb = c.act.alloc_n<bf16>(N * dim);
    f32_to_bf16(latents, lb, N * cfg_.in_ch, c.stream);
    linear(c, img_in_, lb, N, hb);
    bf16_to_f32(hb, ximg, N * dim, c.stream);
  }
  for (int i = 0; i < cfg_.layers; i++) {
    if (streamer_) streamer_->acquire(c, i);
    run_block(c, seq, i, x, rows, prefill, mod);
  }
  seq.cached = true;

  // 只對目標列做輸出頭。
  bf16* h = c.act.alloc_n<bf16>(N * dim);
  layernorm_mod(ximg, N, dim, cfg_.eps, RowMod{oscale, oscale, 0}, {}, h, c.stream);
  bf16* o = c.act.alloc_n<bf16>(N * proj_out_.out);
  linear(c, proj_out_, h, N, o);
  bf16_to_f32(o, v, N * proj_out_.out, c.stream);
}

}  // namespace mx
