#include "text/qwen3_vision.h"

#include <algorithm>
#include <cmath>

#include "core/dequant.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

namespace mx {

namespace {
constexpr int kPatch = 16, kMerge = 2;

std::vector<float> host_values(const WeightRef& r) {
  std::vector<float> v(size_t(r.numel()));
  dequant_cpu(r.data, r.type, r.numel(), v.data());
  return v;
}
}  // namespace

// 兩種命名對照：llama.cpp mmproj / HuggingFace。
struct VisionNames {
  bool hf;
  std::string blk(int i, const char* llama, const char* h) const {
    return hf ? strf("visual.blocks.%d.%s", i, h) : strf("v.blk.%d.%s", i, llama);
  }
};

Qwen3VisionEncoder::Norm Qwen3VisionEncoder::load_norm(const std::string& p) {
  const WeightRef& w = dw_.store().get(p + ".weight");
  std::vector<float> v = host_values(w);
  for (auto& x : v) x -= 1.f;
  Norm n;
  n.scale = DevBuf(v.size() * 4, dw_.device());
  MX_CUDA(cudaMemcpy(n.scale.get(), v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  n.bias = dw_.f32(p + ".bias");
  return n;
}

Qwen3VisionEncoder::Qwen3VisionEncoder(const TensorStore& st, int device) : dw_(st, device) {
  VisionNames nm{st.has("visual.patch_embed.proj.weight")};
  MX_CHECK(nm.hf || st.has("v.patch_embd.weight"), "找不到 Qwen3-VL 視覺塔權重（需要 mmproj 或 visual.*）");
  const std::string pos_name = nm.hf ? "visual.pos_embed.weight" : "v.position_embd.weight";
  const WeightRef& pr = st.get(pos_name);
  MX_CHECK(pr.shape.size() == 2, "視覺位置嵌入形狀錯誤");
  dim_ = pr.shape[1];
  grid_side_ = int(std::lround(std::sqrt(double(pr.shape[0]))));
  MX_CHECK(int64_t(grid_side_) * grid_side_ == pr.shape[0], "視覺位置嵌入不是方格");
  pos_ = host_values(pr);
  head_dim_ = 72;
  heads_ = int(dim_ / head_dim_);

  // patch 嵌入：Conv3d(3→dim, kernel 2×16×16, stride 同 kernel) = 線性層，輸入依 (通道, 時間, 列, 行) 攤平。
  const int P2 = kPatch * kPatch, K = 3 * 2 * P2;
  std::vector<float> wsum(size_t(dim_) * K, 0.f);
  if (nm.hf) {
    const WeightRef& r = st.get("visual.patch_embed.proj.weight");  // [dim·3, 2, 16, 16] = [dim, 3, 2, 16, 16]
    MX_CHECK(r.numel() == dim_ * K, "patch_embed 形狀錯誤");
    wsum = host_values(r);
  } else {
    const char* names[2] = {"v.patch_embd.weight", "v.patch_embd.weight.1"};  // 時間切片 0／1，各 [dim, 3, 16, 16]
    for (int t = 0; t < 2; t++) {
      const WeightRef& r = st.get(names[t]);
      MX_CHECK(r.numel() == dim_ * 3 * P2, "patch_embd 形狀錯誤");
      auto v = host_values(r);
      for (int64_t o = 0; o < dim_; o++)
        for (int ch = 0; ch < 3; ch++)
          for (int k = 0; k < P2; k++) wsum[o * K + (ch * 2 + t) * P2 + k] = v[(o * 3 + ch) * P2 + k];
    }
  }
  std::vector<bf16> wb(wsum.size());
  for (size_t i = 0; i < wb.size(); i++) wb[i] = __float2bfloat16(wsum[i]);
  patch_w_ = DevBuf(wb.size() * 2, device);
  MX_CUDA(cudaMemcpy(patch_w_.get(), wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
  patch_.name = "vision.patch_embed";
  patch_.w = {patch_w_.get(), GType::BF16, {dim_, K}};
  patch_.in = K, patch_.out = dim_;
  patch_.bias = dw_.f32(nm.hf ? "visual.patch_embed.proj.bias" : "v.patch_embd.bias");

  int depth = 0;
  while (st.has(nm.blk(depth, "attn_qkv.weight", "attn.qkv.weight"))) depth++;
  MX_CHECK(depth > 0, "視覺塔沒有任何層");
  blocks_.reserve(depth);
  for (int i = 0; i < depth; i++) {
    auto pre = [&](const char* l, const char* h) { return nm.blk(i, l, h); };
    Block b;
    b.n1 = load_norm(pre("ln1", "norm1"));
    b.n2 = load_norm(pre("ln2", "norm2"));
    b.qkv = make_linear(dw_, pre("attn_qkv", "attn.qkv"));
    b.proj = make_linear(dw_, pre("attn_out", "attn.proj"));
    b.fc1 = make_linear(dw_, pre("ffn_up", "mlp.linear_fc1"));
    b.fc2 = make_linear(dw_, pre("ffn_down", "mlp.linear_fc2"));
    blocks_.push_back(std::move(b));
  }
  if (nm.hf) {
    merger_.norm = load_norm("visual.merger.norm");
    merger_.fc1 = make_linear(dw_, "visual.merger.linear_fc1");
    merger_.fc2 = make_linear(dw_, "visual.merger.linear_fc2");
    const int defaults[3] = {8, 16, 24};  // HF 權重沒有 metadata，用 Qwen3-VL 的 deepstack_visual_indexes
    for (int j = 0; st.has(strf("visual.deepstack_merger_list.%d.linear_fc1.weight", j)); j++) {
      MX_CHECK(j < 3, "DeepStack 層數超出預期");
      std::string p = strf("visual.deepstack_merger_list.%d.", j);
      Merger m;
      m.post_shuffle = true;
      m.norm = load_norm(p + "norm");
      m.fc1 = make_linear(dw_, p + "linear_fc1");
      m.fc2 = make_linear(dw_, p + "linear_fc2");
      ds_layers_.push_back(defaults[j]);
      ds_.push_back(std::move(m));
    }
  } else {
    merger_.norm = load_norm("v.post_ln");
    merger_.fc1 = make_linear(dw_, "mm.0");
    merger_.fc2 = make_linear(dw_, "mm.2");
    for (int l = 0; l < depth; l++) {
      std::string p = strf("v.deepstack.%d.", l);
      if (!st.has(p + "fc1.weight")) continue;
      Merger m;
      m.post_shuffle = true;
      m.norm = load_norm(p + "norm");
      m.fc1 = make_linear(dw_, p + "fc1");
      m.fc2 = make_linear(dw_, p + "fc2");
      ds_layers_.push_back(l);
      ds_.push_back(std::move(m));
    }
  }
  MX_INFO("Qwen3-VL 視覺塔：%d 層、dim %lld、輸出 %lld、DeepStack %zu 層，權重 %.2f GiB", depth, (long long)dim_,
          (long long)out_dim(), ds_.size(), dw_.bytes() / 1073741824.0);
}

size_t Qwen3VisionEncoder::scratch_bytes() const {
  size_t m = 0;
  for (auto& b : blocks_)
    for (const Linear* l : {&b.qkv, &b.proj, &b.fc1, &b.fc2}) m = std::max(m, linear_scratch_bytes(*l));
  for (const Merger* g : {&merger_}) m = std::max({m, linear_scratch_bytes(g->fc1), linear_scratch_bytes(g->fc2)});
  for (auto& g : ds_) m = std::max({m, linear_scratch_bytes(g.fc1), linear_scratch_bytes(g.fc2)});
  return m;
}

size_t Qwen3VisionEncoder::act_bytes(int h, int w) const {
  const size_t n = size_t(h / kPatch) * (w / kPatch);
  const size_t ffn = size_t(blocks_[0].fc1.out), od = size_t(out_dim());
  // x(f32) + cos/sin + 每層暫存（qkv、q/k/v、注意力輸出、FFN）+ 合併器。
  return n * dim_ * 4 + n * head_dim_ * 4 + n * (dim_ * 3 + dim_ * 4 + ffn) * 2 +
         (n / 4) * (dim_ * 4 * 3 + od) * 2 + (size_t(64) << 20);
}

// LayerNorm → （DeepStack 先 shuffle）→ fc1 → GELU(erf) → fc2。
void Qwen3VisionEncoder::merge(Ctx& c, const Merger& m, const float* x, int64_t n, bf16* out) {
  ArenaScope sc(c.act);
  const int64_t g = n / 4, wide = dim_ * 4;
  bf16* h = c.act.alloc_n<bf16>(n * dim_);
  RowMod sc_mod{m.norm.scale.as<float>(), m.norm.scale.as<float>(), 0};
  RowMod sh_mod{m.norm.bias, m.norm.bias, 0};
  // 4 個相鄰 token 的 [dim] 在記憶體中連續，所以 [n, dim] 與 [n/4, 4·dim] 是同一塊資料。
  if (m.post_shuffle)
    layernorm_mod(x, g, wide, eps_, sc_mod, sh_mod, h, c.stream);
  else
    layernorm_mod(x, n, dim_, eps_, sc_mod, sh_mod, h, c.stream);
  bf16* t = c.act.alloc_n<bf16>(g * wide);
  linear(c, m.fc1, h, g, t);
  gelu_erf_inplace(t, g * wide, c.stream);
  linear(c, m.fc2, t, g, out);
}

VisionFeatures Qwen3VisionEncoder::encode_pair(Ctx& c, const uint8_t* rgb0, const uint8_t* rgb1, int h, int w) {
  MX_CHECK(h > 0 && w > 0 && h % (kPatch * kMerge) == 0 && w % (kPatch * kMerge) == 0,
           "視覺塔輸入尺寸必須是 32 的倍數（%dx%d）", h, w);
  c.use();
  const int64_t ph = h / kPatch, pw = w / kPatch, n = ph * pw, K = 3 * 2 * kPatch * kPatch;
  // token 順序：2×2 合併區塊優先（區塊列、區塊行、區塊內列、區塊內行）。
  std::vector<int64_t> rr(n), cc(n);
  for (int64_t i = 0; i < n; i++) {
    int64_t blk = i / 4, in = i % 4;
    rr[i] = (blk / (pw / 2)) * 2 + in / 2;
    cc[i] = (blk % (pw / 2)) * 2 + in % 2;
  }
  std::vector<bf16> patches(n * K);
  for (int64_t i = 0; i < n; i++)
    for (int ch = 0; ch < 3; ch++)
      for (int t = 0; t < 2; t++)
        for (int y = 0; y < kPatch; y++)
          for (int x = 0; x < kPatch; x++) {
            int64_t py = rr[i] * kPatch + y, px = cc[i] * kPatch + x;
            const uint8_t* rgb = t ? rgb1 : rgb0;
            float v = float(rgb[(py * w + px) * 3 + ch]) / 255.f * 2.f - 1.f;  // mean 0.5、std 0.5
            patches[i * K + ((ch * 2 + t) * kPatch + y) * kPatch + x] = __float2bfloat16(v);
          }
  // 學到的 48×48 位置表以雙線性（align_corners）縮放到 ph×pw。
  std::vector<float> pos(n * dim_, 0.f);
  auto taps = [&](int64_t idx, int64_t size, int64_t& i0, int64_t& i1, float& f) {
    double src = size > 1 ? double(idx) * (grid_side_ - 1) / double(size - 1) : 0.0;
    i0 = int64_t(std::floor(src));
    f = float(src - double(i0));
    i1 = std::min<int64_t>(i0 + 1, grid_side_ - 1);
  };
  for (int64_t i = 0; i < n; i++) {
    int64_t y0, y1, x0, x1;
    float fy, fx;
    taps(rr[i], ph, y0, y1, fy);
    taps(cc[i], pw, x0, x1, fx);
    const float wt[4] = {(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx};
    const int64_t id[4] = {y0 * grid_side_ + x0, y0 * grid_side_ + x1, y1 * grid_side_ + x0, y1 * grid_side_ + x1};
    for (int t = 0; t < 4; t++)
      for (int64_t d = 0; d < dim_; d++) pos[i * dim_ + d] += wt[t] * pos_[id[t] * dim_ + d];
  }
  // 2D RoPE：head_dim 72，前 18 個頻率給列、後 18 個給行（rotate_half，rot = 72）。
  const int half = head_dim_ / 2, quarter = half / 2;
  std::vector<float> hc(n * half), hs(n * half);
  for (int64_t i = 0; i < n; i++)
    for (int j = 0; j < half; j++) {
      int k = j < quarter ? j : j - quarter;
      double inv = std::pow(10000.0, -2.0 * k / double(half));
      double ang = double(j < quarter ? rr[i] : cc[i]) * inv;
      hc[i * half + j] = float(std::cos(ang));
      hs[i * half + j] = float(std::sin(ang));
    }

  ArenaScope sc(c.act);
  float* x = c.act.alloc_n<float>(n * dim_);
  float* cs = c.act.alloc_n<float>(n * half);
  float* sn = c.act.alloc_n<float>(n * half);
  bf16* hb = c.act.alloc_n<bf16>(n * std::max<int64_t>(dim_, K));
  {
    ArenaScope s2(c.act);
    bf16* pin = c.act.alloc_n<bf16>(n * K);
    float* pdev = c.act.alloc_n<float>(n * dim_);
    MX_CUDA(cudaMemcpyAsync(pin, patches.data(), patches.size() * 2, cudaMemcpyHostToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(pdev, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(cs, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(sn, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice, c.stream));
    linear(c, patch_, pin, n, hb);
    bf16_to_f32(hb, x, n * dim_, c.stream);
    axpy_f32(x, pdev, 1.f, n * dim_, c.stream);
    c.sync();  // 主機緩衝區即將離開作用域
  }

  VisionFeatures out;
  out.gh = ph / kMerge, out.gw = pw / kMerge;
  const int64_t g = n / 4, od = out_dim();
  bf16* mo = c.act.alloc_n<bf16>(g * od);
  auto fetch = [&](std::vector<bf16>& dst) {
    dst.resize(size_t(g) * od);
    MX_CUDA(cudaMemcpyAsync(dst.data(), mo, dst.size() * 2, cudaMemcpyDeviceToHost, c.stream));
    c.sync();
  };
  for (size_t li = 0; li < blocks_.size(); li++) {
    const Block& b = blocks_[li];
    ArenaScope s2(c.act);
    RowMod s1{b.n1.scale.as<float>(), b.n1.scale.as<float>(), 0}, t1{b.n1.bias, b.n1.bias, 0};
    layernorm_mod(x, n, dim_, eps_, s1, t1, hb, c.stream);
    bf16* qkv = c.act.alloc_n<bf16>(n * dim_ * 3);
    linear(c, b.qkv, hb, n, qkv);
    bf16* q = c.act.alloc_n<bf16>(n * dim_);
    bf16* k = c.act.alloc_n<bf16>(n * dim_);
    bf16* v = c.act.alloc_n<bf16>(n * dim_);
    head_norm_rope(qkv, n, dim_ * 3, 0, heads_, head_dim_, nullptr, 0.f, cs, sn, head_dim_, ROPE_HALF, q, c.stream);
    head_norm_rope(qkv, n, dim_ * 3, dim_, heads_, head_dim_, nullptr, 0.f, cs, sn, head_dim_, ROPE_HALF, k,
                   c.stream);
    head_norm_rope(qkv, n, dim_ * 3, 2 * dim_, heads_, head_dim_, nullptr, 0.f, nullptr, nullptr, 0, ROPE_NONE, v,
                   c.stream);
    AttnParams ap;
    ap.Sq = ap.Skv = int(n);
    ap.Hq = ap.Hkv = heads_;
    ap.D = head_dim_;
    flash_attn(q, k, v, hb, ap, c.stream);
    bf16* o = c.act.alloc_n<bf16>(n * dim_);
    linear(c, b.proj, hb, n, o);
    gated_residual(x, o, n, dim_, {}, 0, c.stream);

    RowMod s2m{b.n2.scale.as<float>(), b.n2.scale.as<float>(), 0}, t2{b.n2.bias, b.n2.bias, 0};
    layernorm_mod(x, n, dim_, eps_, s2m, t2, hb, c.stream);
    bf16* f = c.act.alloc_n<bf16>(n * b.fc1.out);
    linear(c, b.fc1, hb, n, f);
    gelu_tanh_inplace(f, n * b.fc1.out, c.stream);
    linear(c, b.fc2, f, n, o);
    gated_residual(x, o, n, dim_, {}, 0, c.stream);

    for (size_t j = 0; j < ds_layers_.size(); j++)
      if (ds_layers_[j] == int(li)) {
        merge(c, ds_[j], x, n, mo);
        out.deepstack.emplace_back();
        fetch(out.deepstack.back());
      }
  }
  merge(c, merger_, x, n, mo);
  fetch(out.embeds);
  return out;
}

}  // namespace mx
