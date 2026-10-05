#include "text/qwen3_text.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/dequant.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

namespace mx {

// 支援兩種命名：llama.cpp（blk.N.attn_q）與 HuggingFace（model.layers.N.self_attn.q_proj，
// 例如 MiniMax-H3 的 32B 編碼器 GGUF，沒有 metadata，超參數由張量形狀推導）。
static const char* kLlama[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
                               "attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"};
static const char* kHf[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm",
                            "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"};

Qwen3TextEncoder::Qwen3TextEncoder(const TensorStore& st, int device, bool stream)
    : st_(st), dw_(st, device) {
  std::string layer_fmt;  // 每層前綴，%d 為層號
  const char** names = kLlama;
  if (st.has("blk.0.attn_q.weight")) {
    layer_fmt = "blk.%d.";
    embed_name_ = "token_embd.weight";
  } else {
    for (const char* pre : {"model.language_model.", "model.", "language_model.model."})
      if (st.has(std::string(pre) + "layers.0.self_attn.q_proj.weight")) {
        layer_fmt = std::string(pre) + "layers.%d.";
        embed_name_ = std::string(pre) + "embed_tokens.weight";
        names = kHf;
        break;
      }
  }
  MX_CHECK(!layer_fmt.empty(), "無法辨識文字編碼器的權重命名（需要 llama.cpp 或 HuggingFace 格式）");
  auto lname = [&](int i, int k) { return strf(layer_fmt.c_str(), i) + names[k]; };
  int present = 0;
  while (st.has(lname(present, 0) + ".weight")) present++;
  const WeightRef& emb = st.get(embed_name_);
  MX_CHECK(emb.shape.size() == 2, "embedding 形狀錯誤");
  dim_ = emb.shape[1];
  head_dim_ = int(st.get(lname(0, 9) + ".weight").numel());
  heads_ = int(st.get(lname(0, 0) + ".weight").shape[0] / head_dim_);
  kv_heads_ = int(st.get(lname(0, 1) + ".weight").shape[0] / head_dim_);
  std::string arch = st.meta_str("general.architecture", "qwen3vl");
  if (auto* v = st.meta(arch + ".attention.layer_norm_rms_epsilon")) eps_ = float(std::get<double>(*v));
  if (auto* v = st.meta(arch + ".rope.freq_base")) theta_ = float(std::get<double>(*v));
  auto mk = [&](const std::string& n) { return stream ? make_linear_streamed(dw_, n) : make_linear(dw_, n); };
  blocks_.reserve(present);  // streamer 持有 Linear 指標，不可重新配置
  for (int i = 0; i < present; i++) {
    Block b;
    b.q = mk(lname(i, 0));
    b.k = mk(lname(i, 1));
    b.v = mk(lname(i, 2));
    b.o = mk(lname(i, 3));
    b.gate = mk(lname(i, 4));
    b.up = mk(lname(i, 5));
    b.down = mk(lname(i, 6));
    b.attn_norm = dw_.f32(lname(i, 7) + ".weight");
    b.ffn_norm = dw_.f32(lname(i, 8) + ".weight");
    b.q_norm = dw_.f32(lname(i, 9) + ".weight");
    b.k_norm = dw_.f32(lname(i, 10) + ".weight");
    blocks_.push_back(b);
  }
  ffn_ = blocks_[0].gate.out;
  if (stream) {
    streamer_ = std::make_unique<BlockStreamer>(device);
    for (auto& b : blocks_) streamer_->add_block(st, {&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down});
    streamer_->finalize();
  }
  MX_INFO("文字編碼器：%d 層、dim %lld、頭 %d/%d×%d，權重 %.2f GiB%s", present, (long long)dim_,
          heads_, kv_heads_, head_dim_, weight_bytes() / 1073741824.0, stream ? "（串流）" : "");
}

size_t Qwen3TextEncoder::scratch_bytes() const {
  size_t m = 0;
  for (auto& b : blocks_)
    for (const Linear* l : {&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down})
      m = std::max(m, size_t(l->in) * l->out * sizeof(bf16));
  return m;
}

// 只取需要的列：在主機端從映射檔案收集量化列，再上傳反量化。
void Qwen3TextEncoder::embed(Ctx& c, const std::vector<int32_t>& ids, float* x) {
  const WeightRef& e = st_.get(embed_name_);
  MX_CHECK(e.shape.size() == 2 && e.shape[1] == dim_, "token_embd 形狀錯誤");
  size_t row = gtype_row_bytes(e.type, dim_);
  std::vector<uint8_t> host(row * ids.size());
  for (size_t i = 0; i < ids.size(); i++) {
    MX_CHECK(ids[i] >= 0 && ids[i] < e.shape[0], "token id %d 超出詞彙表", ids[i]);
    std::memcpy(host.data() + i * row, static_cast<const uint8_t*>(e.data) + size_t(ids[i]) * row,
                row);
  }
  ArenaScope sc(c.act);
  void* d = c.act.alloc(host.size());
  MX_CUDA(cudaMemcpyAsync(d, host.data(), host.size(), cudaMemcpyHostToDevice, c.stream));
  dequant_to_f32(d, e.type, int64_t(ids.size()) * dim_, x, c.stream);
  c.sync();  // host 緩衝區即將釋放
}

void Qwen3TextEncoder::block(Ctx& c, const Block& b, float* x, int64_t n, const float* cs,
                             const float* sn) {
  ArenaScope sc(c.act);
  const int64_t qd = int64_t(heads_) * head_dim_, kd = int64_t(kv_heads_) * head_dim_;
  bf16* h = c.act.alloc_n<bf16>(n * dim_);
  rmsnorm_mod(x, n, dim_, eps_, b.attn_norm, {}, {}, h, c.stream);
  bf16* q = c.act.alloc_n<bf16>(n * qd);
  bf16* k = c.act.alloc_n<bf16>(n * kd);
  bf16* v = c.act.alloc_n<bf16>(n * kd);
  bf16* t = c.act.alloc_n<bf16>(n * std::max(qd, kd));
  linear(c, b.q, h, n, t);
  head_norm_rope(t, n, qd, 0, heads_, head_dim_, b.q_norm, eps_, cs, sn, head_dim_, ROPE_HALF, q,
                 c.stream);
  linear(c, b.k, h, n, t);
  head_norm_rope(t, n, kd, 0, kv_heads_, head_dim_, b.k_norm, eps_, cs, sn, head_dim_, ROPE_HALF, k,
                 c.stream);
  linear(c, b.v, h, n, v);
  AttnParams ap;
  ap.Sq = ap.Skv = int(n);
  ap.Hq = heads_;
  ap.Hkv = kv_heads_;
  ap.D = head_dim_;
  ap.causal = 1;
  flash_attn(q, k, v, t, ap, c.stream);
  linear(c, b.o, t, n, h);
  gated_residual(x, h, n, dim_, {}, 0, c.stream);

  rmsnorm_mod(x, n, dim_, eps_, b.ffn_norm, {}, {}, h, c.stream);
  bf16* gu = c.act.alloc_n<bf16>(n * 2 * ffn_);
  linear(c, b.gate, h, n, gu, 2 * ffn_);
  linear(c, b.up, h, n, gu + ffn_, 2 * ffn_);
  bf16* m = c.act.alloc_n<bf16>(n * ffn_);
  silu_mul(gu, n, ffn_, m, c.stream);
  linear(c, b.down, m, n, h);
  gated_residual(x, h, n, dim_, {}, 0, c.stream);
}

TextVisionInput build_text_vision(const std::vector<int32_t>& ids, int32_t image_pad_id,
                                  const std::vector<const VisionFeatures*>& imgs) {
  TextVisionInput v;
  const int64_t n = int64_t(ids.size());
  v.pos.resize(n * 3);
  int32_t cur = 0;
  size_t img = 0;
  for (int64_t i = 0; i < n;) {
    if (ids[i] != image_pad_id) {
      for (int a = 0; a < 3; a++) v.pos[i * 3 + a] = cur;
      cur++, i++;
      continue;
    }
    int64_t j = i;
    while (j < n && ids[j] == image_pad_id) j++;
    MX_CHECK(img < imgs.size(), "<|image_pad|> 段數多於圖片數");
    const VisionFeatures& f = *imgs[img++];
    MX_CHECK(j - i == f.tokens(), "第 %zu 張圖的佔位數 %lld 與特徵數 %lld 不符", img, (long long)(j - i),
             (long long)f.tokens());
    for (int64_t k = 0; k < f.tokens(); k++) {
      int64_t r = i + k;
      v.pos[r * 3] = cur;
      v.pos[r * 3 + 1] = cur + int32_t(k / f.gw);
      v.pos[r * 3 + 2] = cur + int32_t(k % f.gw);
      v.rows.push_back(int32_t(r));
    }
    v.embeds.insert(v.embeds.end(), f.embeds.begin(), f.embeds.end());
    if (v.deepstack.size() < f.deepstack.size()) v.deepstack.resize(f.deepstack.size());
    for (size_t d = 0; d < f.deepstack.size(); d++)
      v.deepstack[d].insert(v.deepstack[d].end(), f.deepstack[d].begin(), f.deepstack[d].end());
    cur += int32_t(std::max(f.gh, f.gw));
    i = j;
  }
  MX_CHECK(img == imgs.size(), "圖片數 %zu 多於 <|image_pad|> 段數 %zu", imgs.size(), img);
  return v;
}

void Qwen3TextEncoder::encode(Ctx& c, const std::vector<int32_t>& ids, bf16* out, const TextVisionInput* vis) {
  const int64_t n = int64_t(ids.size());
  MX_CHECK(n > 0, "文字編碼器：輸入為空");
  c.use();
  ArenaScope sc(c.act);
  float* x = c.act.alloc_n<float>(n * dim_);
  embed(c, ids, x);
  // 交錯式 mRoPE（mrope_section [24, 20, 20]）：頻率 j 在 j%3==1 用 h、j%3==2 用 w（j < 60），其餘用 t。
  // 純文字時三軸位置相同，等同一般 RoPE（rotate-half）。
  const int half = head_dim_ / 2;
  std::vector<float> hc(n * half), hs(n * half);
  for (int64_t p = 0; p < n; p++)
    for (int j = 0; j < half; j++) {
      int axis = j < 60 && j % 3 == 1 ? 1 : j < 60 && j % 3 == 2 ? 2 : 0;
      double pos = vis ? double(vis->pos[p * 3 + axis]) : double(p);
      double ang = pos * std::pow(double(theta_), -2.0 * j / head_dim_);
      hc[p * half + j] = float(std::cos(ang));
      hs[p * half + j] = float(std::sin(ang));
    }
  float* cs = c.act.alloc_n<float>(n * half);
  float* sn = c.act.alloc_n<float>(n * half);
  MX_CUDA(cudaMemcpyAsync(cs, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice, c.stream));
  MX_CUDA(cudaMemcpyAsync(sn, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice, c.stream));
  const int64_t nv = vis ? int64_t(vis->rows.size()) : 0;
  int32_t* rows = nullptr;
  std::vector<bf16*> ds;
  if (nv) {
    MX_CHECK(vis->pos.size() == size_t(n) * 3 && vis->embeds.size() == size_t(nv * dim_), "圖文輸入大小不符");
    rows = c.act.alloc_n<int32_t>(nv);
    bf16* e = c.act.alloc_n<bf16>(nv * dim_);
    MX_CUDA(cudaMemcpyAsync(rows, vis->rows.data(), nv * 4, cudaMemcpyHostToDevice, c.stream));
    MX_CUDA(cudaMemcpyAsync(e, vis->embeds.data(), nv * dim_ * 2, cudaMemcpyHostToDevice, c.stream));
    scatter_rows(x, dim_, rows, e, nv, 0, c.stream);
    for (auto& d : vis->deepstack) {
      MX_CHECK(d.size() == size_t(nv * dim_), "DeepStack 特徵大小不符");
      ds.push_back(c.act.alloc_n<bf16>(nv * dim_));
      MX_CUDA(cudaMemcpyAsync(ds.back(), d.data(), nv * dim_ * 2, cudaMemcpyHostToDevice, c.stream));
    }
  }
  for (size_t i = 0; i < blocks_.size(); i++) {
    if (streamer_) streamer_->acquire(c, int(i));
    block(c, blocks_[i], x, n, cs, sn);
    if (i < ds.size()) scatter_rows(x, dim_, rows, ds[i], nv, 1, c.stream);
  }
  f32_to_bf16(x, out, n * dim_, c.stream);
  c.sync();
}

}  // namespace mx
