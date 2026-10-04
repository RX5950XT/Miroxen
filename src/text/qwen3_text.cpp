#include "text/qwen3_text.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/dequant.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

namespace mx {

Qwen3TextEncoder::Qwen3TextEncoder(const TensorStore& st, int device) : st_(st), dw_(st, device) {
  std::string arch = st.meta_str("general.architecture", "qwen3vl");
  auto key = [&](const char* k) { return arch + "." + k; };
  int n_layers = int(st.meta_int(key("block_count"), 0));
  dim_ = st.meta_int(key("embedding_length"), 0);
  ffn_ = st.meta_int(key("feed_forward_length"), 0);
  heads_ = int(st.meta_int(key("attention.head_count"), 0));
  kv_heads_ = int(st.meta_int(key("attention.head_count_kv"), heads_));
  head_dim_ = int(st.meta_int(key("attention.key_length"), heads_ ? dim_ / heads_ : 0));
  if (auto* v = st.meta(key("attention.layer_norm_rms_epsilon"))) eps_ = float(std::get<double>(*v));
  if (auto* v = st.meta(key("rope.freq_base"))) theta_ = float(std::get<double>(*v));
  // 截短過的檔案（H3）block_count 可能沒改，以實際存在的層數為準。
  int present = 0;
  while (st.has("blk." + std::to_string(present) + ".attn_q.weight")) present++;
  MX_CHECK(present > 0 && dim_ > 0 && heads_ > 0, "文字編碼器 GGUF 缺少必要欄位（arch=%s）",
           arch.c_str());
  if (n_layers != present) MX_INFO("文字編碼器：metadata 寫 %d 層，實際 %d 層", n_layers, present);
  for (int i = 0; i < present; i++) {
    std::string p = "blk." + std::to_string(i) + ".";
    Block b;
    b.q = make_linear(dw_, p + "attn_q");
    b.k = make_linear(dw_, p + "attn_k");
    b.v = make_linear(dw_, p + "attn_v");
    b.o = make_linear(dw_, p + "attn_output");
    b.gate = make_linear(dw_, p + "ffn_gate");
    b.up = make_linear(dw_, p + "ffn_up");
    b.down = make_linear(dw_, p + "ffn_down");
    b.attn_norm = dw_.f32(p + "attn_norm.weight");
    b.ffn_norm = dw_.f32(p + "ffn_norm.weight");
    b.q_norm = dw_.f32(p + "attn_q_norm.weight");
    b.k_norm = dw_.f32(p + "attn_k_norm.weight");
    blocks_.push_back(b);
  }
  if (!ffn_) ffn_ = blocks_[0].gate.out;
  MX_INFO("文字編碼器：%d 層、dim %lld、頭 %d/%d×%d，權重 %.2f GiB", present, (long long)dim_,
          heads_, kv_heads_, head_dim_, dw_.bytes() / 1073741824.0);
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
  const WeightRef& e = st_.get("token_embd.weight");
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

void Qwen3TextEncoder::encode(Ctx& c, const std::vector<int32_t>& ids, bf16* out) {
  const int64_t n = int64_t(ids.size());
  MX_CHECK(n > 0, "文字編碼器：輸入為空");
  c.use();
  ArenaScope sc(c.act);
  float* x = c.act.alloc_n<float>(n * dim_);
  embed(c, ids, x);
  // 純文字時 mRoPE 三軸位置相同，等同一般 RoPE（rotate-half）。
  const int half = head_dim_ / 2;
  std::vector<float> hc(n * half), hs(n * half);
  for (int64_t p = 0; p < n; p++)
    for (int j = 0; j < half; j++) {
      double ang = double(p) * std::pow(double(theta_), -2.0 * j / head_dim_);
      hc[p * half + j] = float(std::cos(ang));
      hs[p * half + j] = float(std::sin(ang));
    }
  float* cs = c.act.alloc_n<float>(n * half);
  float* sn = c.act.alloc_n<float>(n * half);
  MX_CUDA(cudaMemcpyAsync(cs, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice, c.stream));
  MX_CUDA(cudaMemcpyAsync(sn, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice, c.stream));
  for (auto& b : blocks_) block(c, b, x, n, cs, sn);
  f32_to_bf16(x, out, n * dim_, c.stream);
  c.sync();
}

}  // namespace mx
