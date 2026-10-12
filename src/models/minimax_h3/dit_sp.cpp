// MiniMax-H3 DiT 序列平行（兩張卡，Ulysses 式）：
// 列（token）對半分給兩張卡做線性層／AdaLN／MLP；注意力改依頭分工，每張卡看全部列、算一半的頭。
// 每層交換兩次：qkv 投影後把「對方的頭」送過去，注意力後把「自己列上、對方算的頭」取回來。
// 跨卡複製一律排在送出方的 stream 上，再用 event 讓接收方等待（GeForce 沒有 P2P 時驅動經主機中轉）。
#include <cuda_runtime.h>

#include "kernels/attention.h"
#include "kernels/ops.h"
#include "models/minimax_h3/dit.h"

namespace mx {

struct H3Dit::SpPart {
  H3Dit* d;
  Ctx* c;
  H3Seq* seq;
  int64_t r0, n;  // 負責的列 [r0, r0+n)
  int h0, H;      // 負責的注意力頭 [h0, h0+H)
  const float *video = nullptr, *audio = nullptr;
  float *basis = nullptr, *x = nullptr, *mods = nullptr;  // x：只有自己的列
  bf16 *qh = nullptr, *kh = nullptr, *vh = nullptr, *oh = nullptr;  // [S, H·128]：全部列、自己的頭
  cudaEvent_t sent = nullptr, attn = nullptr, done = nullptr;
};

size_t H3Dit::act_bytes_sp(int64_t S, int64_t n, int64_t chunk) const {
  const size_t D = dim_, I = inner_, F = ffn_, C = chunk > 0 ? chunk : chunk_, Ih = I / 2;
  const bool low = compute_ != QMode::BF16;
  size_t keep = size_t(n) * D * 4 + size_t(S) * Ih * 2 * (low ? 3 : 4) + size_t(4) * 18 * D * 4;
  size_t qkv = C * (D + 3 * I + 3 * Ih) * 2, mlp = C * (I + Ih + D + 3 * F) * 2;
  size_t extra = low ? std::max(std::max(qkv, mlp) + C * std::max(F, I) * 9 / 8, size_t(S + 128) * Ih * 3 + (size_t(4) << 20))
                     : std::max(qkv, mlp);
  return keep + extra + (size_t(64) << 20);
}

void H3Dit::sp_begin(SpPart& p, const std::vector<float>& ts) {
  Ctx& c = *p.c;
  H3Seq& seq = *p.seq;
  const int64_t D = dim_, nt = int64_t(ts.size());
  c.use();
  p.basis = c.act.alloc_n<float>(nt * basis_);
  time_basis(c, ts, p.basis);
  p.x = c.act.alloc_n<float>(p.n * D);
  p.mods = c.act.alloc_n<float>(nt * 18 * D);
  // 輸入列只算落在自己範圍內的部分（off：該段內的起點）。
  auto put = [&](int64_t start, int64_t len, auto&& fill) {
    int64_t s0 = std::max(start, p.r0), s1 = std::min(start + len, p.r0 + p.n);
    if (s0 < s1) fill(s0 - start, s1 - s0, p.x + (s0 - p.r0) * D);
  };
  put(seq.text_start, seq.text_len, [&](int64_t off, int64_t len, float* dst) {
    MX_CUDA(cudaMemcpyAsync(dst, seq.text.as<float>() + off * D, size_t(len) * D * 4, cudaMemcpyDeviceToDevice,
                            c.stream));
  });
  int64_t vo = 0, ao = 0;
  for (auto& b : seq.vref) {
    put(b.start, b.len, [&](int64_t off, int64_t len, float* dst) {
      linear_f32(c, video_in_, p.video + (vo + off) * video_in_.in, len, dst);
    });
    vo += b.len;
  }
  for (auto& b : seq.aref) {
    put(b.start, b.len, [&](int64_t off, int64_t len, float* dst) {
      linear_f32(c, audio_in_, p.audio + (ao + off) * audio_in_.in, len, dst);
    });
    ao += b.len;
  }
  put(seq.video_start, seq.video_len, [&](int64_t off, int64_t len, float* dst) {
    linear_f32(c, video_in_, p.video + (vo + off) * video_in_.in, len, dst);
  });
  put(seq.audio_start, seq.audio_len, [&](int64_t off, int64_t len, float* dst) {
    linear_f32(c, audio_in_, p.audio + (ao + off) * audio_in_.in, len, dst);
  });
}

void H3Dit::sp_qkv(int i, SpPart& p, const SpPart& o) {
  Ctx& c = *p.c;
  const Block& b = blocks_[i];
  const int64_t D = dim_, I = inner_, hd = head_dim_, Ip = p.H * hd, Io = o.H * hd, ms = 18 * D;
  c.use();
  st_blocks_->acquire(c, i);
  linear_f32(c, b.adaln, p.basis, 4, p.mods);
  const float* cs = p.seq->cos.as<float>();
  const float* sn = p.seq->sin.as<float>();
  for (int64_t g0 = p.r0; g0 < p.r0 + p.n; g0 += chunk_) {
    ArenaScope s2(c.act);
    const int64_t n = std::min(chunk_, p.r0 + p.n - g0);
    bf16* h = c.act.alloc_n<bf16>(n * D);
    bf16* t = c.act.alloc_n<bf16>(n * 3 * I);
    bf16* snd[3] = {c.act.alloc_n<bf16>(n * Io), c.act.alloc_n<bf16>(n * Io), c.act.alloc_n<bf16>(n * Io)};
    h3_for_rows(p.seq->segs, g0, n, [&](const H3Segment& g, int64_t s0, int64_t len) {
      const float* m = p.mods + g.t_idx * ms + int64_t(g.tag) * 6 * D;
      rmsnorm_mod(p.x + (s0 - p.r0) * D, len, D, eps_, b.norm1, {m + D, m + D, 0}, {m, m, 0}, h + (s0 - g0) * D,
                  c.stream);
    });
    linear(c, b.qkv, h, n, t);
    const float* cr = cs + g0 * (rot_ / 2);
    const float* sr = sn + g0 * (rot_ / 2);
    // 自己的頭寫進 qh/kh/vh；對方的頭寫進傳送暫存再送過去。
    bf16* mine[3] = {p.qh + g0 * Ip, p.kh + g0 * Ip, p.vh + g0 * Ip};
    const float* norms[3] = {b.q_norm, b.k_norm, nullptr};
    for (int k = 0; k < 3; k++) {
      const bool rope = k < 2;
      head_norm_rope(t, n, 3 * I, k * I + p.h0 * hd, p.H, int(hd), norms[k], rope ? eps_ : 0.f, rope ? cr : nullptr,
                     rope ? sr : nullptr, rope ? rot_ : 0, rope ? ROPE_HALF : ROPE_NONE, mine[k], c.stream);
      head_norm_rope(t, n, 3 * I, k * I + o.h0 * hd, o.H, int(hd), norms[k], rope ? eps_ : 0.f, rope ? cr : nullptr,
                     rope ? sr : nullptr, rope ? rot_ : 0, rope ? ROPE_HALF : ROPE_NONE, snd[k], c.stream);
    }
    bf16* dst[3] = {o.qh, o.kh, o.vh};
    for (int k = 0; k < 3; k++)
      MX_CUDA(cudaMemcpyPeerAsync(dst[k] + g0 * Io, o.c->device, snd[k], c.device, size_t(n) * Io * 2, c.stream));
  }
  MX_CUDA(cudaEventRecord(p.sent, c.stream));
}

void H3Dit::sp_attn(SpPart& p, const SpPart& o) {
  Ctx& c = *p.c;
  c.use();
  MX_CUDA(cudaStreamWaitEvent(c.stream, o.sent, 0));
  AttnParams ap;
  ap.Sq = ap.Skv = int(p.seq->S);
  ap.Hq = ap.Hkv = p.H;
  ap.D = head_dim_;
  if (p.oh == p.qh) {  // FP8 路徑：q/k/v 先量化進工作區，輸出覆寫 q
    ArenaScope s3(c.act);
    flash_attn_fp8(p.qh, p.kh, p.vh, p.oh, ap, c.act.alloc(flash_attn_fp8_workspace(ap)), c.stream);
  } else {
    flash_attn(p.qh, p.kh, p.vh, p.oh, ap, c.stream);
  }
  MX_CUDA(cudaEventRecord(p.attn, c.stream));
}

void H3Dit::sp_mlp(int i, SpPart& p, const SpPart& o) {
  Ctx& c = *p.c;
  const Block& b = blocks_[i];
  const int64_t D = dim_, I = inner_, F = ffn_, hd = head_dim_, Ip = p.H * hd, Io = o.H * hd, ms = 18 * D;
  c.use();
  MX_CUDA(cudaStreamWaitEvent(c.stream, o.attn, 0));
  for (int64_t g0 = p.r0; g0 < p.r0 + p.n; g0 += chunk_) {
    ArenaScope s2(c.act);
    const int64_t n = std::min(chunk_, p.r0 + p.n - g0);
    bf16* oc = c.act.alloc_n<bf16>(n * I);
    bf16* rc = c.act.alloc_n<bf16>(n * Io);
    bf16* h = c.act.alloc_n<bf16>(n * D);
    bf16* gu = c.act.alloc_n<bf16>(n * 2 * F);
    bf16* mid = c.act.alloc_n<bf16>(n * F);
    // 組回自己列上的完整注意力輸出 [n, I]：自己的頭在本卡，對方的頭從對方取回。
    MX_CUDA(cudaMemcpyPeerAsync(rc, c.device, o.oh + g0 * Io, o.c->device, size_t(n) * Io * 2, c.stream));
    MX_CUDA(cudaMemcpy2DAsync(oc + p.h0 * hd, I * 2, p.oh + g0 * Ip, Ip * 2, Ip * 2, n, cudaMemcpyDeviceToDevice,
                              c.stream));
    MX_CUDA(cudaMemcpy2DAsync(oc + o.h0 * hd, I * 2, rc, Io * 2, Io * 2, n, cudaMemcpyDeviceToDevice, c.stream));
    linear(c, b.out, oc, n, h);
    h3_for_rows(p.seq->segs, g0, n, [&](const H3Segment& g, int64_t s0, int64_t len) {
      const float* m = p.mods + g.t_idx * ms + int64_t(g.tag) * 6 * D;
      float* xr = p.x + (s0 - p.r0) * D;
      gated_residual(xr, h + (s0 - g0) * D, len, D, {m + 2 * D, m + 2 * D, 0}, 0, c.stream);
      rmsnorm_mod(xr, len, D, eps_, b.norm2, {m + 4 * D, m + 4 * D, 0}, {m + 3 * D, m + 3 * D, 0},
                  h + (s0 - g0) * D, c.stream);
    });
    linear(c, b.fc1, h, n, gu);
    silu_mul(gu, n, F, mid, c.stream);
    linear(c, b.fc2, mid, n, h);
    h3_for_rows(p.seq->segs, g0, n, [&](const H3Segment& g, int64_t s0, int64_t len) {
      const float* m = p.mods + g.t_idx * ms + int64_t(g.tag) * 6 * D;
      gated_residual(p.x + (s0 - p.r0) * D, h + (s0 - g0) * D, len, D, {m + 5 * D, m + 5 * D, 0}, 0, c.stream);
    });
  }
}

void H3Dit::sp_head(SpPart& p, const SpPart& a, float* v_video, float* v_audio) {
  Ctx& c = *p.c;
  H3Seq& seq = *p.seq;
  const int64_t D = dim_;
  c.use();
  ArenaScope sc(c.act);
  float* fm = c.act.alloc_n<float>(4 * 2 * D);
  linear_f32(c, final_adaln_, p.basis, 4, fm);
  auto head = [&](int64_t start, int64_t len, const Linear& out, float* dst) {
    const int64_t s0 = std::max(start, p.r0), s1 = std::min(start + len, p.r0 + p.n), n = s1 - s0;
    if (n <= 0) return;
    const H3Segment* g = nullptr;
    for (auto& s : seq.segs)
      if (s0 >= s.start && s0 < s.start + s.len) g = &s;
    MX_CHECK(g && s1 <= g->start + g->len, "輸出列跨越多個段");
    const float* row = fm + g->t_idx * 2 * D;
    ArenaScope s2(c.act);
    bf16* hb = c.act.alloc_n<bf16>(n * D);
    float* hf = c.act.alloc_n<float>(n * D);
    rmsnorm_mod(p.x + (s0 - p.r0) * D, n, D, eps_, final_norm_, {row + D, row + D, 0}, {row, row, 0}, hb, c.stream);
    bf16_to_f32(hb, hf, n * D, c.stream);
    float* out_rows = dst + (s0 - start) * out.out;
    if (p.c == a.c) return linear_f32(c, out, hf, n, out_rows);
    float* tmp = c.act.alloc_n<float>(n * out.out);
    linear_f32(c, out, hf, n, tmp);
    MX_CUDA(cudaMemcpyPeerAsync(out_rows, a.c->device, tmp, c.device, size_t(n) * out.out * 4, c.stream));
  };
  head(seq.video_start, seq.video_len, video_out_, v_video);
  head(seq.audio_start, seq.audio_len, audio_out_, v_audio);
  MX_CUDA(cudaEventRecord(p.done, c.stream));
}

void H3Dit::forward_sp(H3Dit& a, Ctx& ca, H3Seq& sa, H3Dit& b, Ctx& cb, H3Seq& sb, const float* video,
                       const float* audio, const std::vector<float>& timesteps, float* v_video, float* v_audio) {
  MX_CHECK(timesteps.size() == 4, "序列平行需要 4 個 timestep");
  MX_CHECK(sa.S == sb.S && a.blocks_.size() == b.blocks_.size() && a.heads_ == b.heads_ && a.heads_ % 2 == 0 &&
               a.dim_ == b.dim_ && a.chunk_ == b.chunk_,
           "序列平行兩張卡的模型或版面不一致");
  const int64_t S = sa.S, half = S / 2, Hh = a.heads_ / 2;
  SpPart P[2] = {{&a, &ca, &sa, 0, half, 0, int(Hh)}, {&b, &cb, &sb, half, S - half, int(Hh), int(Hh)}};
  ArenaScope sca(ca.act), scb(cb.act);
  cudaEvent_t in_ready = nullptr;
  auto destroy = [&] {  // 事件在排隊的工作完成後才真正釋放，不必先同步。
    for (cudaEvent_t e : {P[0].sent, P[0].attn, P[0].done, P[1].sent, P[1].attn, P[1].done, in_ready})
      if (e) cudaEventDestroy(e);
  };
  for (auto& p : P) {
    p.c->use();
    for (cudaEvent_t* e : {&p.sent, &p.attn, &p.done}) MX_CUDA(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  }
  ca.use();
  MX_CUDA(cudaEventCreateWithFlags(&in_ready, cudaEventDisableTiming));
  try {
    // b 的輸入副本：a 的 stream 上送過去（a 的 stream 已等過上一步 b 的 done，所以 b 不會還在讀舊的）。
    const size_t vbytes = size_t(sa.vref_len() + sa.video_len) * a.video_in_.in * 4;
    const size_t abytes = size_t(sa.aref_len() + sa.audio_len) * a.audio_in_.in * 4;
    float* vb = cb.act.alloc_n<float>(vbytes / 4);
    float* ab = cb.act.alloc_n<float>(abytes / 4);
    P[0].video = video, P[0].audio = audio, P[1].video = vb, P[1].audio = ab;
    MX_CUDA(cudaMemcpyPeerAsync(vb, cb.device, video, ca.device, vbytes, ca.stream));
    MX_CUDA(cudaMemcpyPeerAsync(ab, cb.device, audio, ca.device, abytes, ca.stream));
    MX_CUDA(cudaEventRecord(in_ready, ca.stream));
    cb.use();
    MX_CUDA(cudaStreamWaitEvent(cb.stream, in_ready, 0));

    for (auto& p : P) p.d->sp_begin(p, timesteps);
    const size_t mark[2] = {ca.act.mark(), cb.act.mark()};
    for (auto& p : P) {
      p.c->use();
      const size_t cols = size_t(S) * p.H * a.head_dim_;
      p.qh = p.c->act.alloc_n<bf16>(cols);
      p.kh = p.c->act.alloc_n<bf16>(cols);
      p.vh = p.c->act.alloc_n<bf16>(cols);
      AttnParams ap;
      ap.Sq = ap.Skv = int(S), ap.Hq = ap.Hkv = p.H, ap.D = a.head_dim_;
      const bool fp8 = p.d->compute_ != QMode::BF16 && flash_attn_fp8_ok(p.qh, p.kh, p.vh, p.qh, ap);
      p.oh = fp8 ? p.qh : p.c->act.alloc_n<bf16>(cols);
    }
    for (int i = 0; i < int(a.blocks_.size()); i++) {
      a.sp_qkv(i, P[0], P[1]);
      b.sp_qkv(i, P[1], P[0]);
      a.sp_attn(P[0], P[1]);
      b.sp_attn(P[1], P[0]);
      a.sp_mlp(i, P[0], P[1]);
      b.sp_mlp(i, P[1], P[0]);
    }
    ca.act.release(mark[0]);
    cb.act.release(mark[1]);
    b.sp_head(P[1], P[0], v_video, v_audio);
    a.sp_head(P[0], P[0], v_video, v_audio);
    ca.use();
    MX_CUDA(cudaStreamWaitEvent(ca.stream, P[1].done, 0));
  } catch (...) {
    // Arena 即將釋放，先等兩張卡上已排的工作結束（清理路徑，錯誤以原例外為準）。
    cudaSetDevice(ca.device);
    cudaStreamSynchronize(ca.stream);
    cudaSetDevice(cb.device);
    cudaStreamSynchronize(cb.stream);
    destroy();
    throw;
  }
  destroy();
}

}  // namespace mx
