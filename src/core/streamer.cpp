#include "core/streamer.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "core/dequant.h"

namespace mx {

Linear make_linear_streamed(DeviceWeights& dw, const std::string& prefix) {
  const WeightRef& r = dw.store().get(prefix + ".weight");
  MX_CHECK(r.shape.size() == 2, "%s.weight 不是 2 維", prefix.c_str());
  Linear l;
  l.name = prefix;
  l.w.type = r.type;
  l.w.shape = r.shape;
  l.out = r.shape[0];
  l.in = r.shape[1];
  l.bias = dw.f32_or_null(prefix + ".bias");
  return l;
}

Linear make_linear_streamed_concat(DeviceWeights& dw, const std::string& prefix,
                                   const std::string& first, const std::string& second) {
  const WeightRef& a = dw.store().get(first + ".weight");
  const WeightRef& b = dw.store().get(second + ".weight");
  MX_CHECK(a.shape.size() == 2 && b.shape.size() == 2 && a.type == b.type && a.shape[1] == b.shape[1],
           "%s：接合來源的型別或輸入維度不同", prefix.c_str());
  MX_CHECK(!dw.store().has(first + ".bias") && !dw.store().has(second + ".bias"),
           "%s：接合線性層不支援 bias", prefix.c_str());
  Linear l;
  l.name = prefix;
  l.w.type = a.type;
  l.w.shape = {a.shape[0] + b.shape[0], a.shape[1]};
  l.out = l.w.shape[0];
  l.in = l.w.shape[1];
  l.sources = {first + ".weight", second + ".weight"};
  return l;
}

BlockStreamer::~BlockStreamer() {
  if (blas_) cublasDestroy(blas_);
  if (copy_) {
    cudaStreamSynchronize(copy_);
    cudaStreamDestroy(copy_);
  }
  for (int s = 0; s < 2; s++)
    for (cudaEvent_t e : {staged_[s], copied_[s], consumed_[s]})
      if (e) cudaEventDestroy(e);
  if (res_ready_) cudaEventDestroy(res_ready_);
}

static size_t align256(size_t v) { return (v + 255) / 256 * 256; }

// 量化 Linear 在槽內的配置：[q][sf][amax]，各自 256 對齊。
static size_t qlayout(QMode m, const Linear& l, size_t& sf_off, size_t& amax_off) {
  sf_off = align256(qdata_bytes(m, l.out, l.in));
  amax_off = sf_off + align256(qscale_bytes(m, l.out, l.in));
  return amax_off + 256;
}

void BlockStreamer::add_block(const TensorStore& st, std::vector<Linear*> linears) {
  Block b;
  for (Linear* l : linears) {
    std::vector<const WeightRef*> src;
    size_t nbytes = 0;
    for (const auto& name : l->sources.empty() ? std::vector<std::string>{l->name + ".weight"} : l->sources) {
      src.push_back(&st.get(name));
      nbytes += src.back()->nbytes;
    }
    b.raw_offs.push_back(b.raw_bytes);
    b.raw_bytes += align256(nbytes);
    if (mode_ != QMode::BF16) {
      MX_CHECK(l->in % 32 == 0, "%s：輸入維度不是 32 的倍數，無法量化成 %s", l->name.c_str(), qmode_name(mode_));
      size_t sf_off, amax_off;
      b.q_offs.push_back(b.q_bytes);
      b.q_bytes += qlayout(mode_, *l, sf_off, amax_off);
      tmp_max_ = std::max(tmp_max_, size_t(l->in) * l->out * sizeof(bf16));
      l->qw = std::make_shared<QWeight>();
      l->qw->mode = mode_;
    }
    b.refs.push_back(std::move(src));
  }
  b.linears = std::move(linears);
  total_raw_ += b.raw_bytes;
  raw_max_ = std::max(raw_max_, b.raw_bytes);
  q_max_ = std::max(q_max_, b.q_bytes);
  blocks_.push_back(std::move(b));
}

void BlockStreamer::finalize() {
  MX_CHECK(!blocks_.empty(), "BlockStreamer：沒有登記任何塊");
  MX_CUDA(cudaSetDevice(dev_));
  const bool quant = mode_ != QMode::BF16;
  for (int s = 0; s < 2; s++) {
    stage_[s] = PinnedBuf(raw_max_);
    if (s == 0 || !quant) raw_[s] = DevBuf(raw_max_, dev_);
    if (quant) q_[s] = DevBuf(q_max_, dev_);
    for (cudaEvent_t* e : {&staged_[s], &copied_[s], &consumed_[s]})
      MX_CUDA(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  }
  if (quant) tmp_ = DevBuf(tmp_max_, dev_);
  MX_CUDA(cudaEventCreateWithFlags(&res_ready_, cudaEventDisableTiming));
  MX_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
  if (quant) {
    MX_BLAS(cublasCreate(&blas_));
    MX_BLAS(cublasSetStream(blas_, copy_));
  }
  const size_t dev_bytes = raw_[0].size() + raw_[1].size() + q_[0].size() + q_[1].size() + tmp_.size();
  MX_INFO("權重串流：%zu 塊、%.2f GiB 直接讀檔（%s）、鎖頁暫存 2×%.0f MiB、GPU%d 槽 %.0f MiB", blocks_.size(),
          total_raw_ / 1073741824.0, quant ? qmode_name(mode_) : "原始格式", raw_max_ / 1048576.0, dev_,
          dev_bytes / 1048576.0);
}

// 從映射檔案複製到鎖頁暫存；多執行緒讓頁面快取（或磁碟）讀取並行。
void BlockStreamer::fill_stage(const Block& b, char* dst) {
  constexpr size_t kChunk = size_t(8) << 20;
  struct Piece {
    const char* src;
    char* dst;
    size_t n;
  };
  std::vector<Piece> pieces;
  for (size_t j = 0; j < b.refs.size(); j++) {
    size_t off = b.raw_offs[j];
    for (const WeightRef* r : b.refs[j]) {
      for (size_t o = 0; o < r->nbytes; o += kChunk)
        pieces.push_back({static_cast<const char*>(r->data) + o, dst + off + o, std::min(kChunk, r->nbytes - o)});
      off += r->nbytes;
    }
  }
  const int nt = int(std::min<size_t>(4, pieces.size()));
  std::vector<std::thread> pool;
  for (int t = 1; t < nt; t++)
    pool.emplace_back([&, t] {
      for (size_t i = t; i < pieces.size(); i += nt) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
    });
  for (size_t i = 0; i < pieces.size(); i += nt) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
  for (auto& th : pool) th.join();
}

void BlockStreamer::load(int i, int stage, char* raw, char* q) {
  const Block& b = blocks_[i];
  // 暫存區上一次的 H2D 完成後才能覆寫（通常早就完成）。
  MX_CUDA(cudaEventSynchronize(staged_[stage]));
  fill_stage(b, static_cast<char*>(stage_[stage].get()));
  MX_CUDA(cudaMemcpyAsync(raw, stage_[stage].get(), b.raw_bytes, cudaMemcpyHostToDevice, copy_));
  MX_CUDA(cudaEventRecord(staged_[stage], copy_));
  if (mode_ == QMode::BF16) return;
  // 原始權重 → bf16 → 區塊縮放；與 c.stream 上的計算重疊。
  for (size_t j = 0; j < b.linears.size(); j++) {
    const Linear& l = *b.linears[j];
    dequant_to_bf16(raw + b.raw_offs[j], l.w.type, l.in * l.out, tmp_.as<bf16>(), copy_);
    const float one = 1.f;
    for (auto& d : l.loras) {  // W[out,in] += s·B·A（同 DeviceWeights::merge_loras）
      if (d->f32) continue;
      MX_BLAS(cublasGemmEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, int(l.in), int(d->out_len), int(d->rank), &d->scale,
                           d->A.get(), CUDA_R_16BF, int(l.in), d->B.get(), CUDA_R_16BF, int(d->rank), &one,
                           tmp_.as<bf16>() + d->out_off * l.in, CUDA_R_16BF, int(l.in), CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT));
    }
    size_t sf_off, amax_off;
    qlayout(mode_, l, sf_off, amax_off);
    char* base = q + b.q_offs[j];
    quantize_blockscaled(mode_, tmp_.as<bf16>(), l.in, l.out, l.in,
                         {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)},
                         copy_);
  }
}

void BlockStreamer::issue_copy(int i) {
  const int s = slot(i);
  if (slot_block_[s] == i) return;
  // 等這個槽的上一個使用者算完再覆寫（量化模式的 raw_[0] 只是 copy stream 上的轉換來源）。
  MX_CUDA(cudaStreamWaitEvent(copy_, consumed_[s], 0));
  const bool quant = mode_ != QMode::BF16;
  load(i, s, static_cast<char*>(raw_[quant ? 0 : s].get()), quant ? static_cast<char*>(q_[s].get()) : nullptr);
  MX_CUDA(cudaEventRecord(copied_[s], copy_));
  slot_block_[s] = i;
}

int BlockStreamer::set_resident(size_t bytes_budget) {
  invalidate();
  res_.clear();
  int n = 0;
  for (size_t used = 0; n < blocks() && used + dev_bytes(blocks_[n]) <= bytes_budget; n++) used += dev_bytes(blocks_[n]);
  if (n > 0 && (blocks() - n) % 2) n--;
  for (int i = 0; i < n; i++) res_.emplace_back(dev_bytes(blocks_[i]), dev_);
  res_ok_.assign(n, 0);
  n_res_ = n;
  last_ = -1;
  return n;
}

void BlockStreamer::invalidate() {
  if (copy_) MX_CUDA(cudaStreamSynchronize(copy_));
  slot_block_[0] = slot_block_[1] = -1;
  std::fill(res_ok_.begin(), res_ok_.end(), 0);
}

void BlockStreamer::acquire(Ctx& c, int i) {
  MX_CHECK(i >= 0 && i < blocks(), "BlockStreamer：塊索引 %d 超出範圍", i);
  MX_CHECK(c.device == dev_, "BlockStreamer：Ctx 裝置不符");
  if (last_ >= n_res_) MX_CUDA(cudaEventRecord(consumed_[slot(last_)], c.stream));
  const bool quant = mode_ != QMode::BF16;
  char *raw, *q;
  if (i < n_res_) {
    raw = q = static_cast<char*>(res_[i].get());
    if (!res_ok_[i]) {
      load(i, i & 1, quant ? static_cast<char*>(raw_[0].get()) : raw, q);
      MX_CUDA(cudaEventRecord(res_ready_, copy_));
      res_ok_[i] = 1;
    }
    MX_CUDA(cudaStreamWaitEvent(c.stream, res_ready_, 0));
  } else {
    issue_copy(i);
    const int s = slot(i);
    MX_CUDA(cudaStreamWaitEvent(c.stream, copied_[s], 0));
    raw = static_cast<char*>(raw_[s].get());
    q = quant ? static_cast<char*>(q_[s].get()) : nullptr;
  }
  Block& b = blocks_[i];
  for (size_t j = 0; j < b.linears.size(); j++) {
    Linear& l = *b.linears[j];
    if (!quant) {
      l.w.p = raw + b.raw_offs[j];
    } else {
      char* base = q + b.q_offs[j];
      size_t sf_off, amax_off;
      qlayout(mode_, l, sf_off, amax_off);
      l.qw->m = {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)};
      // 槽內權重已含 LoRA（換 LoRA 時 invalidate 過，所以和目前的 l.loras 一致），linear() 不必再另算。
      bool merged = false;
      for (auto& d : l.loras) merged |= !d->f32;
      l.qw->merged = merged;
    }
  }
  last_ = i;
  // 預取下一個串流塊（常駐塊跳過）。
  int j = i + 1 < blocks() ? i + 1 : 0;
  if (j < n_res_) j = n_res_;
  if (j < blocks() && j != i) issue_copy(j);
}

}  // namespace mx
