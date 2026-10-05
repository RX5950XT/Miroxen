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

BlockStreamer::~BlockStreamer() {
  if (copy_) {
    cudaStreamSynchronize(copy_);
    cudaStreamDestroy(copy_);
  }
  for (int s = 0; s < 2; s++)
    for (cudaEvent_t e : {staged_[s], copied_[s], consumed_[s]})
      if (e) cudaEventDestroy(e);
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
    const WeightRef& r = st.get(l->name + ".weight");
    b.raw_offs.push_back(b.raw_bytes);
    b.raw_bytes += align256(r.nbytes);
    if (mode_ != QMode::BF16) {
      MX_CHECK(l->in % 32 == 0, "%s：輸入維度不是 32 的倍數，無法量化成 %s", l->name.c_str(), qmode_name(mode_));
      size_t sf_off, amax_off;
      b.q_offs.push_back(b.q_bytes);
      b.q_bytes += qlayout(mode_, *l, sf_off, amax_off);
      tmp_max_ = std::max(tmp_max_, size_t(l->in) * l->out * sizeof(bf16));
      l->qw = std::make_shared<QWeight>();
      l->qw->mode = mode_;
    }
    b.refs.push_back(&r);
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
  MX_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
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
  for (size_t j = 0; j < b.refs.size(); j++)
    for (size_t o = 0; o < b.refs[j]->nbytes; o += kChunk)
      pieces.push_back({static_cast<const char*>(b.refs[j]->data) + o, dst + b.raw_offs[j] + o,
                        std::min(kChunk, b.refs[j]->nbytes - o)});
  const int nt = int(std::min<size_t>(4, pieces.size()));
  std::vector<std::thread> pool;
  for (int t = 1; t < nt; t++)
    pool.emplace_back([&, t] {
      for (size_t i = t; i < pieces.size(); i += nt) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
    });
  for (size_t i = 0; i < pieces.size(); i += nt) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
  for (auto& th : pool) th.join();
}

void BlockStreamer::issue_copy(int i) {
  const int s = i % 2;
  if (slot_block_[s] == i) return;
  const Block& b = blocks_[i];
  // 暫存區 s 上一次的 H2D 完成後才能覆寫（通常早就完成）。
  MX_CUDA(cudaEventSynchronize(staged_[s]));
  fill_stage(b, static_cast<char*>(stage_[s].get()));
  // 等這個槽的上一個使用者算完再覆寫。
  MX_CUDA(cudaStreamWaitEvent(copy_, consumed_[s], 0));
  const bool quant = mode_ != QMode::BF16;
  char* raw = static_cast<char*>(raw_[quant ? 0 : s].get());
  MX_CUDA(cudaMemcpyAsync(raw, stage_[s].get(), b.raw_bytes, cudaMemcpyHostToDevice, copy_));
  MX_CUDA(cudaEventRecord(staged_[s], copy_));
  if (quant) {
    // 原始權重 → bf16 → 區塊縮放；與 c.stream 上的計算重疊。
    for (size_t j = 0; j < b.linears.size(); j++) {
      const Linear& l = *b.linears[j];
      dequant_to_bf16(raw + b.raw_offs[j], b.refs[j]->type, l.in * l.out, tmp_.as<bf16>(), copy_);
      size_t sf_off, amax_off;
      qlayout(mode_, l, sf_off, amax_off);
      char* base = static_cast<char*>(q_[s].get()) + b.q_offs[j];
      quantize_blockscaled(mode_, tmp_.as<bf16>(), l.in, l.out, l.in,
                           {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)},
                           copy_);
    }
  }
  MX_CUDA(cudaEventRecord(copied_[s], copy_));
  slot_block_[s] = i;
}

void BlockStreamer::acquire(Ctx& c, int i) {
  MX_CHECK(i >= 0 && i < blocks(), "BlockStreamer：塊索引 %d 超出範圍", i);
  MX_CHECK(c.device == dev_, "BlockStreamer：Ctx 裝置不符");
  if (last_ >= 0) MX_CUDA(cudaEventRecord(consumed_[last_ % 2], c.stream));
  issue_copy(i);
  const int s = i % 2;
  MX_CUDA(cudaStreamWaitEvent(c.stream, copied_[s], 0));
  Block& b = blocks_[i];
  for (size_t j = 0; j < b.linears.size(); j++) {
    Linear& l = *b.linears[j];
    if (mode_ == QMode::BF16) {
      l.w.p = static_cast<char*>(raw_[s].get()) + b.raw_offs[j];
    } else {
      char* base = static_cast<char*>(q_[s].get()) + b.q_offs[j];
      size_t sf_off, amax_off;
      qlayout(mode_, l, sf_off, amax_off);
      l.qw->m = {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)};
    }
  }
  last_ = i;
  if (blocks() > 1) issue_copy((i + 1) % blocks());
}

}  // namespace mx
