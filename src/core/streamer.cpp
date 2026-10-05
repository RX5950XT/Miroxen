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
  for (int s = 0; s < 2; s++) {
    if (copied_[s]) cudaEventDestroy(copied_[s]);
    if (consumed_[s]) cudaEventDestroy(consumed_[s]);
  }
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
    b.offs.push_back(b.bytes);
    if (mode_ == QMode::BF16) {
      b.bytes += align256(r.nbytes);
    } else {
      MX_CHECK(l->in % 32 == 0, "%s：輸入維度不是 32 的倍數，無法量化成 %s", l->name.c_str(), qmode_name(mode_));
      size_t sf_off, amax_off;
      b.bytes += qlayout(mode_, *l, sf_off, amax_off);
      l->qw = std::make_shared<QWeight>();
      l->qw->mode = mode_;
    }
    b.refs.push_back(&r);
  }
  b.linears = std::move(linears);
  blocks_.push_back(std::move(b));
}

void BlockStreamer::finalize() {
  MX_CHECK(!blocks_.empty(), "BlockStreamer：沒有登記任何塊");
  size_t total = 0;
  for (auto& b : blocks_) {
    b.host_off = total;
    total += b.bytes;
    slot_bytes_ = std::max(slot_bytes_, b.bytes);
  }
  host_ = PinnedBuf(total);
  MX_CUDA(cudaSetDevice(dev_));
  if (mode_ != QMode::BF16) {
    for (auto& b : blocks_) quantize_block(b, static_cast<char*>(host_.get()) + b.host_off);
  } else {
    // 從映射檔案複製到鎖頁記憶體；多執行緒讓磁碟／頁面快取讀取並行。
    std::vector<std::thread> pool;
    const int nt = 8;
    for (int t = 0; t < nt; t++)
      pool.emplace_back([&, t] {
        for (size_t i = t; i < blocks_.size(); i += nt) {
          auto& b = blocks_[i];
          for (size_t j = 0; j < b.refs.size(); j++)
            std::memcpy(static_cast<char*>(host_.get()) + b.host_off + b.offs[j], b.refs[j]->data,
                        b.refs[j]->nbytes);
        }
      });
    for (auto& th : pool) th.join();
  }
  for (int s = 0; s < 2; s++) {
    slots_[s] = DevBuf(slot_bytes_, dev_);
    MX_CUDA(cudaEventCreateWithFlags(&copied_[s], cudaEventDisableTiming));
    MX_CUDA(cudaEventCreateWithFlags(&consumed_[s], cudaEventDisableTiming));
  }
  MX_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
  MX_INFO("權重串流：%zu 塊、鎖頁 %.2f GiB（%s）、GPU%d 槽 2×%.0f MiB", blocks_.size(), total / 1073741824.0,
          mode_ == QMode::BF16 ? "原始格式" : qmode_name(mode_), dev_, slot_bytes_ / 1048576.0);
}

// 原始權重 → GPU 反量化成 bf16 → 區塊縮放量化 → 下載到鎖頁記憶體（載入時一次）。
void BlockStreamer::quantize_block(const Block& b, char* host) {
  DevBuf dst(b.bytes, dev_);
  for (size_t j = 0; j < b.linears.size(); j++) {
    const Linear& l = *b.linears[j];
    const WeightRef& r = *b.refs[j];
    DevBuf raw(r.nbytes, dev_), tmp(size_t(l.in) * l.out * sizeof(bf16), dev_);
    MX_CUDA(cudaMemcpy(raw.get(), r.data, r.nbytes, cudaMemcpyHostToDevice));
    dequant_to_bf16(raw.get(), r.type, l.in * l.out, tmp.as<bf16>(), nullptr);
    size_t sf_off, amax_off;
    qlayout(mode_, l, sf_off, amax_off);
    char* base = static_cast<char*>(dst.get()) + b.offs[j];
    quantize_blockscaled(mode_, tmp.as<bf16>(), l.in, l.out, l.in,
                         {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)},
                         nullptr);
  }
  MX_CUDA(cudaMemcpy(host, dst.get(), b.bytes, cudaMemcpyDeviceToHost));
}

void BlockStreamer::issue_copy(int i) {
  int s = i % 2;
  if (slot_block_[s] == i) return;
  // 等這個槽的上一個使用者算完再覆寫。
  MX_CUDA(cudaStreamWaitEvent(copy_, consumed_[s], 0));
  const Block& b = blocks_[i];
  MX_CUDA(cudaMemcpyAsync(slots_[s].get(), static_cast<char*>(host_.get()) + b.host_off, b.bytes,
                          cudaMemcpyHostToDevice, copy_));
  MX_CUDA(cudaEventRecord(copied_[s], copy_));
  slot_block_[s] = i;
}

void BlockStreamer::acquire(Ctx& c, int i) {
  MX_CHECK(i >= 0 && i < blocks(), "BlockStreamer：塊索引 %d 超出範圍", i);
  MX_CHECK(c.device == dev_, "BlockStreamer：Ctx 裝置不符");
  if (last_ >= 0) MX_CUDA(cudaEventRecord(consumed_[last_ % 2], c.stream));
  issue_copy(i);
  int s = i % 2;
  MX_CUDA(cudaStreamWaitEvent(c.stream, copied_[s], 0));
  Block& b = blocks_[i];
  for (size_t j = 0; j < b.linears.size(); j++) {
    Linear& l = *b.linears[j];
    char* base = static_cast<char*>(slots_[s].get()) + b.offs[j];
    if (mode_ == QMode::BF16) {
      l.w.p = base;
    } else {
      size_t sf_off, amax_off;
      qlayout(mode_, l, sf_off, amax_off);
      l.qw->m = {base, reinterpret_cast<uint8_t*>(base + sf_off), reinterpret_cast<float*>(base + amax_off)};
    }
  }
  last_ = i;
  if (blocks() > 1) issue_copy((i + 1) % blocks());
}

}  // namespace mx
