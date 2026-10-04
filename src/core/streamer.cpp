#include "core/streamer.h"

#include <algorithm>
#include <cstring>
#include <thread>

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

void BlockStreamer::add_block(const TensorStore& st, std::vector<Linear*> linears) {
  Block b;
  for (Linear* l : linears) {
    const WeightRef& r = st.get(l->name + ".weight");
    b.offs.push_back(b.bytes);
    b.bytes += (r.nbytes + 255) / 256 * 256;
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
  MX_CUDA(cudaSetDevice(dev_));
  for (int s = 0; s < 2; s++) {
    slots_[s] = DevBuf(slot_bytes_, dev_);
    MX_CUDA(cudaEventCreateWithFlags(&copied_[s], cudaEventDisableTiming));
    MX_CUDA(cudaEventCreateWithFlags(&consumed_[s], cudaEventDisableTiming));
  }
  MX_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
  MX_INFO("權重串流：%zu 塊、鎖頁 %.2f GiB、GPU%d 槽 2×%.0f MiB", blocks_.size(),
          total / 1073741824.0, dev_, slot_bytes_ / 1048576.0);
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
  for (size_t j = 0; j < b.linears.size(); j++)
    b.linears[j]->w.p = static_cast<char*>(slots_[s].get()) + b.offs[j];
  last_ = i;
  if (blocks() > 1) issue_copy((i + 1) % blocks());
}

}  // namespace mx
