#pragma once
// 逐塊權重串流：每一塊（通常是一層 transformer）的線性層權重放在鎖頁主機記憶體，
// 執行時用獨立的 copy stream 雙緩衝搬上 GPU——計算第 i 塊時預取第 i+1 塊。
// 擴散模型每層計算時間遠大於 PCIe 搬運時間，所以串流幾乎不增加總時間，VRAM 只需兩層權重。
#include <vector>

#include "core/weights.h"

namespace mx {

// 只讀 metadata、不上傳權重的線性層（w.p 由 BlockStreamer 填入）；bias 仍常駐 GPU（f32）。
Linear make_linear_streamed(DeviceWeights& dw, const std::string& prefix);

class BlockStreamer {
 public:
  explicit BlockStreamer(int device) : dev_(device) {}
  ~BlockStreamer();
  BlockStreamer(const BlockStreamer&) = delete;
  BlockStreamer& operator=(const BlockStreamer&) = delete;

  // 依序登記每一塊包含的線性層（Linear 物件必須在 streamer 存活期間保持位址不變）。
  void add_block(const TensorStore& st, std::vector<Linear*> linears);
  // 全部登記完後呼叫：把權重複製到鎖頁記憶體並配置兩個裝置槽。
  void finalize();

  // 讓第 i 塊的 Linear::w.p 指向裝置上的資料（必要時等待搬運），並預取第 i+1 塊（循環）。
  // 呼叫前，前一塊的所有核心都必須已排入 c.stream。
  void acquire(Ctx& c, int i);
  int blocks() const { return int(blocks_.size()); }
  size_t pinned_bytes() const { return host_.size(); }
  size_t device_bytes() const { return 2 * slot_bytes_; }

 private:
  struct Block {
    std::vector<Linear*> linears;
    std::vector<const WeightRef*> refs;
    std::vector<size_t> offs;  // 塊內偏移
    size_t host_off = 0, bytes = 0;
  };
  void issue_copy(int i);

  int dev_;
  std::vector<Block> blocks_;
  PinnedBuf host_;
  DevBuf slots_[2];
  size_t slot_bytes_ = 0;
  cudaStream_t copy_ = nullptr;
  cudaEvent_t copied_[2] = {}, consumed_[2] = {};
  int slot_block_[2] = {-1, -1};  // 槽內（或正在搬入）的塊
  int last_ = -1;                  // 上一次 acquire 的塊
};

}  // namespace mx
