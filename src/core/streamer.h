#pragma once
// 逐塊權重串流：每一塊（通常是一層 transformer）的線性層權重直接從模型檔（mmap）讀，
// 經兩份鎖頁暫存區送上 GPU 雙緩衝——計算第 i 塊時預取第 i+1 塊。
// 不另外保留整份主機副本：鎖頁記憶體只有兩塊大小，檔案內容由作業系統的頁面快取管理（記憶體吃緊時可回收）。
// 擴散模型每層計算時間遠大於讀取＋PCIe 搬運時間，所以串流幾乎不增加總時間，VRAM 只需兩層權重。
// QMode 為 FP8/FP4 時，原始權重上 GPU 後在 copy stream 上反量化＋（併入 LoRA）＋區塊縮放量化（與計算重疊），
// 所以 LoRA 不佔取樣時間。換 LoRA 後要呼叫 invalidate()，丟掉已預取的槽。
#include <vector>

#include "core/weights.h"

namespace mx {

// 只讀 metadata、不上傳權重的線性層（w.p 由 BlockStreamer 填入）；bias 仍常駐 GPU（f32）。
Linear make_linear_streamed(DeviceWeights& dw, const std::string& prefix);
// 分開存放的 gate/proj 串流時在槽內沿輸出列接合（位元組相接，與 make_linear_concat 相同）。
Linear make_linear_streamed_concat(DeviceWeights& dw, const std::string& prefix,
                                   const std::string& first, const std::string& second);

class BlockStreamer {
 public:
  explicit BlockStreamer(int device, QMode mode = QMode::BF16) : dev_(device), mode_(mode) {}
  ~BlockStreamer();
  BlockStreamer(const BlockStreamer&) = delete;
  BlockStreamer& operator=(const BlockStreamer&) = delete;

  // 依序登記每一塊包含的線性層（Linear 物件必須在 streamer 存活期間保持位址不變）。
  void add_block(const TensorStore& st, std::vector<Linear*> linears);
  // 全部登記完後呼叫：配置鎖頁暫存與裝置槽。
  void finalize();

  // 讓第 i 塊的 Linear::w.p（量化時是 Linear::qw）指向裝置上的資料（必要時等待搬運），並預取第 i+1 塊（循環）。
  // 呼叫前，前一塊的所有核心都必須已排入 c.stream。
  void acquire(Ctx& c, int i);
  int blocks() const { return int(blocks_.size()); }
  // 丟掉槽內已搬好的內容（例如 LoRA 改了），下次 acquire 重新搬。
  void invalidate();
  size_t weight_bytes() const { return total_raw_; }  // 原始權重總量（在檔案裡，不佔鎖頁記憶體）

 private:
  struct Block {
    std::vector<Linear*> linears;
    std::vector<std::vector<const WeightRef*>> refs;  // 每個線性層的來源（接合層有兩份）
    std::vector<size_t> raw_offs, q_offs;  // 塊內偏移：原始資料／量化資料（sf、amax 緊接在 q 後）
    size_t raw_bytes = 0, q_bytes = 0;
  };
  void issue_copy(int i);
  void fill_stage(const Block& b, char* dst);

  int dev_;
  QMode mode_;
  std::vector<Block> blocks_;
  size_t total_raw_ = 0, raw_max_ = 0, q_max_ = 0, tmp_max_ = 0;
  PinnedBuf stage_[2];
  DevBuf raw_[2];   // BF16：計算直接讀這裡；量化模式只用 raw_[0] 當轉換來源
  DevBuf q_[2], tmp_;  // 量化模式：轉換後的槽與 bf16 中繼
  cudaStream_t copy_ = nullptr;
  cublasHandle_t blas_ = nullptr;  // 綁在 copy_ 上，量化模式併 LoRA 用
  cudaEvent_t staged_[2] = {}, copied_[2] = {}, consumed_[2] = {};
  int slot_block_[2] = {-1, -1};  // 槽內（或正在搬入）的塊
  int last_ = -1;                  // 上一次 acquire 的塊
};

}  // namespace mx
