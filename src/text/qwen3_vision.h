#pragma once
// Qwen3-VL 視覺塔：16×16 patch → 27 層 ViT（2D RoPE）→ 2×2 合併成 LLM token，另輸出 DeepStack 特徵
// （第 8/16/24 層，加進語言模型前 3 層的視覺位置）。Qwen-Image-2.1 編輯（8B，out 4096）與
// MiniMax-H3 首尾幀（32B，out 5120）共用。讀 llama.cpp mmproj（v.*）或 HuggingFace（visual.*）命名。
#include <vector>

#include "core/weights.h"

namespace mx {

struct VisionFeatures {
  int64_t gh = 0, gw = 0;  // 合併後的網格（像素 / 32），token 依列優先排列
  int64_t tokens() const { return gh * gw; }
  std::vector<bf16> embeds;                  // [tokens, out_dim]
  std::vector<std::vector<bf16>> deepstack;  // 每個 DeepStack 層一份 [tokens, out_dim]
};

class Qwen3VisionEncoder {
 public:
  Qwen3VisionEncoder(const TensorStore& st, int device);
  // rgb：主機端 [h, w, 3] uint8，h、w 為 32 的倍數（呼叫端先縮放好）。
  VisionFeatures encode(Ctx& c, const uint8_t* rgb, int h, int w);
  int64_t out_dim() const { return merger_.fc2.out; }
  size_t act_bytes(int h, int w) const;
  size_t scratch_bytes() const;
  int device() const { return dw_.device(); }

 private:
  struct Norm {
    DevBuf scale;  // w - 1（layernorm_mod 的 (1+scale) 形式）
    const float* bias = nullptr;
  };
  struct Block {
    Norm n1, n2;
    Linear qkv, proj, fc1, fc2;
  };
  struct Merger {
    Norm norm;
    bool post_shuffle = false;  // DeepStack：先合併 4 個 token 再正規化
    Linear fc1, fc2;
  };
  Norm load_norm(const std::string& prefix);
  void merge(Ctx& c, const Merger& m, const float* x, int64_t n, bf16* out);

  DeviceWeights dw_;
  int64_t dim_ = 1152;
  int heads_ = 16, head_dim_ = 72, grid_side_ = 48;
  float eps_ = 1e-6f;
  DevBuf patch_w_;  // [dim, 3·16·16] bf16（兩個時間切片相加：單張圖的兩幀相同）
  Linear patch_;
  std::vector<float> pos_;  // [grid_side², dim]
  std::vector<Block> blocks_;
  Merger merger_;
  std::vector<int> ds_layers_;
  std::vector<Merger> ds_;
};

}  // namespace mx
