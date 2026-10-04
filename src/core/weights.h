#pragma once
// 權重常駐 GPU、線性層（量化權重即時反量化 + cuBLAS）與 LoRA。
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/ctx.h"
#include "core/model_file.h"

namespace mx {

struct DevTensor {
  const void* p = nullptr;
  GType type = GType::F32;
  std::vector<int64_t> shape;
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

// 依名稱把原始（可能是量化）權重搬上指定 GPU，重複呼叫會重用。
// ponytail: 整個模型常駐；VRAM 不夠的大模型之後改成逐層串流。
class DeviceWeights {
 public:
  DeviceWeights(const TensorStore& st, int device) : st_(st), dev_(device) {}
  const DevTensor& raw(const std::string& name);
  const float* f32(const std::string& name);      // 小張量（norm、bias）轉 f32 常駐
  const float* f32_or_null(const std::string& name) { return st_.has(name) ? f32(name) : nullptr; }
  const TensorStore& store() const { return st_; }
  size_t bytes() const { return bytes_; }
  int device() const { return dev_; }

 private:
  const TensorStore& st_;
  int dev_;
  std::map<std::string, std::pair<DevBuf, DevTensor>> raw_;
  std::map<std::string, DevBuf> f32_;
  size_t bytes_ = 0;
};

// 一個 LoRA 對某個線性層輸出區段 [out_off, out_off+out_len) 的增量：y += scale · (x Aᵀ) Bᵀ。
struct LoraDelta {
  DevBuf A, B;  // A [r, in]、B [out_len, r]，bf16
  int64_t rank = 0, out_off = 0, out_len = 0;
  float scale = 1.f;
};

struct Linear {
  std::string name;
  DevTensor w;                 // [out, in]
  const float* bias = nullptr; // [out]
  int64_t in = 0, out = 0;
  std::vector<std::shared_ptr<LoraDelta>> loras;
  explicit operator bool() const { return w.p != nullptr; }
};

// prefix 例如 "transformer_blocks.0.attn.to_q"，讀 prefix.weight（與 prefix.bias，若有）。
Linear make_linear(DeviceWeights& dw, const std::string& prefix);
// 最大權重 bf16 所需的反量化暫存大小（給 Ctx scratch 用）。
size_t linear_scratch_bytes(const Linear& l);

// y[M, out]（列距 ldy，0 表示 out）= x[M, in] · Wᵀ + b + LoRA。bf16 輸入輸出、f32 累加。
void linear(Ctx& c, const Linear& l, const bf16* x, int64_t M, bf16* y, int64_t ldy = 0);
// 精度敏感的小層（時間嵌入、調變、輸出頭）：全程 f32。
void linear_f32(Ctx& c, const Linear& l, const float* x, int64_t M, float* y);

// ---- LoRA ----
// 模型提供：LoRA 內部名稱（去掉常見前綴、點號形式，如 "transformer_blocks.0.attn.to_q"）
// → 目標 Linear 與輸出區段。回傳 false 表示這個模型沒有該層。
struct LoraTarget {
  Linear* linear = nullptr;
  int64_t out_off = 0, out_len = -1;  // -1 = 整個輸出
};
using LoraResolver = std::function<bool(const std::string& name, LoraTarget& t)>;

struct LoraLoadStats {
  int applied = 0;
  std::vector<std::string> unmatched;
};

// 讀 safetensors LoRA（diffusers/PEFT lora_A/lora_B、kohya lora_down/lora_up + alpha），掛到 Linear 上。
// kohya 的底線式名稱（lora_unet_transformer_blocks_0_attn_to_q）由 known_names 反查。
LoraLoadStats apply_lora(const std::string& path, float user_scale, int device,
                         const std::vector<std::string>& known_names, const LoraResolver& resolve);
// 移除所有 Linear 上的 LoRA。
void clear_loras(std::vector<Linear*>& linears);

}  // namespace mx
