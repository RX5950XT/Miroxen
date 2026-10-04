#pragma once
// 模型檔讀取：GGUF 與 safetensors，皆以記憶體映射（mmap）開啟，零複製。
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "core/tensor.h"

namespace mx {

// 一個張量在映射檔案中的位置。shape 一律採 PyTorch 慣例（最外層在前、最後一維連續），
// GGUF 的 ne[] 會反轉後放進來。
struct WeightRef {
  std::string name;
  GType type = GType::F32;
  std::vector<int64_t> shape;
  const void* data = nullptr;  // 指向映射記憶體
  size_t nbytes = 0;
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

// GGUF metadata 值。
using GgufValue = std::variant<int64_t, uint64_t, double, bool, std::string, std::vector<int64_t>,
                               std::vector<double>, std::vector<std::string>>;

class MappedFile;

// 開啟一個或多個檔案，依名稱查張量。後加入的檔案若有同名張量會覆蓋（並記錄除錯訊息）。
class TensorStore {
 public:
  TensorStore();
  ~TensorStore();
  TensorStore(TensorStore&&) noexcept;
  TensorStore& operator=(TensorStore&&) noexcept;

  // 依副檔名判斷 .gguf / .safetensors；prefix 會被剝除（例如 "model.diffusion_model."），
  // 傳空字串表示自動剝除常見前綴：model.diffusion_model. / diffusion_model. / model.
  void add_file(const std::string& path, const std::string& strip_prefix = "");

  const WeightRef* find(const std::string& name) const;
  const WeightRef& get(const std::string& name) const;  // 找不到就丟例外（訊息含名稱）
  bool has(const std::string& name) const { return find(name) != nullptr; }
  std::vector<std::string> names() const;
  size_t size() const { return tensors_.size(); }

  // GGUF metadata（多個檔案合併）；safetensors 的 __metadata__ 以字串存入。
  const GgufValue* meta(const std::string& key) const;
  std::string meta_str(const std::string& key, const std::string& def = "") const;
  int64_t meta_int(const std::string& key, int64_t def = 0) const;

 private:
  std::vector<std::unique_ptr<MappedFile>> files_;
  std::map<std::string, WeightRef> tensors_;
  std::map<std::string, GgufValue> meta_;
};

}  // namespace mx
