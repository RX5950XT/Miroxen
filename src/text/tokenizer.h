#pragma once
// Qwen2 / Qwen3 系列的 byte-level BPE tokenizer（讀 HuggingFace tokenizer.json）。
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mx {

class Tokenizer {
 public:
  // 讀取 tokenizer.json（含 model.vocab、model.merges、added_tokens）。
  static std::unique_ptr<Tokenizer> load(const std::string& tokenizer_json_path);
  virtual ~Tokenizer() = default;

  // 編碼 UTF-8 文字。文字中出現的 added/special token 原字串（如 "<|im_start|>"）
  // 會直接對應到其 id，其餘片段走 Qwen2 pre-tokenizer + BPE。不加 BOS/EOS。
  virtual std::vector<int32_t> encode(const std::string& text) const = 0;
  virtual std::string decode(const std::vector<int32_t>& ids) const = 0;
  virtual int32_t token_id(const std::string& token) const = 0;  // 找不到回傳 -1
  virtual size_t vocab_size() const = 0;
};

}  // namespace mx
