#pragma once
// 資料型別（沿用 GGUF/ggml 的型別編號）、主機／裝置緩衝區、啟用值配置器。
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/common.h"

namespace mx {

// 編號與 ggml_type 一致，GGUF 檔頭可直接轉型。
enum class GType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
  IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
  MXFP4 = 39, NVFP4 = 40,
  // ggml 沒有的型別放在 1000 以後（safetensors 用）。
  F8_E4M3 = 1000, F8_E5M2 = 1001, U8 = 1002, BOOL = 1003,
};

const char* gtype_name(GType t);
int64_t gtype_block(GType t);       // 每個量化區塊的元素數（非量化型別為 1）
size_t gtype_block_bytes(GType t);  // 每個區塊的位元組數
bool gtype_is_quant(GType t);
inline size_t gtype_row_bytes(GType t, int64_t n) {
  return size_t(n / gtype_block(t)) * gtype_block_bytes(t);
}

// 裝置記憶體（RAII）。
class DevBuf {
 public:
  DevBuf() = default;
  DevBuf(size_t bytes, int device);
  ~DevBuf();
  DevBuf(DevBuf&& o) noexcept { *this = std::move(o); }
  DevBuf& operator=(DevBuf&& o) noexcept;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  void* get() const { return p_; }
  template <class T> T* as() const { return static_cast<T*>(p_); }
  size_t size() const { return n_; }
  int device() const { return dev_; }

 private:
  void* p_ = nullptr;
  size_t n_ = 0;
  int dev_ = -1;
};

// 鎖頁主機記憶體（RAII），用於非同步 H2D 串流。
class PinnedBuf {
 public:
  PinnedBuf() = default;
  explicit PinnedBuf(size_t bytes);
  ~PinnedBuf();
  PinnedBuf(PinnedBuf&& o) noexcept { *this = std::move(o); }
  PinnedBuf& operator=(PinnedBuf&& o) noexcept;
  PinnedBuf(const PinnedBuf&) = delete;
  PinnedBuf& operator=(const PinnedBuf&) = delete;
  void* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  void* p_ = nullptr;
  size_t n_ = 0;
};

// 一次 cudaMalloc，之後線性配置；用 mark()/release() 做區段性暫存。
// ponytail: 只支援堆疊式釋放；需要任意順序釋放時再換成真正的配置器。
class Arena {
 public:
  Arena() = default;
  Arena(size_t bytes, int device) : buf_(bytes, device) {}
  void* alloc(size_t bytes, size_t align = 256);
  template <class T> T* alloc_n(size_t n) { return static_cast<T*>(alloc(n * sizeof(T))); }
  size_t mark() const { return off_; }
  void release(size_t m) { off_ = m; }
  void reset() { off_ = 0; }
  size_t used() const { return off_; }
  size_t peak() const { return peak_; }
  size_t capacity() const { return buf_.size(); }
  int device() const { return buf_.device(); }

 private:
  DevBuf buf_;
  size_t off_ = 0, peak_ = 0;
};

// 區段結束時自動還原 Arena。
struct ArenaScope {
  Arena& a;
  size_t m;
  explicit ArenaScope(Arena& a_) : a(a_), m(a_.mark()) {}
  ~ArenaScope() { a.release(m); }
};

}  // namespace mx
