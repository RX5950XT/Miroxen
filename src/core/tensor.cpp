#include "core/tensor.h"

#include <cstdlib>
#include <mutex>

namespace mx {

// CUDA 預設把 SM 較多的卡排第一（本機 3060 Ti 38 SM > 5060 Ti 36 SM），
// 和 nvidia-smi 相反。在任何 CUDA 呼叫前固定成 PCI 匯流排順序（使用者已設定則不覆寫）。
static const int kDeviceOrder = [] {
  if (!std::getenv("CUDA_DEVICE_ORDER")) {
#ifdef _WIN32
    _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#endif
  }
  return 0;
}();

int log_level() {
  static int lv = [] {
    const char* e = std::getenv("MX_LOG");
    return e ? std::atoi(e) : 1;
  }();
  return lv;
}

void log_msg(int level, const char* fmt, ...) {
  if (level > log_level()) return;
  static std::mutex m;
  va_list ap;
  va_start(ap, fmt);
  char buf[4096];
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lk(m);
  std::fprintf(level == 0 ? stderr : stdout, "%s%s\n", level == 0 ? "[錯誤] " : "", buf);
  std::fflush(level == 0 ? stderr : stdout);
}

struct TypeInfo {
  GType t;
  const char* name;
  int64_t blk;
  size_t bytes;
};

static const TypeInfo kTypes[] = {
    {GType::F32, "F32", 1, 4},          {GType::F16, "F16", 1, 2},
    {GType::Q4_0, "Q4_0", 32, 18},      {GType::Q4_1, "Q4_1", 32, 20},
    {GType::Q5_0, "Q5_0", 32, 22},      {GType::Q5_1, "Q5_1", 32, 24},
    {GType::Q8_0, "Q8_0", 32, 34},      {GType::Q8_1, "Q8_1", 32, 36},
    {GType::Q2_K, "Q2_K", 256, 84},     {GType::Q3_K, "Q3_K", 256, 110},
    {GType::Q4_K, "Q4_K", 256, 144},    {GType::Q5_K, "Q5_K", 256, 176},
    {GType::Q6_K, "Q6_K", 256, 210},    {GType::Q8_K, "Q8_K", 256, 292},
    {GType::IQ2_XXS, "IQ2_XXS", 256, 66}, {GType::IQ2_XS, "IQ2_XS", 256, 74},
    {GType::IQ3_XXS, "IQ3_XXS", 256, 98}, {GType::IQ1_S, "IQ1_S", 256, 50},
    {GType::IQ4_NL, "IQ4_NL", 32, 18},  {GType::IQ3_S, "IQ3_S", 256, 110},
    {GType::IQ2_S, "IQ2_S", 256, 82},   {GType::IQ4_XS, "IQ4_XS", 256, 136},
    {GType::I8, "I8", 1, 1},            {GType::I16, "I16", 1, 2},
    {GType::I32, "I32", 1, 4},          {GType::I64, "I64", 1, 8},
    {GType::F64, "F64", 1, 8},          {GType::IQ1_M, "IQ1_M", 256, 56},
    {GType::BF16, "BF16", 1, 2},        {GType::F8_E4M3, "F8_E4M3", 1, 1},
    {GType::F8_E5M2, "F8_E5M2", 1, 1},  {GType::U8, "U8", 1, 1},
    {GType::BOOL, "BOOL", 1, 1},
    {GType::MXFP4, "MXFP4", 32, 17},   {GType::NVFP4, "NVFP4", 64, 36},
};

static const TypeInfo& info(GType t) {
  for (auto& i : kTypes)
    if (i.t == t) return i;
  throw std::runtime_error(strf("未知的張量型別 %u", unsigned(t)));
}

const char* gtype_name(GType t) { return info(t).name; }
int64_t gtype_block(GType t) { return info(t).blk; }
size_t gtype_block_bytes(GType t) { return info(t).bytes; }
bool gtype_is_quant(GType t) { return info(t).blk > 1; }

DevBuf::DevBuf(size_t bytes, int device) : n_(bytes), dev_(device) {
  if (!bytes) return;
  int prev;
  MX_CUDA(cudaGetDevice(&prev));
  MX_CUDA(cudaSetDevice(device));
  cudaError_t e = cudaMalloc(&p_, bytes);
  cudaSetDevice(prev);
  MX_CHECK(e == cudaSuccess, "GPU%d 配置 %.1f MiB 失敗：%s", device, bytes / 1048576.0,
           cudaGetErrorString(e));
}

DevBuf::~DevBuf() {
  if (p_) cudaFree(p_);
}

DevBuf& DevBuf::operator=(DevBuf&& o) noexcept {
  if (this != &o) {
    if (p_) cudaFree(p_);
    p_ = o.p_, n_ = o.n_, dev_ = o.dev_;
    o.p_ = nullptr, o.n_ = 0;
  }
  return *this;
}

PinnedBuf::PinnedBuf(size_t bytes) : n_(bytes) {
  if (bytes) MX_CUDA(cudaHostAlloc(&p_, bytes, cudaHostAllocPortable));
}

PinnedBuf::~PinnedBuf() {
  if (p_) cudaFreeHost(p_);
}

PinnedBuf& PinnedBuf::operator=(PinnedBuf&& o) noexcept {
  if (this != &o) {
    if (p_) cudaFreeHost(p_);
    p_ = o.p_, n_ = o.n_;
    o.p_ = nullptr, o.n_ = 0;
  }
  return *this;
}

void* Arena::alloc(size_t bytes, size_t align) {
  size_t off = (off_ + align - 1) / align * align;
  MX_CHECK(off + bytes <= buf_.size(), "Arena 不足：需要 %.1f MiB，已用 %.1f / %.1f MiB",
           bytes / 1048576.0, off_ / 1048576.0, buf_.size() / 1048576.0);
  off_ = off + bytes;
  if (off_ > peak_) peak_ = off_;
  return static_cast<char*>(buf_.get()) + off;
}

}  // namespace mx
