#include "core/dequant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

using namespace mx;
namespace {
const GType types[] = {GType::F32, GType::F16, GType::BF16, GType::F8_E4M3, GType::F8_E5M2,
  GType::Q4_0, GType::Q4_1, GType::Q5_0, GType::Q5_1, GType::Q8_0, GType::Q2_K, GType::Q3_K,
  GType::Q4_K, GType::Q5_K, GType::Q6_K, GType::IQ4_NL, GType::IQ4_XS};

bool same(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(float)) == 0;
}
std::vector<uint8_t> random_data(GType t, int64_t n) {
  std::mt19937 rng(100 + unsigned(t));
  const auto bytes = gtype_block_bytes(t), block = size_t(gtype_block(t));
  std::vector<uint8_t> data(n / block * bytes);
  for (auto& v : data) v = uint8_t(rng());
  if (t == GType::F32) {
    for (int64_t i = 0; i < n; ++i) { const float v = int(rng() % 65537) * 0.001f - 32.f; std::memcpy(data.data() + i * 4, &v, 4); }
    const uint32_t edges[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345};
    if (n >= int64_t(std::size(edges))) std::memcpy(data.data(), edges, sizeof(edges));
    return data;
  }
  std::vector<int> offsets;
  if (t == GType::F16 || t == GType::BF16 || t == GType::Q4_0 || t == GType::Q5_0 ||
      t == GType::Q8_0 || t == GType::IQ4_NL || t == GType::IQ4_XS) offsets = {0};
  if (t == GType::Q4_1 || t == GType::Q5_1 || t == GType::Q4_K || t == GType::Q5_K) offsets = {0, 2};
  if (t == GType::Q2_K) offsets = {80, 82};
  if (t == GType::Q3_K) offsets = {108};
  if (t == GType::Q6_K) offsets = {208};
  for (size_t b = 0; b < size_t(n) / block; ++b) for (auto off : offsets) {
    // 有限 scale，包含 subnormal 與負值；bf16 也限制在有限範圍。
    const uint16_t v = uint16_t(rng() % 0x7c00) | uint16_t((rng() & 1) << 15);
    data[b * bytes + off] = uint8_t(v); data[b * bytes + off + 1] = uint8_t(v >> 8);
  }
  return data;
}

struct Stream {
  cudaStream_t s;
  Stream() { MX_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s); }
};
void compare_gpu(GType t, const std::vector<uint8_t>& raw, const std::vector<float>& ref, int dev, bool unaligned = false) {
  const int64_t n = ref.size();
  DevBuf input(raw.size() + 1, dev), outf((n + 2) * 4, dev), outb((n + 2) * 2, dev);
  Stream stream;
  auto src = input.as<uint8_t>() + (unaligned ? 1 : 0);
  auto dstf = outf.as<float>() + (unaligned ? 1 : 0);
  auto dstb = outb.as<bf16>() + (unaligned ? 1 : 0);
  MX_CUDA(cudaMemsetAsync(outf.get(), 0xa5, outf.size(), stream.s));
  MX_CUDA(cudaMemsetAsync(outb.get(), 0xa5, outb.size(), stream.s));
  MX_CUDA(cudaMemcpyAsync(src, raw.data(), raw.size(), cudaMemcpyHostToDevice, stream.s));
  dequant_to_f32(src, t, n, dstf, stream.s);
  dequant_to_bf16(src, t, n, dstb, stream.s);
  std::vector<float> f(n);
  std::vector<bf16> b(n);
  MX_CUDA(cudaMemcpyAsync(f.data(), dstf, n * 4, cudaMemcpyDeviceToHost, stream.s));
  MX_CUDA(cudaMemcpyAsync(b.data(), dstb, n * 2, cudaMemcpyDeviceToHost, stream.s));
  MX_CUDA(cudaStreamSynchronize(stream.s));
  for (int64_t i = 0; i < n; ++i) {
    MX_CHECK(same(f[i], ref[i]), "%s GPU%d f32[%lld]: %.9g != %.9g", gtype_name(t), dev, (long long)i, f[i], ref[i]);
    if (t == GType::F32) MX_CHECK(std::memcmp(&f[i], &ref[i], 4) == 0, "F32 NaN payload or signed zero changed");
    const float rounded = __bfloat162float(__float2bfloat16_rn(ref[i]));
    MX_CHECK(same(__bfloat162float(b[i]), rounded), "%s GPU%d bf16[%lld] rounding mismatch", gtype_name(t), dev, (long long)i);
  }
  uint32_t guard;
  MX_CUDA(cudaMemcpy(&guard, dstf + n, 4, cudaMemcpyDeviceToHost));
  MX_CHECK(guard == 0xa5a5a5a5, "f32 end guard overwritten");
  uint16_t guardb;
  MX_CUDA(cudaMemcpy(&guardb, dstb + n, 2, cudaMemcpyDeviceToHost));
  MX_CHECK(guardb == 0xa5a5, "bf16 end guard overwritten");
  if (unaligned) {
    MX_CUDA(cudaMemcpy(&guard, outf.get(), 4, cudaMemcpyDeviceToHost));
    MX_CUDA(cudaMemcpy(&guardb, outb.get(), 2, cudaMemcpyDeviceToHost));
    MX_CHECK(guard == 0xa5a5a5a5 && guardb == 0xa5a5, "start guard overwritten");
  }
}
template<class T> T read(std::ifstream& f) {
  T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(v)); MX_CHECK(bool(f), "Reference vector truncated"); return v;
}
void reference_vectors(int dev) {
  std::ifstream f("tests/data/dequant_vectors.bin", std::ios::binary);
  MX_CHECK(bool(f), "Missing tests/data/dequant_vectors.bin");
  char magic[8]; f.read(magic, 8); MX_CHECK(std::memcmp(magic, "MXDQ0001", 8) == 0, "Invalid vector magic");
  const auto count = read<uint32_t>(f); MX_CHECK(count == std::size(types), "Vector format count mismatch");
  std::vector<GType> seen;
  size_t elements = 0;
  for (uint32_t r = 0; r < count; ++r) {
    const auto t = GType(read<uint32_t>(f));
    const auto n = read<uint32_t>(f), bytes = read<uint32_t>(f);
    MX_CHECK(dequant_supported(t) && n < 100000 && bytes == gtype_row_bytes(t, n), "Invalid vector sizes");
    MX_CHECK(std::find(seen.begin(), seen.end(), t) == seen.end(), "Duplicate reference format"); seen.push_back(t);
    std::vector<uint8_t> raw(bytes); std::vector<float> ref(n), actual(n);
    f.read(reinterpret_cast<char*>(raw.data()), bytes); f.read(reinterpret_cast<char*>(ref.data()), n * 4);
    MX_CHECK(bool(f), "Reference vector data truncated");
    dequant_cpu(raw.data(), t, n, actual.data());
    for (size_t i = 0; i < n; ++i) MX_CHECK(same(ref[i], actual[i]), "%s Python/CPU[%zu]: %.9g != %.9g", gtype_name(t), i, ref[i], actual[i]);
    compare_gpu(t, raw, actual, dev);
    elements += n;
  }
  MX_CHECK(f.peek() == EOF, "Reference vector trailing data");
  std::printf("GPU%d Python/CPU/GPU reference: %u formats, %zu elements PASS (gguf-py 15, FP8 table 2)\n", dev, count, elements);
}

template<class F> void rejects(F f) {
  try { f(); } catch (const std::runtime_error&) { return; }
  throw std::runtime_error("Expected invalid input to throw");
}
void invalid_inputs() {
  float v = 0;
  rejects([&] { dequant_cpu(&v, GType::I8, 1, &v); });
  rejects([&] { dequant_to_f32(&v, GType(9999), 1, &v, nullptr); });
  rejects([&] { dequant_to_bf16(&v, GType::I8, 1, nullptr, nullptr); });
  rejects([&] { dequant_to_bf16(&v, GType::Q8_0, 31, nullptr, nullptr); });
  rejects([&] { dequant_cpu(&v, GType::F32, -1, &v); });
  rejects([&] { dequant_cpu(nullptr, GType::F32, 1, &v); });
  dequant_cpu(nullptr, GType::F32, 0, nullptr);
  dequant_to_f32(nullptr, GType::F32, 0, nullptr, nullptr);
  dequant_to_bf16(nullptr, GType::F32, 0, nullptr, nullptr);
}

void benchmark(GType t, int dev) {
  constexpr int64_t n = int64_t(5376) * 28672;
  const auto seed = random_data(t, gtype_block(t));
  std::vector<uint8_t> raw(gtype_row_bytes(t, n));
  for (size_t i = 0; i < raw.size(); i += seed.size()) std::memcpy(raw.data() + i, seed.data(), seed.size());
  DevBuf src(raw.size(), dev), dst(n * 2, dev);
  MX_CUDA(cudaMemcpy(src.get(), raw.data(), raw.size(), cudaMemcpyHostToDevice));
  Stream stream;
  for (int i = 0; i < 3; ++i) dequant_to_bf16(src.get(), t, n, dst.as<bf16>(), stream.s);
  MX_CUDA(cudaStreamSynchronize(stream.s));
  cudaEvent_t start, end;
  MX_CUDA(cudaEventCreate(&start)); MX_CUDA(cudaEventCreate(&end));
  std::vector<float> times;
  for (int i = 0; i < 15; ++i) {
    MX_CUDA(cudaEventRecord(start, stream.s));
    dequant_to_bf16(src.get(), t, n, dst.as<bf16>(), stream.s);
    MX_CUDA(cudaEventRecord(end, stream.s)); MX_CUDA(cudaEventSynchronize(end));
    float ms; MX_CUDA(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms);
  }
  MX_CUDA(cudaEventDestroy(start)); MX_CUDA(cudaEventDestroy(end));
  std::sort(times.begin(), times.end()); const float ms = times[times.size() / 2];
  std::vector<float> ref(gtype_block(t)); std::vector<bf16> got(ref.size());
  dequant_cpu(seed.data(), t, ref.size(), ref.data());
  MX_CUDA(cudaMemcpy(got.data(), dst.as<bf16>() + n - got.size(), got.size() * 2, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < ref.size(); ++i) MX_CHECK(same(__bfloat162float(got[i]), __bfloat162float(__float2bfloat16(ref[i]))), "Large matrix end mismatch");
  cudaDeviceProp prop; MX_CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("BENCH GPU%d %s %s -> bf16 5376x28672: median %.3f ms, min %.3f ms, max %.3f ms, %.1f GB/s (input+output; 15 runs)\n",
              dev, prop.name, gtype_name(t), ms, times.front(), times.back(), (raw.size() + n * 2) / (ms * 1e6));
}
}  // namespace

int main() {
  try {
    invalid_inputs();
    int devices; MX_CUDA(cudaGetDeviceCount(&devices)); MX_CHECK(devices > 0, "No CUDA device");
    int benchmark_device = 0;
    for (int dev = 0; dev < devices; ++dev) {
      MX_CUDA(cudaSetDevice(dev)); cudaDeviceProp prop; MX_CUDA(cudaGetDeviceProperties(&prop, dev));
      std::printf("GPU%d %s sm_%d%d\n", dev, prop.name, prop.major, prop.minor);
      if (prop.major == 12) benchmark_device = dev;
      reference_vectors(dev);
      for (auto t : types) {
        const auto n = gtype_block(t) == 1 ? 8193 : gtype_block(t) * 33;
        auto raw = random_data(t, n); std::vector<float> ref(n);
        dequant_cpu(raw.data(), t, n, ref.data());
        compare_gpu(t, raw, ref, dev); compare_gpu(t, raw, ref, dev, true);
      }
      std::printf("GPU%d random/unaligned/stream/guards: 17 formats PASS\n", dev);
    }
    MX_CUDA(cudaSetDevice(benchmark_device)); benchmark(GType::Q4_K, benchmark_device); benchmark(GType::Q8_0, benchmark_device);
    std::puts("dequant_test PASS"); return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "dequant_test FAIL: %s\n", e.what()); return 1; }
}
