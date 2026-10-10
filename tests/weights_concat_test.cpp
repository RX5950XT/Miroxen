// 分開的 FFN 權重接合：列順序、輸入驗證、FP8 LoRA 合併及移除後還原。
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>
#include "core/streamer.h"
#include "core/weights.h"

using namespace mx;

int main() {
  auto path = std::filesystem::temp_directory_path() /
      ("miroxen-concat-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
  try {
    const std::string header = R"({"gate.weight":{"dtype":"F32","shape":[64,64],"data_offsets":[0,16384]},"proj.weight":{"dtype":"F32","shape":[64,64],"data_offsets":[16384,32768]},"bad.weight":{"dtype":"F32","shape":[64,32],"data_offsets":[32768,40960]}})";
    const uint64_t len = header.size();
    std::vector<float> data(10240, 3.f);
    std::fill(data.begin(), data.begin() + 4096, 1.f);
    std::fill(data.begin() + 4096, data.begin() + 8192, 2.f);
    { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(&len), 8);
      f.write(header.data(), len); f.write(reinterpret_cast<const char*>(data.data()), data.size() * 4); }
    {
      TensorStore st; st.add_file(path.string());
      DeviceWeights dw(st, 0);
      auto l = make_linear_concat(dw, "fused", "gate", "proj");
      MX_CHECK(l.out == 128 && l.in == 64, "接合形狀錯誤");
      MX_CHECK(dw.raw_concat("fused.weight", "gate.weight", "proj.weight").p == l.w.p, "接合快取失效");
      bool rejected = false;
      try { make_linear_concat(dw, "invalid", "gate", "bad"); }
      catch (const std::runtime_error&) { rejected = true; }
      MX_CHECK(rejected, "未拒絕不相容形狀");
      Ctx c(0, 1 << 20, linear_scratch_bytes(l));
      std::vector<bf16> x(64, __float2bfloat16(1.f)), got(128);
      DevBuf xd(x.size() * 2, 0), yd(got.size() * 2, 0);
      MX_CUDA(cudaMemcpy(xd.get(), x.data(), xd.size(), cudaMemcpyHostToDevice));
      auto check = [&](float first) {
        linear(c, l, xd.as<bf16>(), 1, yd.as<bf16>()); c.sync();
        MX_CUDA(cudaMemcpy(got.data(), yd.get(), yd.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 128; i++)
          MX_CHECK(std::abs(__bfloat162float(got[i]) - (i < 64 ? first : 128.f)) < 1.f, "接合／LoRA 列順序錯誤");
      };
      check(64.f);
      if (qmode_supported(QMode::MXFP8, 0)) {
        dw.quantize(l, QMode::MXFP8); check(64.f);
        auto d = std::make_shared<LoraDelta>(); d->rank = 1; d->out_len = 64;
        std::vector<bf16> b(64, __float2bfloat16(.25f));
        d->A = DevBuf(128, 0); d->B = DevBuf(128, 0);
        MX_CUDA(cudaMemcpy(d->A.get(), x.data(), 128, cudaMemcpyHostToDevice));
        MX_CUDA(cudaMemcpy(d->B.get(), b.data(), 128, cudaMemcpyHostToDevice));
        l.loras.push_back(d); dw.merge_loras(c, l); check(80.f);
        l.loras.clear(); dw.merge_loras(c, l); check(64.f);
      }
      // 串流版：兩份來源在槽內位元組相接，結果與常駐接合相同（BF16 與 FP8 槽）。
      for (QMode m : {QMode::BF16, QMode::MXFP8}) {
        if (m != QMode::BF16 && !qmode_supported(m, 0)) continue;
        Linear sl = make_linear_streamed_concat(dw, "fused_stream", "gate", "proj");
        MX_CHECK(sl.out == 128 && sl.in == 64 && sl.sources.size() == 2, "串流接合形狀錯誤");
        BlockStreamer bs(0, m);
        bs.add_block(st, {&sl});
        bs.finalize();
        bs.acquire(c, 0);
        std::swap(l, sl);
        check(64.f);
        std::swap(l, sl);
      }
    }
    std::filesystem::remove(path);
    std::puts("PASS：FFN 接合、形狀驗證、LoRA 合併／還原、串流接合");
    return 0;
  } catch (const std::exception& e) {
    std::filesystem::remove(path);
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
