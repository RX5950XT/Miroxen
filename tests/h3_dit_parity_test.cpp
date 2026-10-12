// MiniMax-H3 DiT 與 diffusers 的數值比對（tools/ref/h3_dit_ref.py 產生參考；需要 pruned GGUF）。
// 同時驗證打包版面（RoPE 座標）與官方 build_packed_sequence 一致。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "core/dequant.h"
#include "models/minimax_h3/pipeline.h"

using namespace mx;

static std::vector<float> host_f32(const WeightRef& r) {
  std::vector<float> v(r.numel());
  dequant_cpu(r.data, r.type, r.numel(), v.data());
  return v;
}

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); i++) num += (a[i] - b[i]) * double(a[i] - b[i]), den += double(b[i]) * b[i];
  return std::sqrt(num / (den + 1e-30));
}

int main() {
  const char* gguf = "models/video/minimax-h3/minimax_h3_fl2va_pruned-Q4_K.gguf";
  const char* ref = std::getenv("MX_H3_REF") ? std::getenv("MX_H3_REF") : "models/ref/h3_dit_ref.safetensors";
  if (!std::filesystem::exists(gguf) || !std::filesystem::exists(ref)) {
    std::printf("SKIP：缺少 %s 或 %s\n", gguf, ref);
    return 0;
  }
  try {
    TensorStore st, rs;
    st.add_file(gguf);
    rs.add_file(ref, "\x01");
    auto mi = [&](const char* k) { return std::atoll(rs.meta_str(k).c_str()); };
    const int64_t T = mi("T"), F = mi("F_lat"), H = mi("H"), W = mi("W"), A = mi("A");
    const float tv = float(std::atof(rs.meta_str("tv").c_str())), ta = float(std::atof(rs.meta_str("ta").c_str()));
    H3Layout L = h3_build_layout(T, F, H, W, A, {});
    auto pos_ref = host_f32(rs.get("positions"));
    double pmax = 0;
    for (size_t i = 0; i < pos_ref.size(); i++) pmax = std::max(pmax, double(std::fabs(pos_ref[i] - L.positions[i])));
    int fails = 0;
    bool pos_ok = pos_ref.size() == L.positions.size() && pmax < 1e-4;
    fails += !pos_ok;
    std::printf("打包版面：S=%lld、RoPE 座標最大差 %.2e → %s\n", (long long)L.seq.S, pmax, pos_ok ? "PASS" : "FAIL");

    const int nl = int(mi("layers"));
    H3Dit dit(st, 0, nl);
    Ctx c(0, dit.act_bytes(L.seq.S), dit.scratch_bytes());
    const WeightRef& tr = rs.get("txt");
    DevBuf txt(tr.nbytes, 0);
    MX_CUDA(cudaMemcpy(txt.get(), tr.data, tr.nbytes, cudaMemcpyHostToDevice));
    dit.prepare(c, L.seq, L.positions, txt.as<bf16>());
    auto vin = host_f32(rs.get("video")), ain = host_f32(rs.get("audio"));
    DevBuf dv(vin.size() * 4, 0), da(ain.size() * 4, 0), ov(vin.size() * 4, 0), oa(ain.size() * 4, 0);
    MX_CUDA(cudaMemcpy(dv.get(), vin.data(), vin.size() * 4, cudaMemcpyHostToDevice));
    MX_CUDA(cudaMemcpy(da.get(), ain.data(), ain.size() * 4, cudaMemcpyHostToDevice));
    dit.forward(c, L.seq, dv.as<float>(), da.as<float>(), {tv, ta, std::max(tv, 0.999f)}, ov.as<float>(),
                oa.as<float>());
    c.sync();
    std::vector<float> gv(vin.size()), ga(ain.size());
    MX_CUDA(cudaMemcpy(gv.data(), ov.get(), gv.size() * 4, cudaMemcpyDeviceToHost));
    MX_CUDA(cudaMemcpy(ga.data(), oa.get(), ga.size() * 4, cudaMemcpyDeviceToHost));
    double ev = rel(gv, host_f32(rs.get("v_video"))), ea = rel(ga, host_f32(rs.get("v_audio")));
    std::printf("影片速度：相對誤差 %.4f → %s\n", ev, ev < 0.05 ? "PASS" : "FAIL");
    std::printf("音訊速度：相對誤差 %.4f → %s\n", ea, ea < 0.05 ? "PASS" : "FAIL");
    fails += ev >= 0.05;
    fails += ea >= 0.05;

    // 序列平行（各算一半列與一半頭）。只切列會改變 GEMM 的 M，cuBLAS 換 kernel 後 bf16 捨入不同（音訊對此很敏感），
    // 所以拿「分段 = S/2 的單卡」當基準：同一張卡上兩邊 GEMM 形狀完全相同，結果應幾乎一致。
    const int64_t half = L.seq.S / 2;
    auto read = [](const DevBuf& b, size_t n) {
      std::vector<float> h(n);
      MX_CUDA(cudaMemcpy(h.data(), b.get(), n * 4, cudaMemcpyDeviceToHost));
      return h;
    };
    const std::vector<float> ts4 = {tv, ta, std::max(tv, 0.999f), 1.f};
    dit.set_chunk(half);
    DevBuf bv(vin.size() * 4, 0), ba(ain.size() * 4, 0);
    dit.forward(c, L.seq, dv.as<float>(), da.as<float>(), ts4, bv.as<float>(), ba.as<float>());
    c.sync();
    const auto base_v = read(bv, vin.size()), base_a = read(ba, ain.size());
    dit.set_chunk(H3Dit::kMaxChunk);
    auto run_sp = [&](int dev2, double tol) {
      H3Dit dit2(st, dev2, nl);
      H3Seq seq2 = L.seq.layout();
      Ctx ca(0, dit.act_bytes_sp(L.seq.S, half), dit.scratch_bytes());
      Ctx cb(dev2, dit2.act_bytes_sp(L.seq.S, L.seq.S - half) + (vin.size() + ain.size()) * 4, dit2.scratch_bytes());
      dit.prepare(ca, L.seq, L.positions, txt.as<bf16>());
      DevBuf txt2(tr.nbytes, dev2);
      MX_CUDA(cudaMemcpy(txt2.get(), tr.data, tr.nbytes, cudaMemcpyHostToDevice));
      dit2.prepare(cb, seq2, L.positions, txt2.as<bf16>());
      DevBuf sv(vin.size() * 4, 0), sa(ain.size() * 4, 0);
      H3Dit::forward_sp(dit, ca, L.seq, dit2, cb, seq2, dv.as<float>(), da.as<float>(), ts4, sv.as<float>(),
                        sa.as<float>());
      ca.sync();
      const double e1 = rel(read(sv, vin.size()), base_v), e2 = rel(read(sa, ain.size()), base_a);
      const bool ok = e1 < tol && e2 < tol;
      std::printf("序列平行 GPU0+GPU%d：對同分段單卡 影片 %.5f／音訊 %.5f → %s\n", dev2, e1, e2, ok ? "PASS" : "FAIL");
      fails += !ok;
    };
    run_sp(0, 1e-4);
    int ngpu = 0;
    MX_CUDA(cudaGetDeviceCount(&ngpu));
    // 跨架構（例如 sm_120 + sm_86）的 kernel 不同，只能抓傳輸／分工錯誤（錯的話誤差接近 100%）。
    if (ngpu > 1) run_sp(1, 0.2);
    return fails ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("FAIL：%s\n", e.what());
    return 1;
  }
}
