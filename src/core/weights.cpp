#include "core/weights.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "core/dequant.h"
#include "kernels/ops.h"

namespace mx {

static DevBuf upload_raw(const WeightRef& r, int dev) {
  DevBuf b(r.nbytes, dev);
  MX_CUDA(cudaMemcpy(b.get(), r.data, r.nbytes, cudaMemcpyHostToDevice));
  return b;
}

static DevBuf upload_concat(const WeightRef& a, const WeightRef& b, int dev) {
  MX_CHECK(a.shape.size() == 2 && b.shape.size() == 2 && a.shape[1] == b.shape[1] && a.type == b.type &&
           a.shape[1] % gtype_block(a.type) == 0 && dequant_supported(a.type),
           "張量 %s 與 %s 無法接合：形狀、型別或區塊不相容", a.name.c_str(), b.name.c_str());
  DevBuf out(a.nbytes + b.nbytes, dev);
  MX_CUDA(cudaMemcpy(out.get(), a.data, a.nbytes, cudaMemcpyHostToDevice));
  MX_CUDA(cudaMemcpy(static_cast<char*>(out.get()) + a.nbytes, b.data, b.nbytes, cudaMemcpyHostToDevice));
  return out;
}

const DevTensor& DeviceWeights::raw_concat(const std::string& name, const std::string& first,
                                          const std::string& second) {
  MX_CHECK(!st_.has(name), "張量 %s 已存在，不能覆寫為接合張量", name.c_str());
  auto it = raw_.find(name);
  if (it != raw_.end()) {
    MX_CHECK(concat_sources_.at(name) == std::make_pair(first, second), "張量 %s 接合來源不同", name.c_str());
    return it->second.second;
  }
  const auto& a = st_.get(first);
  const auto& b = st_.get(second);
  DevBuf buf = upload_concat(a, b, dev_);
  DevTensor t{buf.get(), a.type, {a.shape[0] + b.shape[0], a.shape[1]}};
  concat_sources_[name] = {first, second};
  bytes_ += buf.size();
  auto& slot = raw_[name];
  slot = {std::move(buf), std::move(t)};
  return slot.second;
}

const DevTensor& DeviceWeights::raw(const std::string& name) {
  auto it = raw_.find(name);
  if (it != raw_.end()) return it->second.second;
  const WeightRef& r = st_.get(name);
  MX_CHECK(dequant_supported(r.type), "張量 %s 的型別 %s 不支援", name.c_str(), gtype_name(r.type));
  DevBuf b = upload_raw(r, dev_);
  DevTensor t{b.get(), r.type, r.shape};
  bytes_ += r.nbytes;
  auto& slot = raw_[name];
  slot.first = std::move(b);
  slot.second = t;
  return slot.second;
}

const float* DeviceWeights::f32(const std::string& name) {
  auto it = f32_.find(name);
  if (it != f32_.end()) return it->second.as<float>();
  const WeightRef& r = st_.get(name);
  MX_CHECK(r.numel() % gtype_block(r.type) == 0, "張量 %s 大小不是區塊倍數", name.c_str());
  DevBuf tmp = upload_raw(r, dev_);
  DevBuf out(size_t(r.numel()) * sizeof(float), dev_);
  MX_CUDA(cudaSetDevice(dev_));
  dequant_to_f32(tmp.get(), r.type, r.numel(), out.as<float>(), nullptr);
  MX_CUDA(cudaDeviceSynchronize());
  bytes_ += out.size();
  return (f32_[name] = std::move(out)).as<float>();
}

void DeviceWeights::quantize(Linear& l, QMode m) {
  if (m == QMode::BF16 || l.qw || !l.w.p) return;
  MX_CHECK(l.in % 32 == 0, "%s：輸入維度 %lld 不是 32 的倍數，無法量化成 %s", l.name.c_str(), (long long)l.in,
           qmode_name(m));
  MX_CUDA(cudaSetDevice(dev_));
  DevBuf tmp(size_t(l.in) * l.out * sizeof(bf16), dev_);
  dequant_to_bf16(l.w.p, l.w.type, l.in * l.out, tmp.as<bf16>(), nullptr);
  auto q = std::make_shared<QWeight>();
  q->mode = m;
  q->q = DevBuf(qdata_bytes(m, l.out, l.in), dev_);
  q->sf = DevBuf(qscale_bytes(m, l.out, l.in), dev_);
  q->amax = DevBuf(sizeof(float), dev_);
  q->m = {q->q.get(), q->sf.as<uint8_t>(), q->amax.as<float>()};
  quantize_blockscaled(m, tmp.as<bf16>(), l.in, l.out, l.in, q->mat(), nullptr);
  MX_CUDA(cudaDeviceSynchronize());
  bytes_ += q->q.size() + q->sf.size();
  auto it = raw_.find(l.name + ".weight");
  if (it != raw_.end()) {
    bytes_ -= it->second.first.size();
    raw_.erase(it);
  }
  l.w.p = nullptr;
  l.qw = std::move(q);
}

void DeviceWeights::merge_loras(Ctx& c, Linear& l) {
  if (!l.qw || !l.qw->q.get() || (l.loras.empty() && !l.qw->merged)) return;
  c.use();
  const auto src = concat_sources_.find(l.name + ".weight");
  DevBuf raw = src == concat_sources_.end() ? upload_raw(st_.get(l.name + ".weight"), dev_)
      : upload_concat(st_.get(src->second.first), st_.get(src->second.second), dev_);
  DevBuf tmp(size_t(l.in) * l.out * sizeof(bf16), dev_);
  dequant_to_bf16(raw.get(), l.w.type, l.in * l.out, tmp.as<bf16>(), c.stream);
  const float one = 1.f;
  for (auto& d : l.loras) {
    if (d->f32) continue;
    // row-major W[out,in] += s·B[out_len,r]·A[r,in]；column-major 下是 Wᵀ[in,out] += Aᵀ[in,r]·Bᵀ[r,out]。
    MX_BLAS(cublasGemmEx(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, int(l.in), int(d->out_len), int(d->rank), &d->scale,
                         d->A.get(), CUDA_R_16BF, int(l.in), d->B.get(), CUDA_R_16BF, int(d->rank), &one,
                         tmp.as<bf16>() + d->out_off * l.in, CUDA_R_16BF, int(l.in), CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT));
  }
  quantize_blockscaled(l.qw->mode, tmp.as<bf16>(), l.in, l.out, l.in, l.qw->mat(), c.stream);
  c.sync();
  for (auto& d : l.loras)
    if (!d->f32) d->A = DevBuf(), d->B = DevBuf();  // 已併入，釋放 VRAM
  l.qw->merged = !l.loras.empty();
}

Linear make_linear(DeviceWeights& dw, const std::string& prefix) {
  Linear l;
  l.name = prefix;
  l.w = dw.raw(prefix + ".weight");
  MX_CHECK(l.w.shape.size() == 2, "%s.weight 不是 2 維", prefix.c_str());
  l.out = l.w.shape[0];
  l.in = l.w.shape[1];
  l.bias = dw.f32_or_null(prefix + ".bias");
  return l;
}

Linear make_linear_concat(DeviceWeights& dw, const std::string& prefix,
                          const std::string& first, const std::string& second) {
  MX_CHECK(!dw.store().has(first + ".bias") && !dw.store().has(second + ".bias"),
           "%s：接合線性層不支援 bias", prefix.c_str());
  Linear l;
  l.name = prefix;
  l.w = dw.raw_concat(prefix + ".weight", first + ".weight", second + ".weight");
  l.out = l.w.shape[0];
  l.in = l.w.shape[1];
  return l;
}

size_t linear_scratch_bytes(const Linear& l) { return l.qw ? 0 : size_t(l.in) * l.out * sizeof(float); }

static const bf16* weight_bf16(Ctx& c, const Linear& l) {
  if (l.w.type == GType::BF16) return static_cast<const bf16*>(l.w.p);
  size_t need = size_t(l.in) * l.out * sizeof(bf16);
  MX_CHECK(need <= c.scratch.size(), "反量化暫存不足：%s 需要 %.1f MiB", l.name.c_str(),
           need / 1048576.0);
  dequant_to_bf16(l.w.p, l.w.type, l.in * l.out, c.scratch.as<bf16>(), c.stream);
  return c.scratch.as<bf16>();
}

// xq：已量化的 x（l 有 qw 時必須提供）。
static void linear_one(Ctx& c, const Linear& l, const bf16* x, const QMat* xq, int64_t M, bf16* y,
                       int64_t ldy) {
  if (!ldy) ldy = l.out;
  if (l.qw) gemm_blockscaled(c, l.qw->mode, l.qw->mat(), *xq, y, ldy, M, l.out, l.in);
  else gemm_bf16(c, x, l.in, weight_bf16(c, l), l.in, y, false, ldy, M, l.out, l.in);
  if (l.bias) {
    MX_CHECK(ldy == l.out, "%s：有 bias 時不支援自訂列距", l.name.c_str());
    add_bias_bf16(y, l.bias, M, l.out, c.stream);
  }
  for (auto& d : l.loras) {
    if (l.qw && l.qw->merged) break;
    MX_CHECK(!d->f32, "%s：LoRA 是 f32 版本", l.name.c_str());
    ArenaScope sc(c.act);
    bf16* t = c.act.alloc_n<bf16>(size_t(M) * d->rank);
    gemm_bf16(c, x, l.in, d->A.as<bf16>(), l.in, t, false, d->rank, M, d->rank, l.in);
    gemm_bf16(c, t, d->rank, d->B.as<bf16>(), d->rank, y + d->out_off, false, ldy, M, d->out_len,
              d->rank, 1.f, d->scale);
  }
}

void linear_multi(Ctx& c, const bf16* x, int64_t M, std::initializer_list<LinearOut> outs) {
  if (M == 0) return;
  const QWeight* qw = nullptr;
  for (auto& o : outs) {
    if (!o.l->qw) continue;
    MX_CHECK(!qw || (qw->mode == o.l->qw->mode && o.l->in == outs.begin()->l->in), "%s：量化模式或輸入維度不一致",
             o.l->name.c_str());
    qw = o.l->qw.get();
  }
  ArenaScope sc(c.act);
  QMat xq;
  if (qw) {
    const int64_t K = outs.begin()->l->in;
    xq = {c.act.alloc(qdata_bytes(qw->mode, M, K)), c.act.alloc_n<uint8_t>(qscale_bytes(qw->mode, M, K)),
          c.act.alloc_n<float>(1)};
    quantize_blockscaled(qw->mode, x, K, M, K, xq, c.stream);
  }
  for (auto& o : outs) linear_one(c, *o.l, x, &xq, M, o.y, o.ldy);
}

void linear(Ctx& c, const Linear& l, const bf16* x, int64_t M, bf16* y, int64_t ldy) {
  linear_multi(c, x, M, {{&l, y, ldy}});
}

void linear_f32(Ctx& c, const Linear& l, const float* x, int64_t M, float* y) {
  if (M == 0) return;
  MX_CHECK(!l.qw, "%s：f32 路徑不支援量化權重", l.name.c_str());
  const float* w;
  if (l.w.type == GType::F32) {
    w = static_cast<const float*>(l.w.p);
  } else {
    MX_CHECK(linear_scratch_bytes(l) <= c.scratch.size(), "反量化暫存不足：%s", l.name.c_str());
    dequant_to_f32(l.w.p, l.w.type, l.in * l.out, c.scratch.as<float>(), c.stream);
    w = c.scratch.as<float>();
  }
  gemm_f32(c, x, l.in, w, l.in, y, l.out, M, l.out, l.in);
  if (l.bias) add_bias_f32(y, l.bias, M, l.out, c.stream);
  for (auto& d : l.loras) {
    MX_CHECK(d->f32, "%s：LoRA 不是 f32 版本", l.name.c_str());
    ArenaScope sc(c.act);
    float* t = c.act.alloc_n<float>(size_t(M) * d->rank);
    gemm_f32(c, x, l.in, d->A.as<float>(), l.in, t, d->rank, M, d->rank, l.in);
    gemm_f32(c, t, d->rank, d->B.as<float>(), d->rank, y + d->out_off, l.out, M, d->out_len, d->rank, 1.f, d->scale);
  }
}

// ---------------- LoRA ----------------

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string strip_lora_prefix(std::string n) {
  static const char* kPrefixes[] = {"base_model.model.", "model.diffusion_model.",
                                    "diffusion_model.", "transformer.", "unet.", "model."};
  for (bool again = true; again;) {
    again = false;
    for (auto p : kPrefixes) {
      std::string ps = p;
      if (n.rfind(ps, 0) == 0) {
        n = n.substr(ps.size());
        again = true;
      }
    }
  }
  return n;
}

static DevBuf to_dev(const WeightRef& r, int dev, bool f32) {
  if (!f32 && r.type == GType::BF16) return upload_raw(r, dev);  // 已是 bf16：直接複製，省一次反量化與同步
  DevBuf tmp = upload_raw(r, dev);
  DevBuf out(size_t(r.numel()) * (f32 ? sizeof(float) : sizeof(bf16)), dev);
  if (f32) dequant_to_f32(tmp.get(), r.type, r.numel(), out.as<float>(), nullptr);
  else dequant_to_bf16(tmp.get(), r.type, r.numel(), out.as<bf16>(), nullptr);
  MX_CUDA(cudaDeviceSynchronize());
  return out;
}

LoraLoadStats apply_lora(const std::string& path, float user_scale, int device,
                         const std::vector<std::string>& known_names, const LoraResolver& resolve) {
  TensorStore st;
  st.add_file(path, "\x01");  // 不自動剝前綴，這裡自己處理
  std::unordered_map<std::string, std::string> kohya;
  for (auto& n : known_names) {
    std::string k = n;
    std::replace(k.begin(), k.end(), '.', '_');
    kohya[k] = n;
  }
  struct Pair {
    const WeightRef *down = nullptr, *up = nullptr, *alpha = nullptr;
  };
  std::map<std::string, Pair> pairs;
  // 標記後面可以接 PEFT adapter 名稱，例如 x.lora_A.default.weight。
  static const std::pair<const char*, int> kMarker[] = {
      {".lora_A.", 0}, {".lora_down.", 0}, {".lora.down.", 0}, {".lora_B.", 1}, {".lora_up.", 1}, {".lora.up.", 1}};
  LoraLoadStats stats;
  for (auto& name : st.names()) {
    bool hit = false;
    const WeightRef* r = &st.get(name);
    if (ends_with(name, ".alpha")) {
      pairs[name.substr(0, name.size() - 6)].alpha = r;
      continue;
    }
    for (auto& [mk, kind] : kMarker) {
      size_t pos = name.rfind(mk);
      if (pos == std::string::npos || !ends_with(name, "weight")) continue;
      Pair& p = pairs[name.substr(0, pos)];
      (kind == 0 ? p.down : p.up) = r;
      hit = true;
      break;
    }
    if (!hit) stats.unmatched.push_back(name);
  }
  const float meta_alpha = float(std::atof(st.meta_str("alpha", "0").c_str()));
  MX_CUDA(cudaSetDevice(device));
  for (auto& [base, p] : pairs) {
    if (!p.down || !p.up) {
      stats.unmatched.push_back(base + "（缺 down/up）");
      continue;
    }
    std::string n = strip_lora_prefix(base);
    for (const char* kp : {"lora_unet_", "lora_transformer_", "lora_te_"}) {
      if (n.rfind(kp, 0) == 0) {
        auto it = kohya.find(n.substr(strlen(kp)));
        n = it == kohya.end() ? n : it->second;
      }
    }
    LoraTarget t;
    if (!resolve(n, t) || !t.linear) {
      stats.unmatched.push_back(base);
      continue;
    }
    int64_t out_len = t.out_len < 0 ? t.linear->out : t.out_len;
    MX_CHECK(p.down->shape.size() == 2 && p.up->shape.size() == 2, "LoRA %s 形狀錯誤", base.c_str());
    int64_t r = p.down->shape[0];
    MX_CHECK(p.down->shape[1] == t.linear->in && p.up->shape[0] == out_len && p.up->shape[1] == r,
             "LoRA %s 形狀 [%lld,%lld]/[%lld,%lld] 與 %s (in %lld, out %lld) 不符", base.c_str(),
             (long long)p.down->shape[0], (long long)p.down->shape[1], (long long)p.up->shape[0],
             (long long)p.up->shape[1], t.linear->name.c_str(), (long long)t.linear->in,
             (long long)out_len);
    auto d = std::make_shared<LoraDelta>();
    d->rank = r;
    d->out_off = t.out_off;
    d->out_len = out_len;
    // 沒有逐層 alpha 時用檔案 metadata 的 alpha（LightX2V 的 H3 加速 LoRA：alpha 8、rank 128 → 縮放 1/16）。
    float alpha = meta_alpha > 0.f ? meta_alpha : float(r);
    if (p.alpha) {
      float a = 0;
      dequant_cpu(p.alpha->data, p.alpha->type, 1, &a);
      alpha = a;
    }
    d->scale = user_scale * alpha / float(r);
    d->f32 = t.f32;
    d->A = to_dev(*p.down, device, t.f32);
    d->B = to_dev(*p.up, device, t.f32);
    if (t.swap_halves) {  // B 的前後兩半輸出列對調（例如 diffusers 的 [value; gate] → checkpoint 的 [gate; value]）
      MX_CHECK(out_len % 2 == 0, "LoRA %s：輸出維度不是偶數，無法對調", base.c_str());
      const size_t half = size_t(out_len / 2) * r * (t.f32 ? 4 : 2);
      DevBuf sw(2 * half, device);
      MX_CUDA(cudaMemcpy(sw.get(), static_cast<char*>(d->B.get()) + half, half, cudaMemcpyDeviceToDevice));
      MX_CUDA(cudaMemcpy(static_cast<char*>(sw.get()) + half, d->B.get(), half, cudaMemcpyDeviceToDevice));
      d->B = std::move(sw);
    }
    t.linear->loras.push_back(d);
    stats.applied++;
  }
  MX_INFO("LoRA %s：套用 %d 層，未對應 %zu 個張量", path.c_str(), stats.applied,
          stats.unmatched.size());
  return stats;
}

void clear_loras(std::vector<Linear*>& linears) {
  for (auto* l : linears) l->loras.clear();
}

}  // namespace mx
