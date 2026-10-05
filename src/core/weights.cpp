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
  MX_CHECK(l.loras.empty(), "%s：f32 路徑不支援 LoRA", l.name.c_str());
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

static DevBuf to_bf16_dev(const WeightRef& r, int dev) {
  DevBuf tmp = upload_raw(r, dev);
  DevBuf out(size_t(r.numel()) * sizeof(bf16), dev);
  dequant_to_bf16(tmp.get(), r.type, r.numel(), out.as<bf16>(), nullptr);
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
  static const std::pair<const char*, int> kSuffix[] = {
      {".lora_A.weight", 0}, {".lora_down.weight", 0}, {".lora.down.weight", 0},
      {".lora_B.weight", 1}, {".lora_up.weight", 1},   {".lora.up.weight", 1},
      {".alpha", 2}};
  LoraLoadStats stats;
  for (auto& name : st.names()) {
    bool hit = false;
    for (auto& [suf, kind] : kSuffix) {
      if (!ends_with(name, suf)) continue;
      Pair& p = pairs[name.substr(0, name.size() - strlen(suf))];
      const WeightRef* r = &st.get(name);
      (kind == 0 ? p.down : kind == 1 ? p.up : p.alpha) = r;
      hit = true;
      break;
    }
    if (!hit) stats.unmatched.push_back(name);
  }
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
    float alpha = float(r);
    if (p.alpha) {
      float a = 0;
      dequant_cpu(p.alpha->data, p.alpha->type, 1, &a);
      alpha = a;
    }
    d->scale = user_scale * alpha / float(r);
    d->A = to_bf16_dev(*p.down, device);
    d->B = to_bf16_dev(*p.up, device);
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
