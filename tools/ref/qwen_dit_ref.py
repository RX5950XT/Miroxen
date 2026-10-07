"""產生 Qwen-Image-2.1 DiT 的數值比對資料（diffusers 官方實作 + 同一份 GGUF 權重）。

用法：python tools/ref/qwen_dit_ref.py [gguf路徑] [輸出.safetensors] [edit]
輸出包含輸入（txt、lat1、lat2）與兩步的輸出（out1＝prefill+寫快取，out2＝讀快取）。
第三個參數為 edit 時加入兩張條件圖（cond0、cond1，與文字交錯），測編輯用的 block-causal 版面。
"""
import sys

import gguf
import numpy as np
import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from safetensors.torch import save_file

src = sys.argv[1] if len(sys.argv) > 1 else "models/image/qwen-image-2.1/qwen_image_2.1-Q8_0.gguf"
dst = sys.argv[2] if len(sys.argv) > 2 else "models/ref/qwen_dit_ref.safetensors"


def load_gguf(path):
    r = gguf.GGUFReader(path)
    sd = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]
        arr = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        name = t.name.removeprefix("model.diffusion_model.")
        w = torch.from_numpy(arr).to(torch.bfloat16)
        if name.endswith("img_mlp.gate_up.weight"):  # sd.cpp：前半 gate、後半 proj
            g, p = w.chunk(2, dim=0)
            base = name.removesuffix("gate_up.weight")
            sd[base + "gate_layer.weight"] = g.contiguous()
            sd[base + "proj.weight"] = p.contiguous()
        else:
            sd[name] = w
    return sd


torch.manual_seed(0)
sd = load_gguf(src)
model = QwenImage21Transformer2DModel(
    attention_head_dim=128, axes_dims_rope=(16, 56, 56), context_in_dim=4096, in_channels=64,
    num_attention_heads=32, num_layers=32, out_channels=64, patch_size=1, mlp_ratio=3, eps=1e-6,
    causal_condition=True,
).to(torch.bfloat16)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
del sd
dev = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(dev)
except torch.OutOfMemoryError:
    dev = "cpu"
    model.to(dev)
print("device", dev)

edit = len(sys.argv) > 3 and sys.argv[3] == "edit"
H, W = 16, 20
N = H * W
# 條件圖：(latent 高, 寬)，前面各有一段文字；最後再一段文字。
conds = [(8, 12), (10, 6)] if edit else []
pre_text, tail_text = [10, 5], 9 if edit else 24
slots, mask_parts, pos = [], [], 0
for (ch, cw), tl in zip(conds, pre_text):
    mask_parts += [torch.zeros(tl, dtype=torch.bool), torch.ones(ch * cw // 4, dtype=torch.bool)]
    slots.append(pos + tl)
    pos += tl + ch * cw // 4
mask_parts.append(torch.zeros(tail_text, dtype=torch.bool))
vlm_mask = torch.cat(mask_parts)
T = vlm_mask.numel()
txt = (torch.randn(1, T, 4096) * 4).to(torch.bfloat16)
cond_lat = [torch.randn(1, ch * cw, 64) for ch, cw in conds]
lat1 = torch.randn(1, N, 64)
lat2 = torch.randn(1, N, 64)
img_mask = torch.cat([vlm_mask[None], torch.ones(1, N // 4, dtype=torch.bool)], 1)
shapes = [[*[(1, ch, cw) for ch, cw in conds], (1, H, W)]]
cache = QwenImage21KVCache(32)


def run(lat, t, mode):
    with torch.no_grad():
        return model(
            hidden_states=torch.cat(cond_lat + [lat], 1).to(dev, torch.bfloat16), encoder_hidden_states=txt.to(dev),
            timestep=torch.tensor([t], device=dev), img_shapes=shapes, img_mask=img_mask.to(dev),
            kv_cache=cache, kv_cache_mode=mode, return_dict=False,
        )[0][:, -N:].float().cpu()


out1 = run(lat1, 0.7, "extract")
out2 = run(lat2, 0.45, "cached")
print("out1", out1.abs().mean().item(), "out2", out2.abs().mean().item())
import os

os.makedirs(os.path.dirname(dst), exist_ok=True)
tensors = {"txt": txt[0].contiguous(), "lat1": lat1[0].contiguous(), "lat2": lat2[0].contiguous(),
           "out1": out1[0].contiguous(), "out2": out2[0].contiguous()}
meta = {"T": str(T), "H": str(H), "W": str(W), "t1": "0.7", "t2": "0.45", "conds": str(len(conds))}
for i, ((ch, cw), s) in enumerate(zip(conds, slots)):
    tensors[f"cond{i}"] = cond_lat[i][0].contiguous()
    meta[f"cond{i}"] = f"{s},{ch},{cw}"
save_file(tensors, dst, metadata=meta)
print("saved", dst)
