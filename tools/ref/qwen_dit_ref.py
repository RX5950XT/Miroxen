"""產生 Qwen-Image-2.1 DiT 的數值比對資料（diffusers 官方實作 + 同一份 GGUF 權重）。

用法：python tools/ref/qwen_dit_ref.py [gguf路徑] [輸出.safetensors]
輸出包含輸入（txt、lat1、lat2）與兩步的輸出（out1＝prefill+寫快取，out2＝讀快取）。
"""
import sys

import gguf
import numpy as np
import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from safetensors.torch import save_file

src = sys.argv[1] if len(sys.argv) > 1 else "models/qwen-image-2.1/qwen_image_2.1-Q8_0.gguf"
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

T, H, W = 24, 16, 20
N = H * W
txt = (torch.randn(1, T, 4096) * 4).to(torch.bfloat16)
lat1 = torch.randn(1, N, 64)
lat2 = torch.randn(1, N, 64)
img_mask = torch.cat([torch.zeros(1, T, dtype=torch.bool), torch.ones(1, N // 4, dtype=torch.bool)], 1)
shapes = [[(1, H, W)]]
cache = QwenImage21KVCache(32)


def run(lat, t, mode):
    with torch.no_grad():
        return model(
            hidden_states=lat.to(dev, torch.bfloat16), encoder_hidden_states=txt.to(dev),
            timestep=torch.tensor([t], device=dev), img_shapes=shapes, img_mask=img_mask.to(dev),
            kv_cache=cache, kv_cache_mode=mode, return_dict=False,
        )[0][:, -N:].float().cpu()


out1 = run(lat1, 0.7, "extract")
out2 = run(lat2, 0.45, "cached")
print("out1", out1.abs().mean().item(), "out2", out2.abs().mean().item())
import os

os.makedirs(os.path.dirname(dst), exist_ok=True)
save_file(
    {"txt": txt[0].contiguous(), "lat1": lat1[0].contiguous(), "lat2": lat2[0].contiguous(),
     "out1": out1[0].contiguous(), "out2": out2[0].contiguous()},
    dst, metadata={"T": str(T), "H": str(H), "W": str(W), "t1": "0.7", "t2": "0.45"},
)
print("saved", dst)
