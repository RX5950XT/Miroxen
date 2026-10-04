"""用 diffusers 官方影片 VAE 解碼引擎存下的最終 latent（MX_H3_DUMP），隔離 DiT 與 VAE 的問題。

用法：python tools/ref/h3_decode_dump.py <dump 目錄> <輸出.png>
輸出：第 0、中間、最後一格並排的 PNG。
"""
import json
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3
from PIL import Image

dump, dst = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(dump, "meta.json")))
F, H, W = m["F_lat"], m["H"], m["W"]
src = sys.argv[3] if len(sys.argv) > 3 else "video_final.bin"  # 也可給 ref_final.pt（參考迴圈的結果）
if src.endswith(".pt"):
    rows = torch.load(os.path.join(dump, src))["video"].float().reshape(-1, 96)
else:
    rows = torch.from_numpy(np.fromfile(os.path.join(dump, src), dtype=np.float32)).reshape(-1, 96)
# 反 patchify：rows (f, h/2, w/2) × (c, pt=1, ph, pw) → [1, 24, F, H, W]
lat = rows.reshape(F, H // 2, W // 2, 24, 1, 2, 2).permute(3, 0, 4, 1, 5, 2, 6).reshape(1, 24, F, H, W)
vae = AutoencoderKLMiniMaxH3.from_pretrained("models/minimax-h3/hf/vae", torch_dtype=torch.bfloat16).to("cuda")
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
with torch.no_grad():
    video = vae.decode((lat * std + mean).to("cuda", torch.bfloat16), return_dict=False)[0].float().cpu()
pm = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
ps = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
video = (video * ps + pm).clamp(0, 1)[0]  # [3, frames, h, w]
n = video.shape[1]
print("frames", n, "size", tuple(video.shape[2:]))
tiles = [video[:, i].permute(1, 2, 0).numpy() for i in (0, n // 2, n - 1)]
img = (np.concatenate(tiles, axis=1) * 255).round().astype(np.uint8)
Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)).save(dst)
print("saved", dst)
