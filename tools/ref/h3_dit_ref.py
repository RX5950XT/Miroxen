"""MiniMax-H3 DiT 數值比對資料：diffusers 官方 MiniMaxH3Transformer3DModel + pruned GGUF 權重（CPU bf16）。

pruned GGUF 用 adaln_t_table（t → 8 維曲線基底，線性內插）取代 time_embedder，AdaLN 投影前不過 silu；
這裡以 monkeypatch 讓官方模型走同樣的路徑，其餘（打包、RoPE、注意力、SwiGLU）全用官方程式碼。
用法：python tools/ref/h3_dit_ref.py [gguf] [輸出.safetensors]
"""
import os
import sys

import gguf
import numpy as np
import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers import transformer_minimax_h3 as tm
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
from safetensors.torch import save_file

src = sys.argv[1] if len(sys.argv) > 1 else "models/minimax-h3/minimax_h3_fl2va_pruned-Q4_K.gguf"
dst = sys.argv[2] if len(sys.argv) > 2 else "models/ref/h3_dit_ref.safetensors"
torch.manual_seed(0)

r = gguf.GGUFReader(src)
raw = {}
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape.tolist())]
    w = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape))
    keep32 = any(t.name.startswith(p) for p in ("adaln_t_table", "rope.", "video_patch_proj", "audio_patch_proj",
                                                 "final_layer.video_out", "final_layer.audio_out")) or ".adaln_proj." in t.name
    raw[t.name] = w if keep32 else w.to(torch.bfloat16)
table = raw.pop("adaln_t_table")  # [grid, basis]
raw.pop("rope.inv_freq")
nb = sum(1 for k in raw if k.startswith("blocks.") and k.endswith("qkv_proj.weight"))
if len(sys.argv) > 3:  # 只用前 N 層（除錯）
    nb = int(sys.argv[3])
    raw = {k: v for k, v in raw.items() if not k.startswith("blocks.") or int(k.split(".")[1]) < nb}
H = 5376


def conv(k):
    if k.startswith("token_refiner.blocks."):
        k = k.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif k.startswith("blocks."):
        k = "transformer_blocks." + k[len("blocks."):]
    for a, b in [("video_patch_proj.", "proj_in."), ("audio_patch_proj.", "audio_proj_in."),
                 ("condition_proj.", "context_embedder."), ("final_layer.norm.", "norm_out.norm."),
                 ("final_layer.adaln_proj.linear.", "norm_out.linear."), ("final_layer.video_out.", "proj_out."),
                 ("final_layer.audio_out.", "audio_proj_out."), (".attn.q_norm.", ".attn.norm_q."),
                 (".attn.k_norm.", ".attn.norm_k."), (".attn.out_proj.", ".attn.to_out.0."),
                 (".mlp.fc2.", ".ff.net.2.")]:
        k = k.replace(a, b)
    return k


sd = {}
for k, w in raw.items():
    if k.endswith("attn.qkv_proj.weight"):
        q, kk, v = w.chunk(3, dim=0)  # GGUF 已是連續 [q; k; v]
        base = conv(k).removesuffix("qkv_proj.weight")
        sd[base + "to_q.weight"], sd[base + "to_k.weight"], sd[base + "to_v.weight"] = q, kk, v
    elif k.endswith("mlp.fc1.weight"):
        a, b = w.chunk(2, dim=0)  # 原始：silu(前半)*後半；diffusers SwiGLU：前半*silu(後半) → 交換
        sd[conv(k).replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat([b, a], 0)
    else:
        sd[conv(k)] = w
del raw

with torch.device("meta"):
    model = MiniMaxH3Transformer3DModel(time_embed_dim=table.shape[1], num_layers=nb)


class CurveEmbed(nn.Module):
    def __init__(self, tbl):
        super().__init__()
        self.register_buffer("tbl", tbl)
        self.dummy = nn.Parameter(torch.zeros(1))  # 讓 get_parameter_dtype 回傳 float32

    def forward(self, t):
        t = t.reshape(-1).float().clamp(0, 1) * (self.tbl.shape[0] - 1)
        lo = t.floor().long().clamp(max=self.tbl.shape[0] - 2)
        fr = (t - lo.float())[:, None]
        return self.tbl[lo] + (self.tbl[lo + 1] - self.tbl[lo]) * fr


model.time_proj = nn.Identity()
model.time_embedder = CurveEmbed(table)
tm.MiniMaxH3AdaLayerNormModulation.forward = lambda self, temb: self.linear(
    temb.to(self.linear.weight.dtype)).to(torch.bfloat16).view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def out_forward(self, hs, temb, ti):
    shift, scale = self.linear(temb.to(self.linear.weight.dtype)).to(torch.bfloat16).chunk(2, dim=-1)
    hs = self.norm(hs)
    return hs * (1.0 + scale.index_select(0, ti)) + shift.index_select(0, ti)


tm.MiniMaxH3AdaLayerNormOut.forward = out_forward
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
model.rope.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
missing = [m for m in missing if not m.startswith("time_embedder.") and "rope" not in m]
assert not missing and not unexpected, (missing[:10], unexpected[:10])
del sd
# 官方混合精度：patch 投影、輸出頭、time_embedder 為 fp32，其餘 bf16（載入時已照此轉型）。
model.eval()
FP32 = os.environ.get("MX_REF_FP32", "1") == "1"  # 預設 fp32：bf16 參考經 50 層放大後誤差達 5–20%，不適合當裁判
if FP32:  # fp32 計算（權重仍是 bf16 儲存、使用時升精度），當作「真值」參考
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            for name, prm in list(mod.named_parameters(recurse=False)):
                setattr(mod, name, nn.Parameter(prm.float(), requires_grad=False))
    nn.Linear.forward = lambda self, x: torch.nn.functional.linear(
        x.float(), self.weight.float(), None if self.bias is None else self.bias.float())
    tm.MiniMaxH3AdaLayerNormModulation.forward = lambda self, temb: self.linear(temb.float()).view(
        -1, 6 * self.hidden_size).chunk(6, dim=-1)

    def out_forward32(self, hs, temb, ti):
        shift, scale = self.linear(temb.float()).chunk(2, dim=-1)
        hs = self.norm(hs.float())
        return hs * (1.0 + scale.index_select(0, ti)) + shift.index_select(0, ti)

    tm.MiniMaxH3AdaLayerNormOut.forward = out_forward32

T, F_lat, LH, LW, A = 20, 2, 16, 28, 8
txt = (torch.randn(1, T, 5120) * 3).to(torch.bfloat16)
tags = torch.full((T,), 1, dtype=torch.long)
pos, token_tags, vidx, aidx, tidx, ncv, nca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    tags, F_lat, LH, LW, A, (1, 2, 2), 2, 2, 0)
lat = torch.randn(1, 24, F_lat, LH, LW)
vrows = patchify_video_latents(lat, (1, 2, 2))
arows = torch.randn(2 * A, 32)
tv, ta = 0.37, 0.62
uniq, tind = MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, ncv, nca, T, tv, ta, max(tv, 0.999), 1.0)
with torch.no_grad():
    v, a = model(hidden_states=vrows[None], audio_hidden_states=arows[None],
                 encoder_hidden_states=txt.float() if FP32 else txt,
                 timestep=uniq, timestep_indices=tind, token_tags=token_tags, position_ids=pos,
                 video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
print("video", v.float().abs().mean().item(), "audio", a.float().abs().mean().item())
import os

os.makedirs(os.path.dirname(dst), exist_ok=True)
save_file({"txt": txt[0].contiguous(), "video": vrows.float().contiguous(), "audio": arows.contiguous(),
           "positions": pos.float().contiguous(), "v_video": v[0].float().contiguous(),
           "v_audio": a[0].float().contiguous()}, dst,
          metadata={"T": str(T), "F_lat": str(F_lat), "H": str(LH), "W": str(LW), "A": str(A),
                    "tv": str(tv), "ta": str(ta), "layers": str(nb)})
print("saved", dst)
