"""產生 Qwen3-VL 視覺塔 + 圖文編碼的比對資料（transformers 官方實作 + llama.cpp GGUF 權重，CPU float32）。

用法：python tools/ref/qwen3_vl_ref.py [圖片] [輸出.safetensors]
輸出：rgb（輸入像素，已裁成 32 的倍數）、embeds / ds0..ds2（視覺塔輸出）、
      ids（Qwen-Image 編輯範本 tokenize 結果）、hidden（最後一層輸出、未套最終 norm，已去掉 system 前綴）。
"""
import json
import os
import sys

import gguf
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen3VLProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

args = sys.argv[1:] + [None] * 2
img_path = args[0] or "out/images/ref_input.png"
dst = args[1] or "models/ref/qwen3_vl_ref.safetensors"
ROOT = "models/image/qwen-image-2.1"

img = Image.open(img_path).convert("RGB")
w, h = img.size
w, h = min(w, 512) // 32 * 32, min(h, 384) // 32 * 32
img = img.crop((0, 0, w, h))
rgb = np.asarray(img).copy()

SYS = "Comprehend and analyze the provided prompt."
prompt = "把背景換成夕陽下的海灘，保留人物"
text = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|>"
        f"<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n")
proc = Qwen3VLProcessor.from_pretrained(f"{ROOT}/hf/processor")
inputs = proc(text=[text], images=[img], return_tensors="pt")
ids = inputs.input_ids[0]
drop_idx = len(proc.tokenizer.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n", add_special_tokens=False))
print("tokens", ids.numel(), "grid", inputs.image_grid_thw.tolist(), "drop", drop_idx)

cfg = Qwen3VLConfig(**json.load(open(f"{ROOT}/hf/text_encoder/config.json")))
with torch.device("meta"):
    model = Qwen3VLModel(cfg)


def load(path):
    r = gguf.GGUFReader(path)
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]
        yield t.name, torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape))


sd = {}
TM = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
      "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
      "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
      "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
for name, wt in load(f"{ROOT}/Qwen3VL-8B-Instruct-Q8_0.gguf"):
    if name == "token_embd.weight":
        sd["language_model.embed_tokens.weight"] = wt
    elif name == "output_norm.weight":
        sd["language_model.norm.weight"] = wt
    elif name.startswith("blk."):
        _, i, rest = name.split(".", 2)
        kind, suf = rest.rsplit(".", 1)
        sd[f"language_model.layers.{i}.{TM[kind]}.{suf}"] = wt

VM = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
      "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
patch = {}
ds_idx = {8: 0, 16: 1, 24: 2}
for name, wt in load(f"{ROOT}/mmproj-Qwen3VL-8B-Instruct-F16.gguf"):
    if name.startswith("v.patch_embd.weight"):
        patch[name] = wt
    elif name == "v.patch_embd.bias":
        sd["visual.patch_embed.proj.bias"] = wt
    elif name == "v.position_embd.weight":
        sd["visual.pos_embed.weight"] = wt
    elif name.startswith("v.blk."):
        _, _, i, kind, suf = name.split(".")
        sd[f"visual.blocks.{i}.{VM[kind]}.{suf}"] = wt
    elif name.startswith("v.post_ln."):
        sd[f"visual.merger.norm.{name.split('.')[-1]}"] = wt
    elif name.startswith("mm."):
        _, k, suf = name.split(".")
        sd[f"visual.merger.linear_fc{1 if k == '0' else 2}.{suf}"] = wt
    elif name.startswith("v.deepstack."):
        _, _, layer, kind, suf = name.split(".")
        kind = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}[kind]
        sd[f"visual.deepstack_merger_list.{ds_idx[int(layer)]}.{kind}.{suf}"] = wt
sd["visual.patch_embed.proj.weight"] = torch.stack([patch["v.patch_embd.weight"], patch["v.patch_embd.weight.1"]], 2)
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
assert not missing and not unexpected, (missing, unexpected)
DT = torch.bfloat16 if os.environ.get("MX_BF16") else torch.float32
model = model.to(DT).eval()
model.language_model.rotary_emb = type(model.language_model.rotary_emb)(cfg.text_config)
model.config._attn_implementation = "sdpa"
model.visual.rotary_pos_emb = type(model.visual.rotary_pos_emb)(cfg.vision_config)

with torch.no_grad():
    vis = model.get_image_features(inputs.pixel_values.to(DT), inputs.image_grid_thw, return_dict=True)
    captured = {}
    hook = model.language_model.norm.register_forward_hook(lambda m, a, o: captured.setdefault("pre", a[0]))
    model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values.to(DT),
          image_grid_thw=inputs.image_grid_thw, mm_token_type_ids=inputs.mm_token_type_ids, use_cache=False)
    hook.remove()
hidden = captured["pre"][0, drop_idx:].float()
out = {"rgb": torch.from_numpy(rgb), "ids": ids.to(torch.int32), "hidden": hidden.contiguous(),
       "embeds": vis.pooler_output[0].float().contiguous()}
for i, f in enumerate(vis.deepstack_features):
    out[f"ds{i}"] = f[0].float().contiguous()
print("embeds", out["embeds"].shape, "hidden", hidden.shape)
os.makedirs(os.path.dirname(dst), exist_ok=True)
save_file(out, dst, metadata={"drop_idx": str(drop_idx), "h": str(h), "w": str(w), "text": text})
print("saved", dst)
