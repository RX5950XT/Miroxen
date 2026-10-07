"""產生 Qwen3-VL 文字編碼器比對資料：transformers 官方實作 + llama.cpp GGUF 權重（CPU float32）。

用法：python tools/ref/qwen3_text_ref.py [gguf] [tokenizer.json] [config.json] [輸出.safetensors] [層數上限]
輸出：ids（Qwen-Image 範本 tokenize 結果）、drop_idx、hidden（最後一層輸出、未套最終 norm）。
"""
import json
import sys

import gguf
import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

args = sys.argv[1:] + [None] * 5
src = args[0] or "models/image/qwen-image-2.1/Qwen3VL-8B-Instruct-Q8_0.gguf"
tok_path = args[1] or "models/image/qwen-image-2.1/hf/processor/tokenizer.json"
cfg_path = args[2] or "models/image/qwen-image-2.1/hf/text_encoder/config.json"
dst = args[3] or "models/ref/qwen3_text_ref.safetensors"

SYS = "Comprehend and analyze the provided prompt."
prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, 雨夜的霓虹燈招牌"
text = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
tok = Tokenizer.from_file(tok_path)
ids = tok.encode(text, add_special_tokens=False).ids
drop_idx = len(tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n", add_special_tokens=False).ids)
print("tokens", len(ids), "drop_idx", drop_idx)

cfg = json.load(open(cfg_path))
tcfg = Qwen3VLTextConfig(**cfg["text_config"])
r = gguf.GGUFReader(src)
n_layers = sum(1 for t in r.tensors if t.name.endswith(".attn_q.weight"))
tcfg.num_hidden_layers = n_layers
tcfg.torch_dtype = torch.float32
model = Qwen3VLTextModel(tcfg).float().eval()

M = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
     "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
     "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
     "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
sd = {}
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape.tolist())]
    w = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape))
    if t.name == "token_embd.weight":
        sd["embed_tokens.weight"] = w
    elif t.name == "output_norm.weight":
        sd["norm.weight"] = w
    elif t.name.startswith("blk."):
        _, i, rest = t.name.split(".", 2)
        kind = rest.removesuffix(".weight")
        sd[f"layers.{i}.{M[kind]}.weight"] = w
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if not m.startswith("norm.")]
assert not missing and not unexpected, (missing, unexpected)

captured = {}
model.norm.register_forward_hook(lambda m, a, o: captured.setdefault("pre", a[0]))
with torch.no_grad():
    model(input_ids=torch.tensor([ids]), use_cache=False)
hidden = captured["pre"][0].float()
print("hidden", hidden.shape, hidden.abs().mean().item())
import os

os.makedirs(os.path.dirname(dst), exist_ok=True)
save_file({"ids": torch.tensor(ids, dtype=torch.int32), "hidden": hidden.contiguous()}, dst,
          metadata={"drop_idx": str(drop_idx), "text": text})
print("saved", dst)
