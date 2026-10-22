# GGUF 文字編碼器沒有視覺塔，ComfyUI 認不出是 Qwen3-VL → 依權重形狀與 CLIP 類型補判斷（文生圖／文生影片用不到視覺塔）
p = "/content/ComfyUI/comfy/sd.py"
s = open(p).read()
a = '''    if "model.layers.0.post_attention_layernorm.weight" in sd:
        weight = sd['model.layers.0.post_attention_layernorm.weight']'''
b = '''    if "model.layers.49.self_attn.q_proj.weight" in sd and "model.layers.50.self_attn.q_proj.weight" not in sd and 'model.layers.0.self_attn.q_norm.weight' in sd and sd['model.layers.0.post_attention_layernorm.weight'].shape[0] == 5120:
        return TEModel.QWEN3VL_32B  # GGUF patch: H3 TE without vision
''' + a
assert s.count(a) == 1; s = s.replace(a, b)
a = '''        elif te_model == TEModel.QWEN3_8B:
            if clip_type == CLIPType.IDEOGRAM4:'''
b = '''        elif te_model == TEModel.QWEN3_8B and clip_type == CLIPType.QWEN_IMAGE:  # GGUF patch: Qwen-Image 2.1 TE without vision
            clip_target.clip = comfy.text_encoders.qwen_image21.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.qwen_image21.QwenImage21Tokenizer
''' + a
assert s.count(a) == 1; s = s.replace(a, b)
open(p, "w").write(s)
print("PATCHED")
