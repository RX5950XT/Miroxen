# 在 Colab VM 上裝 ComfyUI（固定 commit）＋ComfyUI-GGUF，模型直接連到 Miroxen 的同一批檔案，背景啟動在 :8190
set -euo pipefail
M=/content/miroxen/models
C=/content/ComfyUI
cd /content
[[ -d $C ]] || git clone -q https://github.com/comfyanonymous/ComfyUI $C
git -C $C checkout -q 7a28332
[[ -d $C/custom_nodes/ComfyUI-GGUF ]] || git clone -q https://github.com/city96/ComfyUI-GGUF $C/custom_nodes/ComfyUI-GGUF
git -C $C/custom_nodes/ComfyUI-GGUF checkout -q 6ea2651
# ComfyUI-GGUF 停在 2026-01，不認得 qwen_image21 與 H3（unsloth 檔沒有 architecture 欄位）→ 放行，讓 ComfyUI 自己用權重名稱判斷模型
python3 - <<'PY'
p = "/content/ComfyUI/custom_nodes/ComfyUI-GGUF/loader.py"
s = open(p).read()
s = s.replace('"wan", "lumina2", "qwen_image"}', '"wan", "lumina2", "qwen_image", "qwen_image21", "minimax_h3"}')
s = s.replace('raise ValueError(f"This model is not currently supported - ({e})")', 'arch_str = "minimax_h3"')
open(p, "w").write(s)
PY
python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
pip install -q -r $C/requirements.txt -r $C/custom_nodes/ComfyUI-GGUF/requirements.txt websocket-client 2>&1 | grep -v -i "warning\|notice" | tail -5 || true
python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda)"

ln -sf $M/image/qwen-image-2.1/qwen-image-2.1-UC-Q8_0.gguf $C/models/diffusion_models/
ln -sf $M/video/minimax-h3/minimax_h3_fl2va_pruned-Q4_K.gguf $C/models/diffusion_models/
ln -sf $M/image/qwen-image-2.1/Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf $C/models/text_encoders/
ln -sf $M/video/minimax-h3/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S_vis.gguf $C/models/text_encoders/
ln -sf $M/image/qwen-image-2.1/vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors $C/models/vae/
ln -sf $M/video/minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors $C/models/vae/
ln -sf $M/video/minimax-h3/vae/minimax_h3_audio_vae_fp32.safetensors $C/models/vae/

fuser -k 8190/tcp || true  # 程序命令列是 python3 main.py，用 port 找才準
cd $C && nohup python3 main.py --listen 127.0.0.1 --port 8190 --disable-auto-launch > /content/comfy.log 2>&1 &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8190/system_stats > /dev/null && { echo COMFY_UP; exit 0; }; sleep 2; done
tail -30 /content/comfy.log; exit 1
