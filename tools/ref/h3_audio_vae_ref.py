"""產生 MiniMax-H3 音訊 VAE 的 fp32 diffusers 參考；不下載或改寫模型。

用法：python tools/ref/h3_audio_vae_ref.py [audio_vae 目錄] [輸出 safetensors]
CUDA_DEVICE_ORDER 在匯入 torch 前設為 PCI_BUS_ID。
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio
from safetensors.torch import save_file


def model_root():
    if os.environ.get("MX_MODELS_DIR"):
        return Path(os.environ["MX_MODELS_DIR"])
    repo = Path(__file__).resolve().parents[2]
    for root in (repo / "models", repo.parent.parent / "Miroxen" / "models"):
        if (root / "video/minimax-h3/hf/audio_vae/config.json").is_file():
            return root
    return repo / "models"


def main():
    root = model_root()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "video/minimax-h3/hf/audio_vae"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "ref/h3_audio_vae_ref.safetensors"
    if not (src / "config.json").is_file() or not (src / "diffusion_pytorch_model.safetensors").is_file():
        print(f"SKIP：缺少音訊模型 {src}")
        return
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print("GPU0", torch.cuda.get_device_name(0), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(src, torch_dtype=torch.float32).to("cuda:0").eval()
    gen = torch.Generator(device="cpu").manual_seed(20261004)
    normalized = torch.randn((2, 32, 80), generator=gen, dtype=torch.float32)
    mean = torch.tensor(cfg["latents_mean"], device="cuda:0").view(1, 32, 1)
    std = torch.tensor(cfg["latents_std"], device="cuda:0").view(1, 32, 1)
    time = torch.arange(64000, dtype=torch.float32) / cfg["sampling_rate"]
    noise = torch.randn((2, 64000), generator=gen) * 0.01
    left = 0.2 * torch.sin(2 * torch.pi * 440 * time) + 0.1 * torch.sin(2 * torch.pi * 997 * time)
    right = 0.15 * torch.sin(2 * torch.pi * 330 * time + 0.3) + 0.08 * torch.sin(2 * torch.pi * 1200 * time)
    mono = torch.stack((left, right)) + noise
    data = {"normalized": normalized, "pcm": mono.t().contiguous()}
    with torch.inference_mode():
        data["decoded"] = vae.decode(normalized.cuda() * std + mean).sample.cpu().contiguous()
        data["encoded"] = ((vae.encode(mono[:, None].cuda()).latent_dist.mode() - mean) / std).cpu().contiguous()
        # 驗證補零及 ceil，801 samples 必須產生兩個 frames。
        data["short_pcm"] = mono[:, :801].t().contiguous()
        data["short_encoded"] = ((vae.encode(mono[:, None, :801].cuda()).latent_dist.mode() - mean) / std).cpu().contiguous()
    for name, tensor in data.items():
        assert torch.isfinite(tensor).all(), name
        print(name, tuple(tensor.shape), "rms", tensor.square().mean().sqrt().item(), flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(data, dst, metadata={"seed": "20261004", "dtype": "float32", "source": str(src),
                                  "torch": torch.__version__, "config": json.dumps(cfg)})
    print("saved", dst, flush=True)


if __name__ == "__main__":
    main()
