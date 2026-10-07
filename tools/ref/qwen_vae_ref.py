"""以 diffusers 官方 VAE 產生固定的 Qwen-Image-2.1 解碼／編碼比對資料。

用法：python tools/ref/qwen_vae_ref.py [--root Miroxen目錄] [--model vae目錄] [--output 輸出檔]
latent 已反正規化；編碼只取平均值，不取樣。使用 BF16 官方運算，不修改 RMS_norm。
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from diffusers import AutoencoderKLQwenImage21
from safetensors.torch import save_file


def synthetic_rgba():
    h, w = 256, 320
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    rgba = torch.stack((x * 255 // (w - 1), y * 255 // (h - 1),
                        (x + y) * 255 // (w + h - 2), 96 + x * 159 // (w - 1)), dim=-1)
    rgba[(x - 95).square() + (y - 95).square() < 48**2] = torch.tensor([240, 48, 96, 192])
    rgba[(x >= 185) & (x < 280) & (y >= 140) & (y < 225)] = torch.tensor([32, 208, 64, 255])
    return rgba.to(torch.uint8).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\Workspace\Personal_Project\Miroxen"))
    parser.add_argument("--model", type=Path, help="diffusers 格式的 vae 目錄")
    parser.add_argument("--output", type=Path, help="輸出的 safetensors 檔")
    args = parser.parse_args()
    model_path = args.model or args.root / "models/image/qwen-image-2.1/hf/vae"
    output = args.output or args.root / "models/ref/qwen_vae_ref.safetensors"
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    expected = {"in_channels": 4, "out_channels": 4, "z_dim": 64, "scale_factor_spatial": 16}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"VAE config 不符合 Qwen-Image-2.1：{expected}")
    if len(config["latents_mean"]) != 64 or len(config["latents_std"]) != 64:
        raise ValueError("VAE config 的 latent mean/std 必須各有 64 個值")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能產生 BF16 參考資料")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loading = AutoencoderKLQwenImage21.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, local_files_only=True,
        use_safetensors=True, output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"VAE 權重載入失敗：{key}={loading[key]}")
    model = model.eval().to("cuda:0")
    model.disable_tiling()
    model.disable_slicing()
    generator = torch.Generator(device="cpu").manual_seed(2026)
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(config["latents_std"], dtype=torch.float32)[:, None, None]
    decode_z = (torch.randn(64, 32, 40, generator=generator) * std + mean).contiguous()
    encode_rgba = synthetic_rgba()
    x = encode_rgba.permute(2, 0, 1).float().div(255).mul(2).sub(1)[None, :, None]
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = model.decode(decode_z[None, :, None].to("cuda:0", torch.bfloat16)).sample
        # float32 做像素換算，floor(v+0.5) 與 C++ 四捨五入一致。
        pixels = decoded[0, :, 0].float().add(1).mul(0.5).clamp(0, 1).mul(255)
        decode_rgba = pixels.add(0.5).floor().to(torch.uint8).permute(1, 2, 0).cpu().contiguous()
        encode_mean = model.encode(x.to("cuda:0", torch.bfloat16)).latent_dist.mean
        encode_mean = encode_mean[0, :, 0].float().cpu().contiguous()
    torch.cuda.synchronize(0)
    tensors = {"decode_z": decode_z, "decode_rgba": decode_rgba,
               "encode_rgba": encode_rgba, "encode_mean": encode_mean}
    shapes = {"decode_z": (64, 32, 40), "decode_rgba": (512, 640, 4),
              "encode_rgba": (256, 320, 4), "encode_mean": (64, 16, 20)}
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != shapes[name] or not torch.isfinite(tensor).all():
            raise ValueError(f"參考資料無效：{name}, shape={tuple(tensor.shape)}")
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"min={tensor.min().item():.6g}, max={tensor.max().item():.6g}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={
        "seed": "2026", "reference": "diffusers.AutoencoderKLQwenImage21",
        "dtype": "BF16", "latent": "denormalized", "rounding": "floor(x+0.5)",
    })
    print(f"GPU0: {torch.cuda.get_device_name(0)}; elapsed={time.perf_counter()-started:.3f}s; "
          f"peak={torch.cuda.max_memory_allocated(0)/1024**3:.3f}GiB", flush=True)
    print(f"已儲存：{output}", flush=True)


if __name__ == "__main__":
    main()
