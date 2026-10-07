"""產生 MiniMax-H3 video VAE 的官方 diffusers 參考資料。

固定 latent 會經過兩段時間解碼與兩塊空間拼接。單張圖使用 posterior mean，
直接正規化；與官方 encode_vae_condition 的 seed 42 抽樣及 float16 捨入不同。
"""
import argparse
import os
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from diffusers import AutoencoderKLMiniMaxH3
from safetensors.torch import save_file


def synthetic_rgb(height, width):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack((x * 255 // (width - 1), y * 255 // (height - 1),
                        (x + y) * 255 // (width + height - 2)), dim=-1).to(torch.uint8)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(
        "D:/Workspace/Personal_Project/Miroxen/models/video/minimax-h3/hf/vae"))
    parser.add_argument("--output", type=Path, default=Path(
        "D:/Workspace/Personal_Project/Miroxen/models/ref/h3_video_vae_ref.safetensors"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("參考資料需要 CUDA GPU")
    torch.cuda.set_device(0)
    print("GPU0", torch.cuda.get_device_name(0), flush=True)
    model = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True).eval().to("cuda:0")
    print("loaded", args.model, "tiling", model.use_tiling, flush=True)
    if not model.use_tiling or model.config.latent_channels != 24:
        raise RuntimeError("需要官方 MiniMax-H3 24 channel、預設 tiling 設定")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn((1, 24, 12, 16, 28), generator=generator, dtype=torch.float32)
    mean = torch.tensor(model.config.latents_mean, device="cuda:0").view(1, 24, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, device="cuda:0").view(1, 24, 1, 1, 1)
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device="cuda:0").view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device="cuda:0").view(1, 3, 1, 1, 1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    video = model.decode(z.to("cuda:0") * std + mean, return_dict=False)[0]
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    # 與 diffusers VideoProcessor/PIL 相同：乘 255，四捨五入後轉 uint8。
    frames = video[0].permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().contiguous()
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) * 1000
    if list(frames.shape) != [39, 256, 448, 3]:
        raise RuntimeError(f"解碼 shape 不符：{list(frames.shape)}")
    del video

    rgb = synthetic_rgb(256, 448)
    pixels = rgb.to("cuda:0", torch.float32).permute(2, 0, 1)[None, :, None].div(255)
    pixels = (pixels - pixel_mean) / pixel_std
    torch.cuda.synchronize()
    start = time.perf_counter()
    posterior_mean = model.encode(pixels, return_dict=False)[0].mean.float()
    encoded = ((posterior_mean - mean) / std)[0].cpu().contiguous()
    torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - start) * 1000
    if list(encoded.shape) != [24, 1, 16, 28] or not torch.isfinite(encoded).all():
        raise RuntimeError("編碼 latent shape 或數值不合法")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"normalized_latent": z[0].contiguous(), "decoded_rgb": frames,
               "encoder_rgb": rgb.contiguous(), "encoded_latent": encoded}, str(args.output),
              metadata={"seed": str(args.seed), "model": str(args.model.resolve()),
                        "dtype": "mixed bf16/f32 (official keep_in_fp32_modules)",
                        "encoder_semantics": "posterior mean, no sampling, no fp16 rounding",
                        "decode_ms": str(decode_ms), "encode_ms": str(encode_ms)})
    print(f"decode {decode_ms:.2f} ms; encode {encode_ms:.2f} ms; frames {tuple(frames.shape)}", flush=True)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
