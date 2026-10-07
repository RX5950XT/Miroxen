"""產生 MiniMax-H3 影片 VAE 多幀編碼（參考影片）的官方 diffusers 參考資料。

22 幀（17n+5，n=1）會分成兩段 17 幀各自因果編碼（第二段補最後一幀），串接後丟掉尾端 3 幀 → 7 個 latent 幀；
256×448 會切成兩塊空間 tile 拼接。取 posterior mean 並正規化。
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from diffusers import AutoencoderKLMiniMaxH3
from safetensors.torch import save_file

MODEL = "D:/Workspace/Personal_Project/Miroxen/models/video/minimax-h3/hf/vae"
OUT = "D:/Workspace/Personal_Project/Miroxen/models/ref/h3_video_encode_ref.safetensors"
F, H, W = 22, 256, 448


@torch.inference_mode()
def main():
    model = AutoencoderKLMiniMaxH3.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.eval().to("cuda:0")
    t, y, x = torch.meshgrid(torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij")
    rgb = torch.stack((((x + 9 * t) * 255 // (W - 1)) % 256, (y * 255 // (H - 1) + 7 * t) % 256,
                       ((x + y) * 255 // (W + H - 2) + 13 * t) % 256), dim=-1).to(torch.uint8)
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device="cuda:0").view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device="cuda:0").view(1, 3, 1, 1, 1)
    pixels = rgb.to("cuda:0", torch.float32).permute(3, 0, 1, 2)[None].div(255)
    pixels = (pixels - pixel_mean) / pixel_std
    mean = torch.tensor(model.config.latents_mean, device="cuda:0").view(1, 24, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, device="cuda:0").view(1, 24, 1, 1, 1)
    posterior = model.encode(pixels, return_dict=False)[0].mean.float()
    encoded = ((posterior - mean) / std)[0].cpu().contiguous()
    print("encoded", tuple(encoded.shape))
    assert list(encoded.shape) == [24, 7, H // 16, W // 16]
    save_file({"rgb": rgb.contiguous(), "encoded": encoded}, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
