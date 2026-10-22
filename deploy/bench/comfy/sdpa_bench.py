# PyTorch 注意力測速，形狀與 Miroxen attention_test --bench 相同（D=128、非因果、bf16）
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
def bench(S, H, backend):
    q, k, v = (torch.randn(1, H, S, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    try:
        with sdpa_kernel([backend]):
            F.scaled_dot_product_attention(q, k, v); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); n = 3
            e0.record()
            for _ in range(n): F.scaled_dot_product_attention(q, k, v)
            e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / n
    except Exception as e:
        return f"不支援（{type(e).__name__}）"
    return f"{ms:.3f} ms {4 * S * S * H * 128 / (ms * 1e9):.1f} TFLOPS"
print(torch.__version__, torch.cuda.get_device_name())
for S, H in [(4096, 32), (16384, 32), (32768, 32), (16384, 56), (130240, 56)]:
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        print(f"SDPA S={S} H={H} {b.name}: {bench(S, H, b)}", flush=True)
