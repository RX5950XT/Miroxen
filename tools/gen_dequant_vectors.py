"""產生獨立參考向量：uv run --with gguf --with numpy python tools/gen_dequant_vectors.py。"""
from pathlib import Path
import struct
import numpy as np
from gguf import GGMLQuantizationType as QT, GGML_QUANT_SIZES
from gguf.quants import dequantize


def fp8_reference(raw, e4):
    # gguf-py 沒有 FP8；用完整 256 值查表，包含 subnormal、Inf 與 NaN。
    values = []
    for byte in raw:
        v = int(byte)
        bits, bias = (3, 7) if e4 else (2, 15)
        exponent, mantissa = (v & 127) >> bits, v & ((1 << bits) - 1)
        if e4 and exponent == 15 and mantissa == 7:
            f = float("nan")
        elif not e4 and exponent == 31:
            f = float("nan") if mantissa else float("inf")
        elif exponent == 0:
            f = mantissa * 2.0 ** (1 - bias - bits)
        else:
            f = (1 + mantissa / 2.0 ** bits) * 2.0 ** (exponent - bias)
        values.append(-f if v & 128 else f)
    return np.asarray(values, dtype="<f4")


def main():
    rng = np.random.default_rng(20261004)
    names = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
             "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS"]
    scales = {"Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2],
              "Q8_0": [0], "Q2_K": [80, 82], "Q3_K": [108], "Q4_K": [0, 2],
              "Q5_K": [0, 2], "Q6_K": [208], "IQ4_NL": [0], "IQ4_XS": [0]}
    records = []
    for name in names:
        qt = QT[name]
        block, size = GGML_QUANT_SIZES[qt]
        raw = rng.integers(0, 256, (19, size), dtype=np.uint8)
        if name in scales:
            for b in range(len(raw)):
                for offset in scales[name]:
                    # 含正負、零、最小 subnormal 與最大有限 half。
                    h = [0, 0x8000, 1, 0x8001, 0x7bff, 0xfbff][b] if b < 6 else int(rng.integers(0, 0x7c00)) | (b % 2 << 15)
                    raw[b, offset:offset + 2] = np.frombuffer(struct.pack("<H", h), dtype=np.uint8)
        elif name == "F32":
            raw = rng.standard_normal(257).astype("<f4").view(np.uint8)
        elif name == "F16":
            raw = np.arange(65536, dtype="<u2").view(np.uint8)
        elif name == "BF16":
            raw = np.arange(65536, dtype="<u2").view(np.uint8)
        expected = dequantize(raw, qt).astype("<f4").reshape(-1)
        records.append((int(qt), raw.tobytes(), expected))
    for qt, e4 in [(1000, True), (1001, False)]:
        raw = np.arange(256, dtype=np.uint8)
        records.append((qt, raw.tobytes(), fp8_reference(raw, e4)))
    path = Path(__file__).resolve().parents[1] / "tests/data/dequant_vectors.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"MXDQ0001" + struct.pack("<I", len(records)))
        for qt, raw, expected in records:
            f.write(struct.pack("<III", qt, len(expected), len(raw)))
            f.write(raw)
            f.write(expected.tobytes())
    print(f"{path.name}: {len(records)} formats, {path.stat().st_size} bytes; gguf-py 15 + FP8 2")


if __name__ == "__main__":
    main()
