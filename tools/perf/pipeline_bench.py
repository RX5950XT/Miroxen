"""真實 API 分段量測；每組先換提示詞，再原樣重送比較快取收益。"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def request(base, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def generate(base, body):
    start = time.perf_counter()
    job = request(base, "/v1/jobs", body)
    deadline = start + 1800
    while job["status"] not in ("succeeded", "failed", "cancelled"):
        if time.perf_counter() > deadline:
            request(base, "/v1/jobs/" + job["id"], method="DELETE")
            raise TimeoutError(job["id"])
        time.sleep(0.25)
        job = request(base, "/v1/jobs/" + job["id"])
    if job["status"] != "succeeded":
        raise RuntimeError(job)
    job["wall_seconds"] = time.perf_counter() - start
    phases = job["result"].get("timings", {})
    if phases:
        assert all(v >= 0 for v in phases.values()), phases
        assert abs(sum(phases.values()) - job["result"]["seconds"]) < 0.1, job
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8189")
    parser.add_argument("--kind", choices=["image", "video", "both"], default="both")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", default="out/perf-20261007/benchmark.json")
    args = parser.parse_args()
    assert 1 <= args.pairs <= 20
    records = []
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    loras = request(args.base, "/v1/loras")["data"]
    for kind in ("image", "video"):
        if args.kind not in (kind, "both"):
            continue
        name = "turbo8_lora_step2500.safetensors" if kind == "image" else "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
        lora = next(row["name"] for row in loras if row["name"].endswith(name))
        model = "qwen-image-2.1" if kind == "image" else "minimax-h3"
        body = dict(type=kind, model=model, prompt="Warmup: a red panda in a green forest.", seed=123, compute="fp8", loras=[dict(name=lora, scale=1)])
        if kind == "image":
            body.update(size="736x1312", steps=8, cfg_scale=1, shift_terminal=False)
        else:
            body.update(size="864x480", num_frames=56, steps=5, video_shift=6)
        warm = generate(args.base, body)
        print(kind, "warmup", round(warm["result"]["seconds"], 3), flush=True)
        for pair in range(args.pairs):
            body["prompt"] = f"A red panda sitting on a mossy log in a green forest, natural light, detailed fur. Scene {pair + 1}."
            for repeat in range(2):
                job = generate(args.base, body)
                record = dict(kind=kind, pair=pair, repeat=repeat, request=body.copy(), job=job)
                records.append(record)
                output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                print(kind, pair, "repeat" if repeat else "new", json.dumps(job["result"]), flush=True)


if __name__ == "__main__":
    main()
