"""Miroxen 端：1280×704 圖片標準 40 步、影片 124 幀 30 步（29 次前向），fp8 與 bf16 各測。每項暖機 1 次＋量 2 次。"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # deploy/bench/，bench.py 在這裡
from bench import Api, PROMPT_IMG, PROMPT_VID, summarize

api = Api("http://127.0.0.1:8188", open(sys.argv[1]).read().strip())
CASES = [
    ("image", {"type": "image", "model": "qwen-image-2.1", "size": "1280x704", "steps": 40}, PROMPT_IMG),
    ("video", {"type": "video", "model": "minimax-h3", "size": "1280x704", "num_frames": 124, "steps": 30}, PROMPT_VID),
]
rows = []
for compute in ("fp8", "bf16"):
    for kind, body, prompt in CASES:
        name = f"{kind} {compute}"
        t = time.time()
        job, _ = api.run(dict(body, compute=compute, prompt=prompt, seed=1))
        print(f"[暖機] {name} {time.time() - t:.1f}s params={json.dumps(job['parameters'], ensure_ascii=False)[:400]}", flush=True)
        runs = []
        for i in range(2):
            job, wall = api.run(dict(body, compute=compute, prompt=f"{prompt} {i + 2}", seed=42 + i))
            r = summarize(name, job, wall)
            runs.append(r)
            print(f"  {name}: 生成 {r['seconds']}s 去噪 {r['denoise']}s timings={r['timings']}", flush=True)
        rows.append({"case": name, "runs": runs, "median_seconds": statistics.median(r["seconds"] for r in runs)})
api.call("POST", "/v1/models/unload", {})  # 讓出顯存給 ComfyUI
json.dump(rows, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("MX_DONE")
