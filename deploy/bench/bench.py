"""Miroxen 測速：完整步數 vs 加速步數，以及併發請求。只用標準函式庫。

  python3 bench.py --key-file ~/.config/miroxen-colab/api_key [--url http://127.0.0.1:8188]
                   [--only image|video] [--concurrency 4] [--out bench.json]
"""
import argparse, concurrent.futures as cf, json, statistics, sys, time, urllib.request

PROMPT_IMG = "雨夜的台北巷口，霓虹招牌倒映在積水上，一隻橘貓坐在機車座墊上，電影感光影，細節豐富"
PROMPT_VID = "一隻橘貓在雨夜的台北巷口慢慢走過積水，霓虹燈倒影晃動，鏡頭緩慢跟拍"
# 名稱、種類、請求本文
CASES = {
    "image": [
        ("圖片 標準 40 步 1024²", {"type": "image", "model": "qwen-image-2.1", "size": "1024x1024", "steps": 40}),
        ("圖片 Turbo 8 步 1024²", {"type": "image", "model": "qwen-image-2.1-turbo", "size": "1024x1024", "steps": 8}),
    ],
    "video": [
        ("影片 完整 30 步 864×480 56 幀", {"type": "video", "model": "minimax-h3", "size": "864x480", "num_frames": 56, "steps": 30}),
        ("影片 8 步 LoRA 864×480 56 幀", {"type": "video", "model": "minimax-h3", "size": "864x480", "num_frames": 56, "steps": 9,
                                       "video_shift": 6, "loras": [{"name": "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"}]}),
        ("影片 4 步 LoRA 864×480 56 幀", {"type": "video", "model": "minimax-h3", "size": "864x480", "num_frames": 56, "steps": 5,
                                       "video_shift": 6, "loras": [{"name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"}]}),
    ],
}


class Api:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key

    def call(self, method, path, body=None, timeout=60):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def run(self, body, timeout=3600):
        """送非同步工作並輪詢到結束；回傳 (工作物件, 從送出到完成的秒數)。"""
        t0 = time.time()
        job = self.call("POST", "/v1/jobs", body)
        while job["status"] in ("queued", "running"):
            if time.time() - t0 > timeout:
                raise TimeoutError(job["id"])
            time.sleep(0.5)
            job = self.call("GET", f"/v1/jobs/{job['id']}")
        if job["status"] != "succeeded":
            raise RuntimeError(f"{job['id']} {job['status']}: {job.get('error')}")
        return job, time.time() - t0


def summarize(name, job, wall):
    r = job["result"]
    t = r.get("timings") or {}
    steps = job["parameters"]["steps"]
    fwd = steps - 1 if job["type"] == "video" else steps  # 影片前向次數＝steps−1
    return {"case": name, "seed": r.get("seed"), "seconds": round(r["seconds"], 2), "wall": round(wall, 2),
            "denoise": round(t.get("denoise", 0), 2), "steps": fwd,
            "sec_per_step": round(t.get("denoise", 0) / max(fwd, 1), 3), "url": r.get("url"), "timings": t}


def bench_cases(api, kinds, repeat):
    rows = []
    for kind in kinds:
        for name, body in CASES[kind]:
            body = dict(body, prompt=PROMPT_IMG if kind == "image" else PROMPT_VID, seed=42)
            t = time.time()
            api.run(body)  # 暖機：首次載入模型與編譯快取不計
            print(f"[暖機] {name} {time.time() - t:.1f}s", flush=True)
            runs = []
            for _ in range(repeat):
                job, wall = api.run(body)
                runs.append(summarize(name, job, wall))
                print(f"  {name}: 生成 {runs[-1]['seconds']}s（去噪 {runs[-1]['denoise']}s，"
                      f"{runs[-1]['sec_per_step']}s/步），含網路 {runs[-1]['wall']}s", flush=True)
            best = min(runs, key=lambda x: x["seconds"])
            best["median_seconds"] = round(statistics.median(x["seconds"] for x in runs), 2)
            rows.append(best)
    return rows


def bench_concurrency(api, n):
    """同時送 n 件 Turbo 圖片，看是否真的平行，以及總吞吐。"""
    body = dict(CASES["image"][1][1], prompt=PROMPT_IMG)
    api.run(dict(body, seed=1))  # 確保 Turbo 已載入
    single = api.run(dict(body, seed=2))[1]
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda i: api.run(dict(body, seed=100 + i)), range(n)))
    total = time.time() - t0
    lat = sorted(w for _, w in res)
    return {"n": n, "single_wall": round(single, 2), "total_wall": round(total, 2),
            "latency_min": round(lat[0], 2), "latency_max": round(lat[-1], 2),
            "throughput_img_per_min": round(n / total * 60, 2),
            "parallel_speedup": round(single * n / total, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--key-file")
    ap.add_argument("--key", default="")
    ap.add_argument("--only", choices=["image", "video"])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    key = a.key or (open(a.key_file).read().strip() if a.key_file else "")
    api = Api(a.url, key)
    system = api.call("GET", "/v1/system")
    print("GPU:", ", ".join(f"{g['name']} {g['total'] / 2**30:.0f} GiB" for g in system["gpus"]), flush=True)
    kinds = [a.only] if a.only else ["image", "video"]
    report = {"gpus": system["gpus"], "cases": bench_cases(api, kinds, a.repeat)}
    if a.concurrency > 1 and "image" in kinds:
        report["concurrency"] = bench_concurrency(api, a.concurrency)
        print("併發：", report["concurrency"], flush=True)
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n| 項目 | 生成秒數 | 去噪秒數 | 步數 | 秒/步 |\n|---|---|---|---|---|")
    for r in report["cases"]:
        print(f"| {r['case']} | {r['seconds']} | {r['denoise']} | {r['steps']} | {r['sec_per_step']} |")
    print(f"\n結果存到 {a.out}")


if __name__ == "__main__":
    sys.exit(main())
