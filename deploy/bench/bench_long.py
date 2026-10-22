"""長影片測速：1280×704 的 5 秒／10 秒，各跑 4 步、8 步 LoRA 與完整 30 步。只暖機一次，每項跑一次。

  python3 bench_long.py --key-file ~/.config/miroxen-colab/api_key [--url http://127.0.0.1:8188] [--out bench-long.json]
"""
import argparse, json, time
from bench import Api, CASES, PROMPT_VID, summarize

SIZE = "1280x704"  # 720 不是 32 的倍數
LENGTHS = {"5 秒": 124, "10 秒": 243}  # 24 fps，幀數對齊 17n+5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", default="bench-long.json")
    a = ap.parse_args()
    api = Api(a.url, open(a.key_file).read().strip())
    modes = {name.split(" 864")[0]: body for name, body in CASES["video"]}  # 完整 30 步／8 步 LoRA／4 步 LoRA

    t = time.time()
    api.run(dict(modes["影片 4 步 LoRA"], size=SIZE, num_frames=124, prompt=PROMPT_VID, seed=1))
    print(f"[暖機] {time.time() - t:.1f}s", flush=True)

    rows = []
    for sec, frames in LENGTHS.items():
        for mode in ("影片 4 步 LoRA", "影片 8 步 LoRA", "影片 完整 30 步"):
            name = f"{mode} {SIZE} {sec}（{frames} 幀）"
            try:
                job, wall = api.run(dict(modes[mode], size=SIZE, num_frames=frames, prompt=PROMPT_VID, seed=42))
            except Exception as e:  # 顯存不足等錯誤記下來，繼續測下一項
                print(f"  {name}: 失敗 {e}", flush=True)
                rows.append({"case": name, "error": str(e)})
                continue
            r = summarize(name, job, wall)
            rows.append(r)
            print(f"  {name}: 生成 {r['seconds']}s（去噪 {r['denoise']}s，{r['sec_per_step']}s/步），含網路 {r['wall']}s", flush=True)

    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\n| 項目 | 生成秒數 | 去噪秒數 | 秒/步 |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r.get('seconds', '失敗')} | {r.get('denoise', '')} | {r.get('sec_per_step', '')} |")


if __name__ == "__main__":
    main()
