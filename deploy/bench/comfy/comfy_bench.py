"""ComfyUI 測速（在 VM 上跑，連 127.0.0.1:8190）。和 Miroxen 同樣的模型檔、尺寸、步數、cfg=1、euler。
每項先暖機一次，再跑 N 次；每次換 seed 並在 prompt 尾端加編號，避免 ComfyUI 快取跳過文字編碼。
用 websocket 的 executing 事件記每個節點的耗時（文字編碼／取樣／VAE 解碼）。

  python3 comfy_bench.py [--only image|video] [--repeat 2] [--te gguf|<檔名>] [--out comfy.json]
"""
import argparse, json, statistics, time, urllib.request, uuid
import websocket

URL = "127.0.0.1:8190"
PROMPT_IMG = "雨夜的台北巷口，霓虹招牌倒映在積水上，一隻橘貓坐在機車座墊上，電影感光影，細節豐富"
PROMPT_VID = "一隻橘貓在雨夜的台北巷口慢慢走過積水，霓虹燈倒影晃動，鏡頭緩慢跟拍"
W, H = 1280, 704  # 720 不是 32 的倍數（Miroxen 與 H3 都要求 32 倍數）


def clip_node(name, kind):
    if name.endswith(".gguf"):
        return {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": name, "type": kind}}
    return {"class_type": "CLIPLoader", "inputs": {"clip_name": name, "type": kind, "device": "default"}}


def image_graph(prompt, seed, te):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-2.1-UC-Q8_0.gguf"}},
        "clip": clip_node(te or "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf", "qwen_image"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"}},
        "te": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["te", 0]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "seed": seed, "steps": 40, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["te", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "cmp_img"}},
    }


def video_graph(prompt, seed, te):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "minimax_h3_fl2va_pruned-Q4_K.gguf"}},
        "clip": clip_node(te or "MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S_vis.gguf", "minimax"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "te": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": W, "height": H, "length": 124}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["te", 0]}},
        # Miroxen 的 30 步＝30 個取樣點＝29 次前向；euler 的 steps 就是前向次數
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "seed": seed, "steps": 29, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["te", 0], "negative": ["neg", 0], "latent_image": ["te", 1], "denoise": 1.0}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["split", 0], "vae": ["vae", 0]}},
        "adecode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["split", 1], "vae": ["avae", 0]}},
        "mux": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0, "audio": ["adecode", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mux", 0], "filename_prefix": "cmp_vid", "format": "mp4", "format.codec": "h264"}},
    }


def run(graph):
    cid = uuid.uuid4().hex
    ws = websocket.create_connection(f"ws://{URL}/ws?clientId={cid}", timeout=7200)
    body = json.dumps({"prompt": graph, "client_id": cid}).encode()
    try:
        pid = json.load(urllib.request.urlopen(urllib.request.Request(f"http://{URL}/prompt", data=body)))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()[:2000])
    t0 = time.time(); node = None; t_node = t0; nodes = {}
    while True:
        msg = ws.recv()
        if not isinstance(msg, str):
            continue
        m = json.loads(msg); d = m.get("data", {})
        if d.get("prompt_id") not in (None, pid):
            continue
        if m["type"] == "executing":
            now = time.time()
            if node is not None:
                nodes[node] = round(nodes.get(node, 0) + now - t_node, 2)
            node, t_node = d.get("node"), now
            if node is None:
                break
        elif m["type"] == "execution_error":
            raise RuntimeError(json.dumps(d, ensure_ascii=False)[:3000])
    ws.close()
    return round(time.time() - t0, 2), nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["image", "video"])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--te", default="", help="改用 ComfyUI 官方文字編碼器檔名（GGUF 編碼器載不進時）")
    ap.add_argument("--out", default="/content/comfy_bench.json")
    a = ap.parse_args()
    cases = [("image", image_graph, PROMPT_IMG), ("video", video_graph, PROMPT_VID)]
    rows = []
    for kind, build, prompt in cases:
        if a.only and kind != a.only:
            continue
        t, nodes = run(build(prompt, 1, a.te))
        print(f"[暖機] {kind} {t}s {nodes}", flush=True)
        runs = []
        for i in range(a.repeat):
            t, nodes = run(build(f"{prompt} {i + 2}", 42 + i, a.te))
            runs.append({"total": t, "nodes": nodes})
            print(f"  {kind}: 總計 {t}s，各節點 {nodes}", flush=True)
        rows.append({"case": kind, "runs": runs, "median_total": statistics.median(r["total"] for r in runs)})
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
