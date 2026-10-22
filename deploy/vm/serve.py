# 在 Colab kernel 裡長時間執行的 cell：讓 kernel 保持忙碌（Colab 依 kernel 活動判斷是否回收 VM），
# 等 setup.sh 寫出 /content/.mx_ready 後啟動 miroxen serve，掛掉就重啟。
# 由 VM 上的 keepalive 迴圈用 colab CLI 送進自己的 kernel；連線斷掉重送時，接手已在跑的伺服器。
import os, subprocess, time, urllib.request

APP, READY, KEYFILE = "/content/miroxen", "/content/.mx_ready", "/content/mx/api_key"
COMPUTE = os.environ.get("MX_COMPUTE", "fp8")


def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8188/health", timeout=5).status == 200
    except Exception:
        return False


def start():
    key = open(KEYFILE).read().strip()
    cmd = [f"{APP}/miroxen", "serve", "--host", "127.0.0.1", "--port", "8188", "--api-key", key,
           "--out", "/content/out", "--models", f"{APP}/models", "--compute", COMPUTE]
    log = open("/content/server.log", "ab")
    # /usr/lib64-nvidia 是 Colab 放 libcuda 的位置
    libs = [f"{APP}/lib", "/usr/lib64-nvidia", os.environ.get("LD_LIBRARY_PATH", "")]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(p for p in libs if p))
    proc = subprocess.Popen(cmd, cwd=APP, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    print(f"miroxen serve 已啟動（pid {proc.pid}）", flush=True)
    return proc


print("等待 setup.sh 完成…", flush=True)
while not os.path.exists(READY):
    time.sleep(2)

proc = None if healthy() else start()
if proc is None:
    print("接手已在執行的伺服器", flush=True)
restarts, misses, last_beat = 0, 0, 0.0
while True:
    time.sleep(10)
    if proc is not None and proc.poll() is not None:
        print(f"伺服器結束（code {proc.returncode}）", flush=True)
        proc, misses = None, 3
    misses = 0 if healthy() else misses + 1
    if proc is None and misses >= 3:  # 30 秒沒回應又不是我們啟動中的程序 → 重啟
        restarts += 1
        if restarts > 20:
            raise SystemExit("重啟太多次，停止")
        proc, misses = start(), 0
    if time.time() - last_beat > 600:
        last_beat = time.time()
        print(time.strftime("%H:%M:%S"), "alive", "healthy" if healthy() else "starting", flush=True)
