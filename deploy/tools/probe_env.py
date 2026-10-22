import subprocess, time

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout + r.stderr).strip()}\n")

sh("cat /etc/os-release | head -2; uname -r; nproc; free -g | head -2; df -h /content / | tail -2")
sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi | head -4")
sh("ls -d /usr/local/cuda*; nvcc --version | tail -2; cmake --version | head -1; ninja --version; gcc --version | head -1")
sh("ffmpeg -version | head -1; which cloudflared ffprobe; python3 --version; ldd --version | head -1")
t = time.time()
sh("curl -sL -o /tmp/t.bin -w '%{size_download} bytes %{speed_download} B/s\\n' "
   "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Q8_0.gguf -r 0-2000000000")
print("hf 2GB range seconds", round(time.time() - t, 1))
sh("rm -f /tmp/t.bin")
