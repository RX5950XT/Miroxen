#!/usr/bin/env bash
# 在 Colab VM 上執行（由 mx-colab up 透過 ssh 呼叫）：
# 1. 從 Drive 拉程式包與模型到本機 SSD；Drive 缺的模型改從 Hugging Face 下載，背景補傳回 Drive。
# 2. 啟動輸出同步（/content/out → Drive）與 cloudflared 公開網址。
# 3. 寫入 /content/.mx_ready，讓 kernel 裡等待中的 serve.py 啟動伺服器。
# 環境變數：MX_REMOTE（預設 gdrive:Miroxen）、MX_GROUPS（預設 "image video"）、MX_PUBLIC（1＝開公開網址）、/content/mx/hf_token（選填）。
set -euo pipefail
REMOTE=${MX_REMOTE:-gdrive:Miroxen}
GROUPS_=${MX_GROUPS:-image video}
APP=/content/miroxen
OUT=/content/out
MANIFEST=/content/mx/models.tsv
# 同一台 VM 重跑 up 沿用同一個輸出資料夾
RUN_ID=$(cat /content/mx/run_id 2>/dev/null || TZ=Asia/Taipei date +%Y%m%d-%H%M%S)
export PATH=/usr/local/bin:$PATH
t0=$(date +%s)
log() { printf '[setup %4ss] %s\n' $(( $(date +%s) - t0 )) "$*"; }

if ! command -v rclone >/dev/null; then
  log "安裝 rclone"
  curl -fsSL -o /tmp/rclone.zip https://downloads.rclone.org/v1.71.1/rclone-v1.71.1-linux-amd64.zip
  python3 -m zipfile -e /tmp/rclone.zip /tmp/ && install -m755 /tmp/rclone-v1.71.1-linux-amd64/rclone /usr/local/bin/rclone
fi
# aria2 用 apt 裝，背景進行，第一次要用前再等；先試現有套件清單，裝不到才 update（update 約 60 秒）
aria_pid=
apt_aria() { DEBIAN_FRONTEND=noninteractive apt-get -qq install -y aria2; }
command -v aria2c >/dev/null || { log "安裝 aria2（背景）"; { apt_aria || { apt-get -qq update && apt_aria; }; } >/dev/null 2>&1 & aria_pid=$!; }
if [[ ${MX_PUBLIC:-1} == 1 ]] && ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
RC="rclone --config /root/.config/rclone/rclone.conf"

# keepalive（見 keepalive.sh）：不依賴本機電腦開著。每次 up 都重啟，避免沿用卡住的舊連線。
if [[ -n ${MX_SESSION:-} ]]; then
  command -v colab >/dev/null || { log "安裝 colab CLI"; pip install -q google-colab-cli >/dev/null 2>&1; }
  [[ -f /content/mx/adc.json ]] && export GOOGLE_APPLICATION_CREDENTIALS=/content/mx/adc.json
  pkill -f keepalive.sh || true; pkill -f "colab.* exec -s $MX_SESSION " || true
  MX_COLAB_AUTH=${MX_COLAB_AUTH:-} MX_COMPUTE=${MX_COMPUTE:-fp8} \
    setsid nohup bash /content/mx/keepalive.sh "$MX_SESSION" >> /content/cell.log 2>&1 &
  log "keepalive 已啟動（log：/content/cell.log、/content/cell.cur）"
fi

# 選定群組的模型清單（dest 與大小）
awk -F'\t' -v g=" $GROUPS_ " '!/^#/ && NF>=6 && index(g, " "$6" ") {print $4"\t"$5"\t"$1"\t"$2"\t"$3}' "$MANIFEST" > /content/mx/want.tsv
cut -f1 /content/mx/want.tsv > /content/mx/want.txt
HF_TOKEN=$(cat /content/mx/hf_token 2>/dev/null || true)
hdr=(); chdr=(); [[ -n ${HF_TOKEN:-} ]] && { hdr=(--header="Authorization: Bearer $HF_TOKEN"); chdr=(-H "Authorization: Bearer $HF_TOKEN"); }
# 測速：最大的檔案兩邊各拉最多 256MB／15 秒，選快的那邊（Drive 同一天下載量大會被限速；Drive 沒有的檔案量到 0）
mbps() { local t0 n; t0=$(date +%s.%N); n=$("$@" 2>/dev/null | head -c 268435456 | wc -c) || true
  awk -v n="$n" -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.0f", n/(b-a)/1048576}'; }
# 程式包在測速後才拉（避免和 Drive 測速搶頻寬），和模型下載同時進行，就緒前再等它
pull_bundle() {
  log "拉程式包（背景）"
  # 保留 models/，重跑 up 不用重下
  { $RC copy "$REMOTE/bin/miroxen-linux.tar.gz" /content/ --drive-chunk-size 128M &&
    rm -rf "$APP"/{miroxen,web,lib} && mkdir -p "$APP" && tar -xzf /content/miroxen-linux.tar.gz -C "$APP"; } > /content/bundle.log 2>&1 &
  bundle_pid=$!
}
read -r dest _ repo rev path < <(sort -t$'\t' -k2 -nr /content/mx/want.tsv | head -1)
drive=$(mbps timeout 15 $RC cat "$REMOTE/models/$dest" --count 268435456)
hf=$(mbps timeout 15 curl -sfL "${chdr[@]}""https://huggingface.co/$repo/resolve/$rev/$path")
if (( hf > drive )); then
  log "測速 Drive ${drive} MB/s、Hugging Face ${hf} MB/s → 全部從 Hugging Face 下載"
  pull_bundle
else
  log "測速 Drive ${drive} MB/s、Hugging Face ${hf} MB/s → 從 Drive 拉模型（$(wc -l < /content/mx/want.txt) 個檔案）"
  pull_bundle
  $RC copy "$REMOTE/models" "$APP/models" --files-from /content/mx/want.txt \
    --transfers 6 --multi-thread-streams 8 --multi-thread-cutoff 64M --buffer-size 64M --stats 30s --stats-one-line -v 2>&1 \
    | grep -E "Transferred:|ERROR" || true
fi

: > /content/mx/missing.txt
while IFS=$'\t' read -r dest size repo rev path; do
  f="$APP/models/$dest"
  [[ -f $f && $(stat -c %s "$f") == "$size" ]] && continue
  log "從 Hugging Face 下載 $dest"
  [[ -n $aria_pid ]] && { wait "$aria_pid" || true; aria_pid=; log "aria2 就緒"; }
  mkdir -p "$(dirname "$f")"
  aria2c -q -x16 -s16 -k64M --file-allocation=none --allow-overwrite=true "${hdr[@]}" \
    -d "$(dirname "$f")" -o "$(basename "$f")" "https://huggingface.co/$repo/resolve/$rev/$path" || { echo "下載失敗：$repo/$path（檢查 models.tsv）"; exit 1; }
  [[ $(stat -c %s "$f") == "$size" ]] || { echo "大小不符：$dest"; exit 1; }
  echo "$dest" >> /content/mx/missing.txt
done < /content/mx/want.tsv
# 用整份 want.txt 補傳（Drive 已有的 rclone 會跳過），之前中斷或上次 VM 下載過但沒傳完的也會補上
if [[ ${MX_SEED:-1} == 1 ]] && { [[ -s /content/mx/missing.txt ]] || ! $RC check "$APP/models" "$REMOTE/models" --files-from /content/mx/want.txt --size-only --one-way -q 2>/dev/null; }; then
  log "背景補傳模型到 Drive（log：/content/seed.log）"
  # --size-only：從 HF 下載的檔案修改時間是新的，不加會把 Drive 已有的全部重傳
  setsid nohup $RC copy "$APP/models" "$REMOTE/models" --files-from /content/mx/want.txt --size-only \
    --transfers 4 --drive-chunk-size 256M --stats 60s --stats-one-line -v > /content/seed.log 2>&1 &
fi

wait "$bundle_pid" || { tail -5 /content/bundle.log; echo "拉程式包失敗"; exit 1; }
log "程式包就緒"
LD_LIBRARY_PATH="$APP/lib" ldd "$APP/miroxen" | grep "not found" && { echo "缺少動態函式庫"; exit 1; }

mkdir -p "$OUT"
log "輸出同步：$OUT → $REMOTE/outputs/$RUN_ID"
echo "$RUN_ID" > /content/mx/run_id
# --min-age 避開還在寫的檔案；tmp/ 是上傳暫存。
pkill -f "copy $OUT $REMOTE/outputs" || true
setsid nohup bash -c "while :; do $RC copy $OUT $REMOTE/outputs/$RUN_ID --exclude 'tmp/**' --min-age 3s -q; sleep 5; done" > /content/sync.log 2>&1 &

if [[ ${MX_PUBLIC:-1} == 1 ]]; then
  pkill -f "cloudflared tunnel" || true
  setsid nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8188 > /content/cloudflared.log 2>&1 &
  for _ in $(seq 60); do
    url=$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' /content/cloudflared.log | head -1 || true)
    [[ -n $url ]] && break; sleep 1
  done
  echo "${url:-}" > /content/endpoint.txt
  $RC copyto /content/endpoint.txt "$REMOTE/endpoint.txt" -q || true
  log "公開網址：${url:-（取得失敗，見 /content/cloudflared.log）}"
fi

touch /content/.mx_ready
log "就緒，等待伺服器啟動"
for _ in $(seq 300); do curl -sf http://127.0.0.1:8188/health >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8188/health >/dev/null && log "伺服器 OK" || { log "伺服器沒起來，看 /content/server.log"; tail -30 /content/server.log; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
