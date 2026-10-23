# Miroxen on Google Colab

用 Colab CLI 開 GPU VM，啟動 Miroxen HTTP API（Qwen-Image-2.1 生圖、MiniMax-H3 生影片）。
程式和模型放在 Google Drive；生成的圖片和影片每 5 秒自動同步到 Drive 的 `Miroxen/outputs/<開機時間>/`。

## 快速上手：開機 → 用 → 關機

以下指令都在 Windows 終端機、repo 根目錄執行（`deploy\mx-colab.cmd` 會轉進 WSL）；在 WSL 裡直接打 `mx-colab`（`~/.local/bin/mx-colab` 連到 `deploy/mx-colab`）。

**1. 開機**（約 4 分鐘）

```bat
deploy\mx-colab.cmd up --gpu G4
```

最後會印出**公開網址**（`https://xxxx.trycloudflare.com`），同時寫到 Drive 的 `Miroxen/endpoint.txt`。
G4 回 `Service Unavailable`（沒空機）就隔幾分鐘重試，或改 `--gpu A100`。

**2. 使用**

- **網頁面板**：Telegram bot 和 `mx-colab tunnel` 給的網址後面帶 `#key=金鑰`，點開就自動登入（金鑰不會送到伺服器，面板讀完即從網址列清掉，並記在這台瀏覽器）。只拿到公開網址時，在「API」分頁貼上金鑰（WSL 的 `~/.config/miroxen-colab/api_key`）。
- **API**：任何能發 HTTP 的地方都能呼叫（手機、其他電腦、n8n、自己的程式），標頭帶 `Authorization: Bearer <金鑰>`。

```bash
# 生圖（同步，幾秒就回）
curl -s https://xxxx.trycloudflare.com/v1/images/generations \
  -H "Authorization: Bearer <金鑰>" -H "Content-Type: application/json" \
  -d '{"model":"qwen-image-2.1-turbo","prompt":"雨夜的台北巷口","size":"1024x1024","steps":8,"response_format":"url"}'

# 生影片 5 秒 720p、4 步加速（非同步：先回工作物件，再 GET /v1/jobs/<id> 查到 succeeded 拿網址）
curl -s https://xxxx.trycloudflare.com/v1/videos/generations \
  -H "Authorization: Bearer <金鑰>" -H "Content-Type: application/json" \
  -d '{"model":"minimax-h3","prompt":"橘貓走過雨夜巷口","size":"1280x704","num_frames":124,"steps":5,"video_shift":6,
       "loras":[{"name":"minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"}]}'
```

完整參數見 [`docs/API.md`](../docs/API.md)。圖片端點的路徑與回傳格式仿 OpenAI 生圖 API。

**3. 關機**（**不用時一定要關，開著就在扣點**）

```bat
deploy\mx-colab.cmd down
```

### 串接到其他服務前要知道

- **網址每次開機都會變**：其他程式可讀 Drive 的 `Miroxen/endpoint.txt` 取得最新網址；要固定網址得改用有帳號＋網域的 Cloudflare 正式通道。
- **單次請求約 100 秒上限**（trycloudflare 臨時通道）：生圖沒問題；影片本來就是非同步工作，不受影響。
- **VM 開著才能呼叫**：開機約 4 分鐘、開著就扣點，適合「要用時開、用完關」，不適合 24 小時待命。

## 點數與使用紀錄

- 剩餘點數：`mx-colab usage`，或網頁上開任一 Colab 筆記本 →「連線」旁的 ▼ →「查看資源」（帳號要選 nico.94624@gmail.com；不要按「連線」本身，會開新機器扣點）。
- **Google 不提供每次使用的明細**。`mx-colab` 每次開機／關機會把時間、session、GPU、剩餘點數記到 Drive 的 `Miroxen/usage.csv`（可用 Google 試算表開），`mx-colab usage` 會列最近 20 筆。

## GPU 與速度

- Miroxen 需要 sm_80 以上（A100／L4／H100／G4…）。**T4 是 sm_75，部署流程能跑完，但生成會回 `no kernel image is available`。**
- 學生免費領的 Google AI Pro 屬促銷方案，**不含 Colab 運算單元**（只能開 T4）；付費 AI Pro 每月 200 單元。
- 付費 AI Pro 實測（2026-10-19）：G4、A100（40GB）、L4、T4 開得到；H100 被拒（Ultra 才有）。G4／A100 常回 `Service Unavailable`（沒空機），隔幾分鐘重試通常就有。
- 同一套程式包與 Drive 模型，換卡只改 `--gpu`；`--compute fp8` 在非 Blackwell 卡線性層自動退回 bf16；L4 等 sm_89 卡注意力仍走 fp8。

| 卡 | 每小時點數 | 圖片 Turbo 8 步 | 圖片標準 40 步 | 影片 4 步 LoRA | 影片完整 30 步 | 4 件併發吞吐 |
|---|---|---|---|---|---|---|
| G4（RTX PRO 6000，96GB，FP8） | ≈ 8.9 | 1.25s | 5.2s | 4.8s | 20.7s | 37 張／分 |
| A100 40GB（bf16） | 5.3 | 3.4s | 15.0s | 13.2s | 61.5s | 15.5 張／分 |
| L4 24GB（bf16） | 1.54 | 13s | 60s | 44.4s | 228s | 未測 |

圖片 1024²、影片 864×480 56 幀；生成秒數不含冷啟動（第一次載入模型多 5～80 秒）。G4 併發有 1.8 倍加速（多件時網路往返和生成重疊），其他卡只是排隊。
G4 每小時點數是推算值（G4＋A100 同時開的 14.2 減 A100 單獨的 5.3）。原始數據：`bench/results/` 的 `bench-G4.json`、`bench-A100.json`、`bench-L4-video.json`。

G4 接近 720p 影片（1280×704；720 不是 32 的倍數，要 720 可用 1280×736 再裁切），生成秒數：

| 長度 | 4 步 LoRA | 8 步 LoRA | 完整 30 步 |
|---|---|---|---|
| 5 秒（124 幀） | 28s | 48s | 153s |
| 10 秒（243 幀） | 74s | 131s | 433s |

原始數據：`bench/results/bench-G4-720p.json`（每項只測一次）。換算：一支 5 秒 4 步約 0.07 點，10 秒 4 步約 0.18 點。

### 和 ComfyUI 比較（G4，2026-10-21）

同一台 G4、同一批模型檔（圖片 `qwen-image-2.1-UC-Q8_0.gguf`，影片 `minimax_h3_fl2va_pruned-Q4_K.gguf`，文字編碼器與 VAE 也相同）、
原生步數不加速、cfg 1、euler；尺寸 1280×704（720 不是 32 的倍數）。每項暖機後量 2 次取中位數，輸出都目視確認正常。

| 項目 | Miroxen fp8（G4 預設） | Miroxen bf16 | ComfyUI（bf16） |
|---|---|---|---|
| 圖片 標準 40 步 | **4.56s** | 7.55s | 8.65s |
| 影片 5 秒 124 幀 30 步（含音訊） | **153s** | 281s | 252s |

- ComfyUI 讀 GGUF 一律用 bf16 計算，所以「bf16 對 bf16」才是同算法比較：圖片 Miroxen 快 13%，影片 ComfyUI 快約 10%（取樣 241s 對 271s）。
- Miroxen 在 Blackwell 上預設 fp8 運算，實際使用比 ComfyUI 快：圖片 1.9 倍、影片 1.6 倍。
- ComfyUI 版本 `7a28332`（0.42.0）＋ ComfyUI-GGUF `6ea2651`，要改兩處才讀得進這兩份 GGUF（`bench/comfy/comfy_setup.sh`、`bench/comfy/patch_te.py`）。
  腳本：`bench/comfy/mx_bench.py`（Miroxen）、`bench/comfy/comfy_bench.py`（ComfyUI，在 VM 上跑）；原始數據 `bench/results/bench-G4-vs-comfy.json`。
- 2026-10-22 優化 bf16 注意力 kernel 後（Miroxen `perf/bf16-attention`）：G4 bf16 影片 281s → 256s，和 ComfyUI 252s 差不多。
  單測注意力（S=32560、56 頭、D=128）：

  | GPU | 舊版 | 新版 | PyTorch FlashAttention |
  |---|---|---|---|
  | A100 | 113 | 178 | 213 |
  | G4 | 291 | 330 | 351 |
  | L4 | 44.6 | 58.6 | 63 |

  單位 TFLOPS。影片 256s 是用 key tile 64 的試驗版量的；正式版為了精度用 32，注意力慢約 4%，預估約 260s。
- 2026-10-22 L4 改讓注意力走 fp8（Miroxen `perf/ada-fp8-attention`，PR #3）：`--compute fp8` 在 L4 線性層仍退回 bf16，但注意力用 INT8/FP8（91 vs 61 TFLOPS）。
  L4 影片 1280×704、124 幀、4 步：總 189s vs 222s（去噪 136s vs 167s），畫面正常。預設就是 fp8，不用加參數；要全 bf16 用 `--compute bf16`。
- 2026-10-22 bf16 softmax 改用硬體 exp2（sm_120 以下）：A100 182→187、L4 59.7→60.3 TFLOPS；G4 不變。
- 2026-10-22 G4 剖析（nsys）：DiT 每步注意力佔 50%、線性層 34%；VAE 解碼改成 CPU 轉 RGB 和 GPU 重疊，4 步影片 22.96s → 22.37s（VAE 7.54s → 6.88s）。
  A100 試過 INT8 注意力：和 bf16 一樣快（190 TFLOPS），沒採用。

## 第一次設定（只做一次，已完成）

```bat
deploy\mx-colab.cmd login
```

1. **Colab CLI 登入**：印出網址 → 用有付費 AI Pro 的帳號開啟並同意 → 把授權碼貼回終端機。
2. **rclone 連 Google Drive**：印出網址 → 同一個帳號同意即可（之後每台 VM 都用這個 token，不用再按）。
3. 產生 API 金鑰，存在 WSL 的 `~/.config/miroxen-colab/api_key`。

確認帳號能開哪些 GPU（每種開一下立刻關，會用到極少量點數）：`deploy\mx-colab.cmd probe`

上傳程式包（換 Miroxen 版本時再跑一次）：`deploy\mx-colab.cmd bundle`。程式包會帶 `models/*/hf/` 底下的 tokenizer／設定檔，打包前本機 `models/` 要先有這些檔（缺了 VM 生圖會回「無法開啟 tokenizer」）。程式包要用 **CUDA ≥ 13.3** 建置（12.8／13.0 會錯編，`build.sh` 會擋）：

```bash
# 在 repo 根目錄（WSL）
CUDACXX=/usr/local/cuda-13.3/bin/nvcc ./build.sh ~/mx-build13
```

模型不用手動上傳：VM 缺模型時會直接從 Hugging Face 下載，並在背景補傳到 Drive（目前 Drive 已有全部 14 個，63.2GB）。
每次開機先拿最大的檔案兩邊各拉最多 256MB／15 秒測速，哪邊快就從哪邊拉（Drive 同一天下載太多會被限速；實測 Drive 0～53 MB/s、HF 37～120 MB/s，選 HF 時 G4 開機 234 秒，從 Drive 拉要 648 秒）。
2026-10-22 起 aria2 安裝與程式包下載改在背景和模型下載同時進行，並拿掉 Drive 限速時會卡 50 秒的 `rclone mkdir`：L4 影片群組開機 319 → 197 秒（Drive 限速中）。

## 全部指令

| 指令 | 作用 |
|---|---|
| `mx-colab up [--gpu G4] [--groups "image video"] [--private] [--compute fp8]` | 開機並上線。`--groups image` 只載圖片模型，開機更快；`--private` 不開公開網址 |
| `mx-colab status` | VM、GPU 顯存、佇列、公開網址 |
| `mx-colab usage` | 剩餘點數＋最近 20 筆開關機紀錄 |
| `mx-colab tunnel` | 重開本機 `http://127.0.0.1:8188` 轉發（WSL 指令結束轉發就斷，平常用公開網址） |
| `mx-colab logs` | 伺服器最近的 log |
| `mx-colab bench [--only image\|video] [--concurrency 8]` | 測速：圖片標準／Turbo、影片完整／8 步／4 步、併發吞吐 |
| `mx-colab bench-long` | 測速：1280×704 的 5 秒／10 秒影片 |
| `mx-colab down` | 最後同步一次輸出後關機 |
| `mx-colab probe [G4 H100 A100 L4 T4]` | 逐一試開 GPU 看帳號權限 |

同時開多台（每台用不同 session 名稱和本機 port；第二台起加 `MX_SEED=0`，不往 Drive 補傳，避免兩台同時傳出重複檔）：

```bash
MX_SESSION=mxa MX_PORT=8189 MX_SEED=0 mx-colab up --gpu A100 --compute bf16
MX_SESSION=mxa MX_PORT=8189 mx-colab bench      # status／logs／down 也要帶同樣的變數
```

多台同時從 Drive 拉模型會互相拖慢（實測 A100 拉了 23 分鐘）；開機測速會自動改走 HF，影響較小。

## 安全

- `/v1/*` 都要 Bearer 金鑰。
- 公開網址（trycloudflare）是隨機子網域；但 Miroxen 的 `/`、`/files/` 本來就不需金鑰，**拿到網址的人可以看到生成檔**。不想公開就用 `--private`，只走本機 SSH 轉發。
- 本機的 Colab 憑證（`~/.config/colab-cli/token.json`）會複製到 VM 上供 keepalive 使用（只有你能存取的 VM，關機即消失）。

## 運作原理

```
本機（WSL Ubuntu-24.04）           Colab VM（G4 / A100 / L4）              Google Drive
mx-colab up ──colab CLI──────────▶ setup.sh：拉程式包＋模型到本機 SSD ◀──── Miroxen/bin、models（或 HF）
             ──ssh（colab ssh）──▶ serve.py（kernel cell 看守伺服器）
127.0.0.1:8188 ◀──SSH 轉發──────── miroxen serve :8188
任何地方 ◀──trycloudflare 網址───── cloudflared                      ────▶ Miroxen/outputs/、endpoint.txt、usage.csv
```

**VM 怎麼保持不被回收**：Colab 只有在 kernel 正在執行、而且有 client 連著時才不回收 VM（實測 client 斷線後 10 分鐘內就被回收）。
所以 `setup.sh` 會在 VM 上裝 colab CLI，**由 VM 自己**把 `serve.py` 送進自己的 kernel，斷線 10 秒後自動重送；
你的電腦關機或 WSL 停掉都不影響（實測 T4：本機零連線，VM 撐滿 3 小時、伺服器全程正常）。
`colab exec` 斷線後會卡住不結束，`keepalive.sh` 看到「Connection was lost」就砍掉重送。
**不要自己對 `mx` session 下 `colab exec`**：Colab 會換一個新 kernel，把 keepalive 的連線擠掉（會自動重送，但伺服器會短暫重啟）。

**SSH**：每台 VM 同時只准一條 `colab ssh`（多的回 HTTP 429），`mx-colab` 用 SSH ControlMaster 讓所有指令共用一條。

## 檔案

- `mx-colab`：控制腳本（WSL）；`mx-colab.cmd`：Windows 入口
- `vm/setup.sh`：VM 上安裝 rclone／aria2／cloudflared、測速選來源、拉程式包與模型、補傳 Drive、輸出同步、公開網址
- `vm/serve.py`：在 kernel 裡長時間執行，啟動並看守 `miroxen serve`
- `vm/keepalive.sh`：VM 上把 `serve.py` 送進自己 kernel 的迴圈（斷線偵測＋重送）
- `models.tsv`：模型清單（Hugging Face 來源、鎖定 commit、位元組數）
- `bot/tg_bot.py`：Telegram bot（自動搶機＋閒置關機，見下節）
- `bench/bench.py`、`bench/bench_long.py`：測速；`bench/results/bench-*.json`：實測原始數據
- `bench/comfy/`：和 ComfyUI 比速度用的安裝、補丁與測速腳本
- `tools/probe_gpus.sh`：試開各種 GPU；`tools/probe_env.py`：印出 VM 環境資訊

## Telegram bot（自動搶機＋閒置關機）

`bot/tg_bot.py`（只用 Python 標準函式庫，在 WSL 跑）：

```bash
TG_TOKEN=<BotFather 給的 token> TG_CHAT_ID=<你的 chat id> nohup python3 deploy/bot/tg_bot.py > ~/tg_bot.log 2>&1 &
```

若只用 BotFather 的允許名單，先在 BotFather 開啟 `Restrict bot usage` 並設定名單；不設定 `TG_CHAT_ID` 時，程式預設就是此模式（不需填數字 ID）：

```bash
TG_TOKEN=<BotFather 給的 token> nohup python3 deploy/bot/tg_bot.py > ~/tg_bot.log 2>&1 &
```

存取控制可選兩種方式：

- **程式 ID 白名單**：設定 `TG_CHAT_ID`，多個 ID 用逗號分隔。若 BotFather 的 `Restrict bot usage` 也開啟，兩層都要允許（實際權限是交集）。
- **只用 BotFather 名單（預設）**：不設 `TG_CHAT_ID` 即可；必須在 BotFather 開啟 `Restrict bot usage` 並設定允許使用者。程式無法讀回或驗證 BotFather 的開關，若關閉此限制，任何能傳訊息的人都可能操作 bot。

若 `TG_CHAT_ID` 與 BotFather 限制同時使用，兩層都要允許；程式實際權限是兩者交集。

bot 啟動時會自動向 Telegram 註冊並讀回驗證 `/up`、`/status`、`/down`、`/id`，指令選單會顯示在聊天輸入框；只想單獨註冊、不啟動 bot 時：

```bash
TG_TOKEN=<BotFather 給的 token> python3 deploy/bot/tg_bot.py --register-commands
```

- `/up`：依 `MX_GPUS`（預設 `G4 A100 L4`）逐一 `mx-colab up`，成功就回公開網址；失敗會先 `mx-colab down` 收掉半開的 VM 再試下一張。
- 每分鐘查公開網址的 `/v1/jobs`：`MX_IDLE_MIN`（預設 30）分鐘內沒有新工作、也沒有排隊／執行中的就自動關機；連續 5 分鐘連不上（VM 被回收）也會關。
- `/status`、`/down`、`/id`。未設定 `TG_CHAT_ID` 時，必須明確選擇 BotFather 存取模式；未配置任何存取控制時 bot 不會啟動。
- bot 跑在本機 WSL，電腦關機或 WSL 結束 bot 就停（VM 不受影響，但不會再自動關機）。
