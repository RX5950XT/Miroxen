# CONTEXT：Miroxen on Google Colab

目標：把 Miroxen（Qwen-Image-2.1／MiniMax-H3 的 C++/CUDA 推論引擎）部署到 Colab GPU，模型與輸出放 Google Drive，隨時一鍵上線。

## 目錄
- `deploy/`：Colab 部署工具（2026-10-22 從 `colab_miroxen/` 搬進本 repo）。根目錄放日常用的 `mx-colab`、`mx-colab.cmd`、`models.tsv`；`vm/` 是在 VM 上跑的腳本；`bot/` 是 Telegram bot；`bench/` 是測速腳本與結果（`results/`）、ComfyUI 對照（`comfy/`）；`tools/` 是一次性診斷工具。說明見 `deploy/README.md`。
- WSL 的 `~/.local/bin/mx-colab` 連到 `deploy/mx-colab`；程式包 `bundle` 直接用本 repo 的 `web/` 與 `models/*/hf/`，Linux 建置在 `~/mx-build13`。

## 已確認的事實（2026-10-19）
- Colab CLI 0.7.4 只支援 Linux/macOS → 裝在 WSL `Ubuntu-24.04`。
- Colab VM：Ubuntu 24.04、glibc 2.39、驅動 580（CUDA 13.0）、libcuda 在 `/usr/lib64-nvidia`（ssh 環境沒有這個路徑）。
- **CUDA 12.8／13.0 的 nvcc 錯編 Miroxen**（half→byte），必須用 13.3（WSL 已裝 `/usr/local/cuda-13.3`）。13.3 的 runtime/cuBLAS 在驅動 580 上實測可用。
- `drivemount` 每台新 VM 都要瀏覽器授權 → 改用 rclone（一次授權）。
- VM 需要「kernel 執行中＋有 client 連著」才不被回收 → VM 自己用 colab CLI 送 cell（keepalive）。
  實測（T4、nico 帳號）：本機零連線下存活 ≥3 小時、health 全程 ok；對照組 client 斷線 10 分鐘內被回收。
- 完整部署流程（rclone／aria2 從 HF 下載／補傳 Drive／輸出同步／cloudflared／API 金鑰）已在 Colab CPU 與 T4 上用假 Drive 跑通。
- 本機 5060 Ti（WSL）實測：Turbo 8 步 5.66s、標準 40 步 25.49s（0.618 s/步）、影片 4 步 LoRA 27.13s（冷啟動 115s）；併發 3 件 speedup 0.88（只排隊）。
- `colab ssh -s NAME` 找不到 session 會自動開新 VM → `mx-colab` 每次 ssh 前先檢查。
- Miroxen 同時只跑一件工作；FP8/FP4 只在 Blackwell（G4 = RTX PRO 6000，sm_120）可用，H100/A100 走 BF16。
- Colab CLI 與 rclone `gdrive:`（Drive 5TB）都登入 nico.94624@gmail.com；`bundle` 已上傳。
- 學生促銷版 AI Pro 沒有 Colab 運算單元（只能開 T4）；使用者 2026-10-19 改為付費 AI Pro 後才有 200 單元。
- T4 實跑（真 Drive）：開機 168 秒（模型從 Drive 拉）、公開網址／401／本機轉發正常、`--fake` 生成的檔案約 20 秒內出現在 Drive。
  真生成在 T4 回 `no kernel image`（sm_75 不支援，Miroxen 用 bf16 mma 需 sm_80+）。
- 每台 VM 同時只准一條 `colab ssh`（多的回 429，剛關掉的連線也會擋一陣子）→ `mx-colab` 用 SSH ControlMaster 共用一條。
- 本機 WSL 結束時 SSH 主連線（含 8188 轉發）會一起消失；公開網址不受影響。
- Hugging Face repo 會改名（AtomicChat 的 AD-Q4_K 已改到 `Qwen-Image-2.1-Turbo-Uncensored-GGUF`），`models.tsv` 盡量鎖 commit。

- 2026-10-19 使用者改為付費 AI Pro（200 單元）：G4／A100 40GB／L4／T4 可開、H100 被拒；G4／A100 常暫時沒空機。
  實測 G4（RTX PRO 6000）圖片 Turbo 1.25s、影片 4 步 4.8s，約 8.9 單元／時；A100 慢約 3 倍、5.3 單元／時；L4 慢約 12 倍。數據見 `deploy/README.md`。
- `mx-colab` 支援多台並行（`MX_SESSION`／`MX_PORT`／`MX_SEED=0`），公開網址存在 `endpoint-<session>`。

- 2026-10-20 Drive 已有全部 14 個模型（63.2GiB）。`setup.sh` 開機時 Drive／HF 各測 256MB 選快的（Drive 當天下載多會被限速到 15～50 MB/s；選 HF 後 G4 開機 234 秒）。
- G4 1280×704 影片：5 秒 4 步 28s、10 秒 4 步 74s、10 秒完整 30 步 433s（`mx-colab bench-long`）。
- Google 不提供使用明細 → `mx-colab` 開關機時把剩餘點數記到 Drive `Miroxen/usage.csv`（`mx-colab usage` 查看）。2026-10-20 10:15 剩 189.37 點。
- 2026-10-21 G4 對 ComfyUI（同模型檔、原生步數、1280×704）：圖片 40 步 Miroxen fp8 4.56s／bf16 7.55s／ComfyUI 8.65s；
  影片 5 秒 30 步 153s／281s／252s。H3 的 bf16 路徑比 ComfyUI 慢約 10%。
  根因（2026-10-21 G4 實測）：Miroxen bf16 注意力 kernel 約 295 TFLOPS，PyTorch SDPA cuDNN 約 390、FlashAttention 約 355；
  cuBLAS bf16 GEMM 416 TFLOPS 沒問題。H3 每步注意力佔約 55% 計算，換算每支影片慢約 25～37 秒，與實測差 30 秒相符。
  Miroxen fp8 注意力約 590 TFLOPS，所以 G4 預設不受影響；影響 A100／L4 等走 bf16 的卡。測速程式：`deploy/bench/comfy/attn_bench.cu`（連結 libmiroxen_core.a 編譯）、`deploy/bench/comfy/sdpa_bench.py`。細節見 `deploy/README.md`。
- 2026-10-22 已優化 bf16 注意力 kernel（Miroxen 分支 `perf/bf16-attention`，Drive 程式包已更新）。S=32560 H=56 D=128：
  A100 113→178、G4 291→330、L4 44.6→58.6 TFLOPS（PyTorch Flash 分別 213／351／63）。key tile 固定 32，改 64 在 G4 再快約 4%，但 `h3_dit_parity` 音訊誤差會超標。
  G4 bf16 影片 30 步實測 256s（用 key tile 64 的試驗版量的，正式版預估約 260s；未重測）。
- 2026-10-22 L4（sm_89）在 `--compute fp8` 下注意力改走 fp8（分支 `perf/ada-fp8-attention`，PR #3 疊在 #2 上）。
  坑：Ada 的 fp8 MMA 累加精度不足 fp32，長序列誤差 0.11、影片變雜訊；P·V 改兩段累加後 0.0084。L4 影片 222s→189s。
  `h3_dit_parity` 不適合驗 fp8 注意力（G4 預設全 fp8 本來就過不了），改用 fp8 vs bf16 單測誤差加看畫面。
- 2026-10-22 bf16 softmax 在 sm_120 以下改用 `ex2.approx.ftz`（沒開 fast-math 的 exp2f 每元素多 3 道指令）：A100 182→187、L4 59.7→60.3；G4 會慢 1% 所以不改。
  `h3_dit_parity` 音訊門檻 0.05 落在 bf16 捨入雜訊內（換 fp32 參考注意力也 0.085 超標）；K=64 在 A100／L4 反而慢，維持 32。
- 2026-10-22 `setup.sh` 開機加速：aria2（apt）與拉程式包改背景並行、測速時限 60→15 秒、拿掉 Drive 限速時卡 50 秒的 `rclone mkdir`。L4 影片群組 319→197 秒。
- 2026-10-22 nsys 剖析 G4 fp8 影片（1280×704、124 幀；工具：WSL 的 nsys target-linux-x64 傳上 VM，`.qdstrm` 拉回本機轉檔）：
  DiT 每步 5.3s、GPU 忙 95%（注意力 50%／線性層 34%／逐元素約 16%）；VAE 7.6s 只忙 72%，空檔是 CPU 轉 RGB → 改背景轉，VAE 7.54→6.88s（PR #4）。
  A100 試 INT8 Q·Kᵀ＋bf16 P·V：和 bf16 一樣 190 TFLOPS、影片 60s 無差，不啟用；只給 sm_86～88（3060 Ti 快 1.5～1.8 倍）。
  去噪本身不是逐位元可重現（`kv_stats` 用 float atomicAdd），新舊版比影片要看畫面，不能比 md5。
- 回報使用者一律繁體中文（使用者多次因英文回覆不滿）。
