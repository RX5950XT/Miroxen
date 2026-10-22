# Miroxen

針對 **Qwen-Image-2.1**（圖片生成／編輯）與 **MiniMax-H3**（影音生成）寫的 C++／CUDA 推論引擎。
提供命令列、HTTP API 與瀏覽器工作台；推論不需要 Python、PyTorch 或 ComfyUI。

- **圖片**：**標準**（40 步）與 **Turbo**（Qwen 官方 8 步蒸餾版）兩個模型可直接切換；支援編輯、多張參考圖、透明圖。
- **影音**：文字／首尾幀生成（FL2VA）與參考素材生成（Ref2VA），含配套音訊。
- **硬體**：雙卡、單張 16 GB、單張 8 GB 都能跑圖片與影片，依剩餘顯存自動配置。
- **LoRA**：支援 diffusers／PEFT／kohya 命名；工作台可拖檔或貼網址安裝社群 LoRA。

開發機：RTX 5060 Ti 16 GB＋RTX 3060 Ti 8 GB、Ryzen 7 5700X、96 GB RAM。

## 快速開始

需要 Windows、Visual Studio 2022 C++ 工具、CUDA 13.3、CMake ≥ 3.28、Ninja；影片另需 `ffmpeg`、`ffprobe` 在 PATH。
模型檔要先下載，見[模型下載](#模型下載)。

**一鍵啟動**：雙擊根目錄的 `start.bat`。沒建置會先建置，服務沒在跑就在背景啟動（不開視窗，日誌在 `out/logs/server.log`），再自動打開工作台 <http://127.0.0.1:8188>。要停止服務就雙擊 `stop.bat`。
額外參數會傳給 `serve`，例如 `start.bat --gpus 0`。

手動啟動：

```bat
build.bat
build\miroxen.exe serve --port 8188
```

`build.bat` 會固定 UTF-8 字碼頁，讓中文版 MSVC 的標頭依賴能被 Ninja 正確記錄。

### Linux / Colab 建置

安裝 **CUDA ≥ 13.3**（12.8／13.0 的 nvcc 會錯編 Q8_0 量化，`build.sh` 會擋下）、CMake ≥ 3.28、Ninja 與 ffmpeg 後，在 Linux 執行 `CUDACXX=/usr/local/cuda-13.3/bin/nvcc ./build.sh ~/mx-build`；預設編譯 sm_120a／80／86／89／90（Windows 的 `build.bat` 相同），也可用 `CUDA_ARCHS="120a;86"` 指定目標。啟動工作台可用 `~/mx-build/miroxen serve --host 0.0.0.0 --port 8188`；sm_90／80 會使用 BF16 運算後備路徑。

## 工作台

- **創作**：圖片（標準／Turbo）或影音、參考素材、LoRA、模型檔與量化版本、運算精度與取樣設定。進階設定預設收合，摘要顯示步數、CFG、種子。
- **頂部列**：連線狀態、目前模型（點開可載入、切換、卸載）、雙卡／單卡標籤。同時只保留一顆模型，生成中不能管理模型。
- **API／硬體**：可複製的 API 網址與區網使用說明，CPU／RAM／各 GPU 的用量、溫度與功耗。

完整 API 契約見 [docs/API.md](docs/API.md)。

## 模型下載

模型權重不放進 Git，請依各來源頁面的授權使用。粗體為預設載入的檔案。

### 圖片：`models/image/qwen-image-2.1/`

原始模型：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。

| 元件 | 檔案 | 來源 |
|---|---|---|
| 標準 DiT · Q8_0 | **`qwen-image-2.1-UC-Q8_0.gguf`** | [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) |
| Turbo DiT · Q8_0 | **`Qwen-Image-2.1-Turbo-Q8_0.gguf`** | [AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)（原始：[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)，Qwen Research License） |
| 文字編碼器 · AD-Q4_K（去拒答） | **`Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf`** | [AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF)（2026-10-13 改名前的 revision `e3b01e9b…`） |
| 視覺編碼器 · F16（編輯／參考圖用） | **`mmproj-Qwen3VL-8B-Instruct-F16.gguf`** | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| 紋理修正 VAE · BF16 | **`vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors`** | [madebyollin/texture-fix-vae-for-qwen-image-2.1](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/tree/1f80456b6de9d89043c133ff1f3875449b8bd016) |
| 官方 VAE · BF16（沒有紋理修正版時使用） | `vae/qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| Tokenizer／設定 | `hf/processor/` 等 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |

```bat
hf download abenzerps/Qwen-Image-2.1-Uncensored-GGUF qwen-image-2.1-UC-Q8_0.gguf --revision 6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 --local-dir models/image/qwen-image-2.1
hf download AtomicChat/Qwen-Image-2.1-Turbo-GGUF Qwen-Image-2.1-Turbo-Q8_0.gguf --local-dir models/image/qwen-image-2.1
hf download AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf --revision e3b01e9bfacc373148bdb93824496e0c72643250 --local-dir models/image/qwen-image-2.1
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir models/image/qwen-image-2.1
```

- **Turbo** 是 Qwen 官方把同一顆模型蒸餾成 8 步、CFG 1、固定排程（`1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`，不位移）的版本，1024² 約比標準 40 步快 4.5 倍。
- **文字編碼器**用 AtomicChat 去拒答版：發布者量測對出圖沒有可量到的差異，AD-Q4_K 比一般 Q4_K_M 更接近 BF16（LPIPS 0.087 對 0.17），5.3 GB 放得進 8 GB 卡。找不到時會退回原版 `Qwen3VL-8B-Instruct-Q4_K_M.gguf`。
- **UC-Q8_0** 固定在 revision `6b34e594…`，SHA-256 `cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8`。名稱與行為依發布者模型卡，本專案未量測其內容限制或與原版的畫質差異。
- **紋理修正 VAE** 固定在表中的 revision（SHA-256 `05e9af8da4697d1a5118b674c3d90e95f932702d2c5a8900d7d4f05076403cc7`），只微調解碼器，與作者最新檔案不同。

### 影音：`models/video/minimax-h3/`

原始模型：[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)。

| 元件 | 檔案 | 來源 |
|---|---|---|
| 文字／首尾幀 DiT · Q4_K | **`minimax_h3_fl2va_pruned-Q4_K.gguf`** | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| 參考生成 DiT · Q4_K | **`minimax_h3_ref2va_pruned-Q4_K.gguf`** | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| 文字／視覺編碼器 · Q3_K_S（去拒答，含視覺塔） | **`MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S_vis.gguf`** | [pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF](https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF) |
| 文字／視覺編碼器 · Q4_K_M（原版，找不到上面時使用） | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| 影片 VAE · FP16 | **`vae/minimax_h3_video_vae_fp16.safetensors`** | [unsloth 配套檔](https://huggingface.co/unsloth/MiniMax-H3-GGUF/tree/d629413c2e5b51b38c453668b75ca3b06ca92703/vae)；[Comfy-Org 原始轉換](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| 音訊 VAE · FP32 | **`vae/minimax_h3_audio_vae_fp32.safetensors`** | 同上 |
| Tokenizer／設定 | `hf/tokenizer/` 等 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |

```bat
hf download unsloth/MiniMax-H3-GGUF minimax_h3_fl2va_pruned-Q4_K.gguf minimax_h3_ref2va_pruned-Q4_K.gguf vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors --local-dir models/video/minimax-h3
hf download pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-Q3_K_S_vis.gguf --local-dir models/video/minimax-h3
```

- 文字編碼器預設用 pottokao 去拒答版的 `_vis` 檔（12.6 GB，比原版 Q4_K_M 小 5.6 GB）；只有 `_vis` 版含參考圖／首尾幀需要的視覺塔。同 seed 文字生影片與參考生影片實測畫面與原版同級，精度較低（約 3.5 bit）對長提示詞的影響未量測。
- FL2VA 與 Ref2VA 是不同 checkpoint，給參考素材時自動換用 Ref2VA。
- 參考圖預設縮放到短邊 1024（官方 2048，可用 `--ref-edge 2048` 或 API `ref_image_short_edge` 改回）。864×480 輸出實測人物身分與細節一致，參考生影片從 104 秒降到 59 秒。
- 參考素材最多 12 個：圖片 9 張、影片 3 段、音訊 3 段；音訊不能單獨當參考。
- 輸出固定 24 fps，幀數向上對齊 `17n+5`、最少 22 幀（例如填 240 幀實際生成 243 幀）。官方模型最長 15 秒（361 幀）；更長的影片要分段，用前一段最後一幀當下一段的 `--first`。
- 影片取樣依序列長度（解析度 × 幀數 × 參考素材）自動配置顯存：放得下就照常；不夠先卸下 VAE，再縮小分段；還是不夠就直接報錯（不讓 Windows 偷借系統記憶體）。有多的顯存就拿來常駐 DiT 層，省掉每步重搬權重。736×1280・15 秒在 16 GB 卡（桌面程式佔 2.5 GB）實測可跑。
- 序列平行：兩張 RTX 50 系列（sm_120）時自動啟用，DiT 的列與注意力頭對半分給兩張卡、每層交換兩次；每張卡的取樣顯存約減半，長影片預估快約 1.8 倍。環境變數 `MX_H3_SP=1` 強制開啟、`=0` 關閉。開發機只有一張 50 系列，跨 5060 Ti + 3060 Ti 只驗證過正確性（3060 Ti 只能用 bf16，反而比單卡慢）。

## 單卡使用（8 GB／16 GB）

```bat
build\miroxen.exe serve --gpus 0     :: 只用 GPU0（開發機的 5060 Ti 16 GB）
build\miroxen.exe serve --gpus 1     :: 只用 GPU1（開發機的 3060 Ti 8 GB）
```

`--gpus` 用 nvidia-smi 的編號；不加則雙卡都用。程式依剩餘顯存自動配置：

| 配置 | 圖片 | 影片 |
|---|---|---|
| 雙卡 | DiT 常駐主卡、文字編碼器在副卡 | DiT／編碼器逐層串流，音訊 VAE 在副卡 |
| 單卡 16 GB | 文字編碼器逐層串流，DiT 常駐 | 顯存不夠時取樣前先卸下 VAE，解碼再載回 |
| 單卡 8 GB | 文字編碼器與 DiT 都逐層串流 | 同上，且 VAE 解碼器權重存成 Q8_0（4.8→2.5 GB） |

開發機實測（相同提示詞與 seed、CLI 單次執行、桌面程式照常開著，非嚴格基準）：

| 工作 | 雙卡 | 只用 16 GB | 只用 8 GB |
|---|---|---|---|
| Turbo 1024²、8 步 | 6.9 秒 | 7.7 秒 | 18.4 秒 |
| 標準 1024²、40 步 | 31.1 秒 | — | — |
| 標準編輯 1024²、12 步 | — | — | 36.4 秒 |
| 影片 864×480、56 幀、4 步 LoRA | 30.5 秒 | 30.6 秒 | 67.8 秒 |

3060 Ti 是 Ampere（sm_86），沒有 FP8 tensor core，一律用 BF16 運算，所以比較慢。
Windows 顯存不夠時不會報錯，而是偷借系統記憶體、速度掉十幾倍；單卡配置因此會主動卸載暫時用不到的元件。

## LoRA

圖片 LoRA 放 `models/loras/image/`，影片 LoRA 放 `models/loras/video/`；API 與 CLI 直接寫檔名，不用加子目錄。

**安裝社群 LoRA**：工作台「LoRA → 安裝社群 LoRA」可拖放 `.safetensors`，或貼 Hugging Face 檔案頁／Civitai 模型頁網址。
伺服器會讀檔內的層名稱，自動判斷是圖片或影片 LoRA，並放進對應資料夾；不是這兩顆模型的檔案會被拒絕，不留殘檔。
需要登入的 Civitai 模型請先設定環境變數 `CIVITAI_TOKEN`（Hugging Face 私有模型用 `HF_TOKEN`）再啟動服務。

圖片 LoRA 標準與 Turbo 共用；在原版訓練的畫風 LoRA 疊在 Turbo 上一樣能用，細節可能稍柔。圖片加速請直接用 Turbo 模型。

| 用途 | 檔案 | 來源 | 設定 |
|---|---|---|---|
| 圖片 · 原神 TCG 畫風（已實測） | `genshintcg_style_st4000.safetensors` | [wesjos/Qwen-Image2.1-Genshin-2D-Lora](https://huggingface.co/wesjos/Qwen-Image2.1-Genshin-2D-Lora) | 依需求調強度 |
| 影音 · FL2VA 8 步加速 | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 9`、`video_shift: 6` |
| 影音 · FL2VA 4 步加速 | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 5`、`video_shift: 6` |
| 影音 · Ref2VA 8 步加速 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 9`、`video_shift: 12` |

工作台勾選影片加速 LoRA 會自動套用上表設定，取消後回到原版排程；FL2VA 與 Ref2VA 的 LoRA 不能混用，分開列出。
FP8／FP4 路徑會把 LoRA 併入權重（H3 在逐層搬運時合併）；BF16 路徑每步即時計算。

### 步數建議

| 模型 | 建議起點 | 說明 |
|---|---|---|
| Qwen-Image-2.1 Turbo | **固定 8 步** | 模型自帶排程、CFG 1，不能改步數 |
| Qwen-Image-2.1 標準：生成／編輯／透明圖 | **40 步** | [官方範例](https://huggingface.co/Qwen/Qwen-Image-2.1#quick-start)；原版排程 `shift_terminal: true` |
| MiniMax-H3：FL2VA／Ref2VA | **30 個取樣點** | 實際 29 次 DiT 計算；影片 shift 12 |

H3 的 `steps` 包含最後的零點，所以「8 次計算」要填 9（[上游排程實作](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_minimax_h3.py)也這樣區分）。
這些是起點不是保證；步數少可當預覽，細節與動作穩定度需逐張確認。

## 命令列

```bat
build\miroxen.exe image -p "一隻在雪地裡的紅狐狸" -W 1024 -H 1024 --steps 40
build\miroxen.exe image --turbo -p "A red panda" -W 1024 -H 1024
build\miroxen.exe image -p "把背景換成夜晚的城市" --ref 原圖.png --steps 40
build\miroxen.exe image --gpus 1 --turbo -p "A red panda" --lora genshintcg_style_st4000.safetensors:0.8
build\miroxen.exe video -p "A red panda eating bamboo" -W 864 -H 480 --frames 56 --steps 30
```

- 圖片最多 10 張參考圖；沒指定尺寸時依最後一張參考圖比例，約 1024² 像素。
- 透明圖／去背使用模型的 RGBA 能力；局部修改靠模型理解遮罩，不保證遮罩外像素完全不變；面板可直接在原圖上畫遮罩。
- `--compute bf16|fp8|fp4` 控制運算精度（預設 FP8，和模型量化是兩回事）；不支援 FP8／FP4 的卡自動退回 BF16。

## 區網與 API

服務預設綁定 `0.0.0.0`，同網路的其他電腦可直接用：

1. 在工作台的「API／硬體」頁找和對方同網段的網址，對方開「工作台」連結即可生成，不用安裝模型。
2. 支援圖片生成 API 的程式，Base URL 填 `http://<這台電腦的區網 IP>:8188/v1`；圖片模型 ID 為 `qwen-image-2.1` 或 `qwen-image-2.1-turbo`。
3. 有設 `--api-key` 時，客戶端要帶 `Authorization: Bearer <API key>`。

圖片用 `POST /v1/images/generations`，圖片／影音非同步工作用 `POST /v1/jobs`；沒有聊天端點。
Windows 防火牆需允許 TCP 8188 入站，可用 `http://<IP>:8188/health` 確認連通。只要本機存取時加 `--host 127.0.0.1`。

## AxonDeck 整合與發行

發行前需安裝上述建置工具、Git、GitHub CLI（`gh`，已登入且有 `RX5950XT/Miroxen` 發行權限），以及 PATH 中的 `ffmpeg.exe`、`ffprobe.exe`。

```bat
release.bat 0.1.0
release.bat 0.1.1 -RebuildDeps
release.bat 0.0.0 -DryRun -RebuildDeps
```

也可直接執行 `powershell -NoProfile -ExecutionPolicy Bypass -File tools\release.ps1 -Version 0.1.0`。版本只接受 `數字.數字.數字`，tag 為 `v<版本>`，已有 tag 或輸出目錄會拒絕覆寫。正式發行要求工作樹乾淨；建置、打包後印出大小與 SHA-256，再建立並推送 tag、建立正式 GitHub Release，說明使用上一個 tag 到 HEAD 的 `git log --oneline`。

產物放在 `out/release/v<版本>/`，ZIP 直接從根層解開：

| 固定檔名 | 根層內容 |
|---|---|
| `miroxen-win-x64.zip` | `miroxen.exe`、單行 UTF-8 無 BOM 的 `VERSION`（例如 `v0.1.0`）、完整原樣的 `web/` |
| `miroxen-deps-win-x64.zip` | `cublas64_13.dll`、`cublasLt64_13.dll`、`msvcp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`、`ffmpeg.exe`、`ffprobe.exe`、`THIRD_PARTY_NOTICES.txt` |

CUDA DLL 取自 `build/CMakeCache.txt` 的 `CMAKE_CUDA_COMPILER` 所屬 Toolkit（依序找 `bin/x64`、`bin`），MSVC runtime 取 VS 2022 最新版 Redist 的 `x64/Microsoft.VC143.CRT`。FFmpeg 工具取 PATH，WinGet 符號連結會先解析成實體檔。授權與來源連結寫入 notices，包含 CUDA cuBLAS、MSVC runtime、FFmpeg GPL 建置資訊。

預設直接下載並重用上一版正式 release 的 deps ZIP，保留原始位元組與 digest；只有 `-RebuildDeps` 或上一版缺少該資產時才重新打包。下載失敗會停止。`-DryRun` 只在本機建置與打包，不要求 `gh`、不連 GitHub、不修改 Git；因此 deps 也從本機重建，工作樹不乾淨只警告。

AxonDeck 從 GitHub 的 **latest release** 下載上述固定檔名，以 API 回傳的 `digest`（SHA-256）驗證；deps digest 沒變就不重下。將執行檔、工作台和依賴解開到 runtime 後啟動：

```bat
miroxen.exe serve --host 127.0.0.1 --port <隨機埠> --api-key <隨機金鑰> --models <模型目錄> --out <輸出目錄> --web <runtime>\web
```

iframe 網址範例：`http://127.0.0.1:<埠>/?embed=1&theme=dark&c-accent=%23ff0000#key=<金鑰>`。

- `embed=1`：隱藏頂部 Miroxen 品牌，其他控制項保留；另外顯示「重新啟動」「停止」（頂列）與「輸出資料夾」（作品庫），按下只對父視窗送 `{ type: "miroxen:host", action: "restart" | "stop" | "output" }`，由外殼執行。「模型與硬體」「API」分頁交給外殼顯示：模型清單改成右上角模型按鈕的彈出面板。
- `theme=light|dark`：強制明暗；省略時跟隨系統。
- `c-<name>=<CSS 顏色>`：覆寫顏色，值需 URL 編碼且通過 `CSS.supports("color", value)`；名稱只接受 `bg`、`panel`、`surface`、`surface-2`、`ink`、`ink-muted`、`line`、`accent`、`accent-ink`、`danger`、`warn`、`ok`。
- `#key=<金鑰>`：只接受 16～128 個英數字、`_`、`-`；第一個 API 請求前套用，再從網址移除 hash。金鑰僅留在頁面記憶體，重新載入時由 AxonDeck 再提供。

內嵌時父視窗可呼叫 `iframe.contentWindow.postMessage({ type: "miroxen:theme", theme: "light", colors: { accent: "#ff0000" } }, "http://127.0.0.1:<埠>")` 即時換主題。工作台只接收 `window.parent` 的訊息；主題與顏色沿用網址的驗證規則，未提供或不合法的值不套用。

## 開發

| 目錄 | 內容 |
|---|---|
| `src/` | C++／CUDA 引擎：`core/`（張量、權重串流、量化）、`kernels/`、`models/`、`text/`（分詞與文字編碼）、`server/` |
| `web/` | 原生 HTML／CSS／JavaScript 工作台，免建置 |
| `tests/` | 單元、數值比對與瀏覽器檢查；每個 `tests/*_test.{cpp,cu}` 自動成為一個測試 |
| `tools/` | 產生數值比對資料的 Python 工具與效能量測腳本，推論時不需要 |
| `third_party/` | cpp-httplib、nlohmann/json、stb_image |
| `models/` | 模型、LoRA 與比對資料（不提交） |
| `out/` | 生成作品、QA 與量測紀錄（不提交）；服務輸出目錄可用 `--out` 指定 |

外部依賴只有 CUDA Toolkit（`cudart`、`cuBLAS`、`cuBLASLt`）與上述單檔標頭庫。

```bat
ctest --test-dir build --output-on-failure
python tools/perf/pipeline_bench.py --base http://127.0.0.1:8188 --out out/perf/benchmark.json
```

部分測試需要對應的模型或參考資料，缺檔時會略過。瀏覽器檢查：

```bat
npx --package @playwright/cli playwright-cli -s=miroxen-ui open http://127.0.0.1:8188 --browser msedge
npx --package @playwright/cli playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js
npx --package @playwright/cli playwright-cli -s=miroxen-ui close
```

文字編碼器會快取最近兩筆輸入（最多 128 MiB），相同提示詞與參考圖不重算；設 `MX_DISABLE_ENCODER_CACHE=1` 可關閉。API 工作結果的 `timings` 列出各階段秒數。

研究筆記：[docs/RESEARCH.md](docs/RESEARCH.md)；進度與實測紀錄：[tasks/todo.md](tasks/todo.md)。
尚未支援 H3-Regenerate-2K 與 Qwen 提示詞改寫模型。[Open-VC Attention](https://www.machgen.ai/blog/open-vc-attn) 只支援 B200／B300，無法用在消費級顯示卡，評估見 [RESEARCH.md §5](docs/RESEARCH.md#5-open-vc-attention-能否套用2026-10-11)。
