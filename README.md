# Miroxen

針對 **Qwen-Image-2.1** 圖片生成／編輯與 **MiniMax-H3** 影音生成的 C++／CUDA 推論引擎。
提供命令列、HTTP API 與瀏覽器工作台；推論不依賴 ComfyUI 或 Python。

本機配置：RTX 5060 Ti 16 GB＋RTX 3060 Ti 8 GB、Ryzen 7 5700X、96 GB RAM。
圖片預設使用 **abenzerps 的 UC-Q8_0 GGUF**，影片使用 unsloth 的 FL2VA／Ref2VA Q4_K。
模型量化與 `--compute` 的運算精度是兩個獨立設定；目前運算預設為 FP8。

## 建置與啟動

Windows 需要 Visual Studio 2022 C++ 工具、CUDA 13.3、CMake ≥ 3.28、Ninja。
影片功能另需 `ffmpeg`、`ffprobe` 在 PATH。

```bat
build.bat
build\miroxen.exe serve --port 8188
```

[開啟本機工作台](http://127.0.0.1:8188)。服務預設綁定 `0.0.0.0`，可從區域網路存取。
`build.bat` 會設定 UTF-8 字碼頁，讓中文版 MSVC 的標頭依賴能正確被 Ninja 辨識。

- **創作**：圖片／影音、參考素材、LoRA、模型檔／量化版本、運算精度與取樣設定。
- **頂部模型選單**：連線狀態旁顯示目前模型，點開可載入、切換或卸載。只保留一顆模型；生成中不能管理模型。
- **API／硬體**：左右兩欄顯示可複製網址、區網使用說明，以及 CPU／RAM／雙 GPU 的用量、溫度與功耗。窄螢幕改為上下排列。

詳細 API 契約：[docs/API.md](docs/API.md)。

## 其他區網電腦怎麼用

1. 兩台電腦連到同一網路，這台電腦保持服務執行。
2. 在 API／硬體頁找與對方同網段的網址；另一台電腦開「工作台」連結即可生成，無需安裝模型。
3. 支援圖片生成 API 的程式，Base URL 填 `http://<這台電腦的區網 IP>:8188/v1`；圖片模型 ID 為 `qwen-image-2.1`。
4. 有設定 `--api-key` 時，客戶端需提供同一金鑰：`Authorization: Bearer <API key>`。

圖片使用 `POST /v1/images/generations`；圖片／影音非同步工作使用 `POST /v1/jobs`。
本服務沒有聊天完成端點，不能當成一般聊天模型。對方不要填 `127.0.0.1` 或 `localhost`。
VPN 網址需要雙方加入同一 VPN；Windows 防火牆需允許 TCP 8188 入站，設定需要管理員權限。
可用 `http://<IP>:8188/health` 確認是否連通。

若只要本機存取：

```bat
build\miroxen.exe serve --host 127.0.0.1 --port 8188
```

## 模型、編碼器與 VAE 來源

以下是這台電腦的已安裝檔案；粗體為預設。來源已對照下載紀錄與 Hub 檔案雜湊。
模型權重不提交 Git，各模型／LoRA 依來源頁面的授權使用。

### 圖片：`models/image/qwen-image-2.1/`

原始架構與模型：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。

| 元件 | 檔案 | 來源 |
|---|---|---|
| **DiT · Q8_0** | **`qwen-image-2.1-UC-Q8_0.gguf`** | [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) |
| **文字編碼器 · Q4_K_M**／備選 Q8_0 | `Qwen3VL-8B-Instruct-Q4_K_M.gguf`、`Qwen3VL-8B-Instruct-Q8_0.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| 視覺編碼器 · F16，編輯時載入 | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| **紋理修正 VAE · BF16** | **`vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors`** | [madebyollin/texture-fix-vae-for-qwen-image-2.1](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/tree/1f80456b6de9d89043c133ff1f3875449b8bd016) |
| 官方 VAE · BF16 | `vae/qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| Tokenizer／設定／數值比對用原始 VAE | `hf/processor/`、`hf/vae/` 等 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |

UC-Q8_0：7,591,557,920 bytes，來源 revision `6b34e59458d3eb7ba6a6f86a116aed5253dc02c3`，
SHA-256：`cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8`。
來源倉庫同時有一般版與 `UC` 版，這裡選的是 `UC` 檔案。名稱與行為說明依發布者模型卡；本專案未量測其內容限制或與原版的畫質差異。

目前紋理修正 VAE 固定在表中的舊 revision，SHA-256 為
`05e9af8da4697d1a5118b674c3d90e95f932702d2c5a8900d7d4f05076403cc7`，與作者目前最新檔案不同。
只有解碼器微調，編碼器與原版相同。未安裝它時自動退回官方 VAE。
圖片 DiT 只保留 UC-Q8_0 作為預設與面板選項；官方文字編碼器、視覺塔與 VAE 仍是必要元件。
三個原版 DiT 已移至 `out/trash/original-image-dit/`，不再載入；來源為
[leejet](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) 與 [unsloth](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)。
刪除受到自動安全審查阻擋，檔案仍占空間。原版 DiT 的數值比對測試會明確標示 Skipped，不能用 UC 權重冒充原版比對。

下載新預設：

```bat
hf download abenzerps/Qwen-Image-2.1-Uncensored-GGUF qwen-image-2.1-UC-Q8_0.gguf --revision 6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 --local-dir models/image/qwen-image-2.1
```

### 影音：`models/video/minimax-h3/`

原始模型：[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)。

| 元件 | 檔案 | 來源 |
|---|---|---|
| **文字／首尾幀 DiT · Q4_K** | **`minimax_h3_fl2va_pruned-Q4_K.gguf`** | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| **參考生成 DiT · Q4_K** | **`minimax_h3_ref2va_pruned-Q4_K.gguf`** | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| **文字／視覺編碼器 · Q4_K_M** | **`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`** | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| **影片 VAE · FP16** | **`vae/minimax_h3_video_vae_fp16.safetensors`** | [unsloth 配套檔](https://huggingface.co/unsloth/MiniMax-H3-GGUF/tree/d629413c2e5b51b38c453668b75ca3b06ca92703/vae)；[Comfy-Org 原始轉換檔](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| **音訊 VAE · FP32** | **`vae/minimax_h3_audio_vae_fp32.safetensors`** | [unsloth 配套檔](https://huggingface.co/unsloth/MiniMax-H3-GGUF/tree/d629413c2e5b51b38c453668b75ca3b06ca92703/vae)；[Comfy-Org 原始轉換檔](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Tokenizer／設定／數值比對用原始 VAE | `hf/tokenizer/`、`hf/vae/`、`hf/audio_vae/` 等 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |

FL2VA 與 Ref2VA 是不同 checkpoint，給參考素材時會自動換用 Ref2VA。
參考圖片最多 9 張、影片 3 段、音訊 3 段，共最多 12 個；音訊不能單獨作為參考。
影片輸出固定 24 fps，幀數向上對齊 `17n+5`、最少 22 幀。
例如目標 10 秒填 240 幀，實際生成 243 幀（10.125 秒）；高解析度長影片仍可能超出 16 GB 顯存。

## LoRA 來源與排程

圖片 LoRA 放在 `models/loras/image/`，影片 LoRA 放在 `models/loras/video/`。
API 與 CLI 仍可直接使用原檔名；不必在 API 名稱填子目錄。舊平面 LoRA 目錄也可讀取。
畫風 LoRA 不等於加速 LoRA；只加畫風時仍用原版步數。

| 模型／用途 | 檔案 | 來源 | Miroxen 設定 |
|---|---|---|---|
| 圖片 · 6 步加速 | `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors` | [Viggle](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | `sigmas: [1, .9375, .875, .75, .5, .25]`、`shift_terminal: false`、CFG 1 |
| 圖片 · 8 步加速 | `turbo8_lora_step2500.safetensors` | [chriswritescode/Turbo8-LoRA-Qwen-Image-2.1](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1) | `steps: 8`、`shift_terminal: false`、CFG 1 |
| 圖片 · ZZZ Chibi 畫風 | `zzz_chibi_v2_st2500.safetensors` | [wesjos/Qwen-Image2.1-ZenlessZoneZeroChibi-Lora](https://huggingface.co/wesjos/Qwen-Image2.1-ZenlessZoneZeroChibi-Lora) | 依畫風需求調強度，不降低原版步數 |
| 影音 · FL2VA 8 步加速 | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 9`、`video_shift: 6` |
| 影音 · FL2VA 4 步加速 | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 5`、`video_shift: 6` |
| 影音 · Ref2VA 8 步加速 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `steps: 9`、`video_shift: 12` |

面板勾選加速 LoRA 會套用設定；取消後回復原版排程。互不相容的 FL2VA／Ref2VA LoRA 分開列出。
支援 diffusers／PEFT／kohya 命名。FP8／FP4 路徑會併入權重，H3 在逐層搬運時合併；BF16 路徑每步即時計算。
新預設已實測文生圖、編輯、Turbo8 套用及移除後還原；其他 LoRA 組合與完整畫質比較尚未完成。

### 不用加速 LoRA 時的步數

| 模型 | 建議起點 | 說明 |
|---|---|---|
| Qwen-Image-2.1：生成／編輯／透明圖 | **40 步** | [官方範例](https://huggingface.co/Qwen/Qwen-Image-2.1#quick-start)；原版排程 `shift_terminal: true`，清空自訂 sigma |
| MiniMax-H3：FL2VA／Ref2VA | **30 個取樣點** | Miroxen 目前預設，實際 **29 次 DiT 計算**；影片 shift 12 |

這是兩顆模型的起點，不是所有擴散模型的通用規則。較少步數可作預覽，但細節與動作穩定度需逐張／逐段確認。
H3 的 `steps` 包含最後的零點，因此「8 次計算」在本 API 填 9；[上游排程實作](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_minimax_h3.py)亦區分取樣點與計算次數。
H3 的 30 是本專案採用的基準，沒有把它說成官方保證的最佳步數。增加步數也不保證每張圖／每段影片都更好。

## 命令列範例

```bat
build\miroxen.exe image -p "一隻在雪地裡的紅狐狸" -W 1024 -H 1024 --steps 40
build\miroxen.exe image -p "把背景換成夜晚的城市" --ref 原圖.png --steps 40
build\miroxen.exe video -p "A red panda eating bamboo" -W 864 -H 480 --frames 56 --steps 30
build\miroxen.exe image -p "A red panda" --lora turbo8_lora_step2500.safetensors:1 --steps 8 --shift-terminal 0
```

圖片可加入最多 10 張參考圖。未指定尺寸時依最後一張參考圖比例，約 1024² 像素。
透明圖／去背使用模型的 RGBA 能力；局部修改靠模型理解遮罩，不能保證遮罩外像素完全不變。
`--compute bf16|fp8|fp4` 控制運算精度；FP8／FP4 需要 Blackwell 路徑，不支援時退回 BF16。
本機分工為 GPU0 執行主要 DiT、GPU1 分擔文字編碼；實際配置以面板模型資訊為準。

## 目錄與驗證

圖片與影片共用的文字編碼器會保留最近兩筆完整圖文輸入及編碼結果，最多 128 MiB；相同輸入可省去重算。
比對包含圖片特徵內容，換提示詞或參考內容會重新計算；切換／卸載模型會清除快取。視覺塔及參考素材 VAE 本身仍會執行。
設 `MX_DISABLE_ENCODER_CACHE=1` 可關閉快取做對照。快取不改步數、量化或 LoRA；圖片 latent 的排列與正規化改在 GPU 執行，省掉 CPU 往返。
API 工作結果的 `timings` 列出各階段秒數，欄位定義見 [API 文件](docs/API.md)。

服務啟動後可重跑實測（會生成圖片與短片，模型須已安裝）：

```bat
python tools/perf/pipeline_bench.py --base http://127.0.0.1:8188 --out out/perf/benchmark.json
```

| 目錄 | 內容 |
|---|---|
| `src/` | C++／CUDA 引擎、模型管線、HTTP 服務 |
| `web/` | 原生 HTML／CSS／JavaScript 工作台 |
| `tests/` | 單元／數值比對／瀏覽器檢查 |
| `tools/` | 產生數值比對資料的 Python 工具，推論時不需要 |
| `third_party/` | 既有第三方標頭 |
| `models/image/`、`models/video/` | 分開存放圖片／影音模型、編碼器、VAE 與 tokenizer |
| `models/loras/image/`、`models/loras/video/` | 圖片／影音 LoRA；API 遇到同名檔案會拒絕載入，避免選錯 |
| `models/ref/` | 數值比對資料，保留供測試 |
| `build/` | CMake／Ninja 產物，保留可執行服務與增量建置 |
| `out/images/`、`out/videos/` | 原本零散的圖片、影片與配套音訊 |
| `out/generated/` | 目前儀表板新生成的作品，服務輸出目錄可用 `--out` 指定 |
| `out/qa/`、`out/perf/` | QA 截圖／腳本／紀錄，以及按日期整理的效能量測與作品 |
| `out/reports/` | 整理紀錄；`organization-20261007.json` 記錄每個舊路徑與新位置，歷史任務紀錄可據此查找 |
| `out/trash/` | 確認無用的隔離暫存及原版圖片 DiT；安全審查拒絕刪除，尚未釋放容量 |
| `docs/`、`tasks/` | API／研究文件與進度、已知限制 |

```bat
ctest --test-dir build -R "weights_concat_test|qwen_dit_parity_test|server_test|model_file_test|h3_frames_test" --output-on-failure
ctest --test-dir build --output-on-failure
```

完整測試含 GPU／模型／參考資料檢查，需已安裝對應檔案。
瀏覽器檢查（本機使用 Edge，另開獨立測試環境）：

```bat
npx --package @playwright/cli playwright-cli -s=miroxen-ui open http://127.0.0.1:8188 --browser msedge
npx --package @playwright/cli playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js
npx --package @playwright/cli playwright-cli -s=miroxen-ui close
```
研究：[docs/RESEARCH.md](docs/RESEARCH.md)；進度與實測紀錄：[tasks/todo.md](tasks/todo.md)。
H3-Regenerate-2K、Qwen 提示詞改寫模型仍未支援；本專案沒有宣稱全球最快。
