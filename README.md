# Miroxen

專為 **Qwen-Image-2.1**（文生圖／編輯）與 **MiniMax-H3**（文字／首尾幀／參考素材 → 影片 + 立體聲）
重新撰寫的 C++/CUDA 推論引擎，支援 GGUF 量化權重與社群 LoRA，提供 HTTP API 與簡易操作面板。

> 開發中。研究筆記：[docs/RESEARCH.md](docs/RESEARCH.md)；進度：[tasks/todo.md](tasks/todo.md)。

## 建置（Windows）

需求：Visual Studio 2022、CUDA 13.x、CMake ≥ 3.28、Ninja。

```bat
build.bat
ctest --test-dir build --output-on-failure
```

`build.bat` 會先切到 UTF-8 字碼頁（`chcp 65001`）。不這樣做的話，ninja 看不懂中文版 MSVC 吐出的標頭依賴訊息，
改了標頭卻不會重編，所以請一律用 `build.bat` 建置。

## 使用

```bat
build\miroxen.exe image -p "一隻在雪地裡的紅狐狸" -W 1024 -H 1024 --steps 20
build\miroxen.exe video -p "A red panda eating bamboo" -W 864 -H 480 --frames 56 --steps 30
build\miroxen.exe serve --port 8188   :: 開瀏覽器到 http://127.0.0.1:8188 使用操作面板
```

模型放在 `models/qwen-image-2.1/`、`models/minimax-h3/`，任何 GGUF 量化（Q2_K～Q8_0、IQ4、MXFP4、NVFP4）都能讀，
可用 `--qi-dit` / `--h3-dit` 指定檔案。影片功能需要 `ffmpeg`、`ffprobe` 在 PATH。API 說明見 [docs/API.md](docs/API.md)。

## 影像編輯（Qwen-Image-2.1）

給參考圖就變成編輯模式，最多 10 張（命令列 `--ref` 可重複，API 用 `images` 陣列）：

```bat
build\miroxen.exe image -p "把背景換成夜晚的城市" --ref 原圖.png --steps 30
```

- 視覺塔用 `models/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf`，第一次編輯時才載入。
- 不給 `-W`／`-H`（API：`"size":"auto"`）時，依最後一張參考圖的比例出圖，約 1024² 像素。
- 實測：1024² 編輯 30 步 fp8 約 28 秒（含載入）；下「不改」的指令能忠實還原原圖。
- 與 diffusers 官方逐段比對：視覺塔誤差 5.6%（官方自己 bf16 對 fp32 也有 5.5%），帶參考圖的 DiT 1.7–1.9%。

## 紋理修正 VAE（選用）

[madebyollin/texture-fix-vae-for-qwen-image-2.1](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1)
是官方 VAE 解碼器的微調版，去掉照片類細節（青苔、草地）上的棋盤格紋，可以直接替換：

```bat
build\miroxen.exe image -p "..." --qi-vae models\qwen-image-2.1\vae\texture_fix_vae_for_qwen_image_2.1_bf16.safetensors
```

實測同一組 latent，紋理明顯乾淨；作者標示 PSNR 略低（32.86 → 32.46），所以預設仍用官方 VAE。
透明圖（RGBA）也能用：同 seed 比較，主體與邊緣幾乎相同（透明度 PSNR 37 dB），而且背景更乾淨
（官方 VAE 有 66% 背景像素殘留 1–9 的透明度，紋理修正版幾乎全是 0）。

## 自訂 sigma 節點

`--sigmas 1,0.9375,0.875,0.75,0.5,0.25`（API：`sigmas` 陣列）直接指定排程節點，步數＝節點數。
給步數蒸餾 LoRA 用，例如下面的 Viggle 6 步加速。

## 影片：首尾幀與參考生成（MiniMax-H3）

**首尾幀（FL2VA）**：`--first 首幀.png --last 尾幀.png`，可以只給其中一張（API：`first_frame`／`last_frame`）。
沒給尺寸時依首幀比例用官方畫布（短邊 768）。關鍵幀會經過 32B 編碼器 GGUF 內建的視覺塔。

**參考生成（Ref2VA）**：依序給圖片、影片、音訊，提示詞裡用 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 指稱
（API：`references: [{type, data}]`）：

```bat
build\miroxen.exe video -p "<Picture 1> 的狗在海邊跑" --ref image=狗.png --ref video=片.mp4 --ref audio=聲.wav --steps 9 --shift 12 --lora minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
```

- 上限：圖 ≤ 9、影片 ≤ 3、音訊 ≤ 3，總共 ≤ 12；音訊不能單獨用；不能和首尾幀同時用。
- 影片自帶的音軌會一起參考，標成 `<Audio j>`，排在該段 `<Video k>` 前面。
- 需要 Ref2VA checkpoint `models/minimax-h3/minimax_h3_ref2va_pruned-Q4_K.gguf`
  （[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)），引擎會自動換 DiT。
- 參考圖預設縮到短邊 2048（官方），可用 `--ref-edge`（API：`ref_image_short_edge`）調小，省時間與顯存。
- 參考影片預設畫布不超過輸出畫布（快約一倍）；`--ref-native 1`（API：`ref_native_canvas: true`）改回官方 768p。
- 實測（864×480・56 幀、8 步加速 LoRA）：一張參考圖約 75 秒；一段 56 幀參考影片約 151 秒（參考編碼 53 秒、DiT 78 秒）。

**shift 與步數**：`--shift`（API：`video_shift`）是影片排程的 shift，官方 12，LightX2V 768p 加速 LoRA 建議 6。
H3 的 `steps` 是取樣點數，實際前向次數＝steps − 1，所以「8 步」加速 LoRA 要用 `--steps 9`。

## 運算精度（`--compute`）

GGUF 量化只決定權重檔大小；`--compute` 決定 DiT 實際用哪種 tensor core 計算。fp8／fp4 用 Blackwell（RTX 50 系列）
的區塊縮放矩陣乘法，其他顯卡自動退回 bf16。RTX 5060 Ti 實測：

| 精度 | 矩陣乘法 | Qwen-Image 1024²・20 步 | H3 864×480・56 幀・30 步 | 畫質 |
|---|---|---|---|---|
| bf16 | 41 TFLOPS | 35.0 秒 | 267 秒 | 基準 |
| fp8（預設） | 152 TFLOPS | 14.0 秒 | 110 秒 | 幾乎相同（圖 31.1 dB） |
| fp4 | 293 TFLOPS | 9.7 秒 | 93.5 秒 | 乾淨，但構圖與提示詞遵循度會變 |

以上是 GPU 沒有其他程式時量的；同時開遊戲、動態桌布等吃顯卡的程式，實測會慢約 10–20%。

Qwen 的 VAE 解碼改用自寫卷積核心（不展開 im2col），2048² 從 8.3 秒降到 2.4 秒。
fp8／fp4 模式下影片 VAE 解碼器的 FFN 也用 FP8（PSNR 40.8 dB）。

fp8／fp4 模式下注意力也改用 INT8（Q·K）＋ FP8（P·V）tensor core（SageAttention2 做法），比 bf16 注意力快 2.4–2.9 倍，
誤差約 0.4–1.4%。

H3 的 DiT（約 10 GiB）與 32B 文字編碼器（約 14 GiB）都是逐層直接從模型檔讀、送上 GPU 才轉格式，
不另外鎖住整份副本：整個程式只佔約 1.7 GB 記憶體，檔案內容交給作業系統快取（記憶體吃緊時可回收）。

## LoRA

LoRA 放在 `models/loras/`，命令列用 `--lora 檔名:強度`（只給檔名會到 `models/loras/` 找），API／面板用 `loras` 欄位。支援 diffusers／PEFT（`lora_A/lora_B`）
與 kohya（`lora_down/lora_up` + `alpha`）命名。fp8／fp4 模式下，換 LoRA 時會把它直接併進量化權重（1.3 GB 的 rank 256 約 3 秒），
之後出圖不會變慢；bf16 模式則每步即時計算。H3 的 fp8 串流模式則是在搬運權重時順手併入，不佔取樣時間。

Qwen-Image 實測：
- [Zenless Zone Zero Chibi](https://huggingface.co/wesjos/Qwen-Image2.1-ZenlessZoneZeroChibi-Lora)（畫風，rank 16）：192 層全部對應。
- [Turbo8](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1)（8 步加速，rank 128，含調變與時間嵌入層）：227 層全部對應。
  用法 `--steps 8 --shift-terminal 0`（API：`"steps":8,"shift_terminal":false`，不設 `cfg_scale`），1024² 約 8.4 秒。
- [Viggle turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)（6 步加速，`Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors`）：
  用法 `--sigmas 1,0.9375,0.875,0.75,0.5,0.25 --shift-terminal 0`，864×480 約 5.5 秒。

MiniMax-H3 加速 LoRA（[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)）：

| 檔案 | 用途 | 參數 |
|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | 文生影片／首尾幀 | `--steps 9 --shift 6` |
| `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | 文生影片／首尾幀 | `--steps 5` |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | 參考生成 | `--steps 9 --shift 12` |

- 縮放用檔案 metadata 的 `alpha`（8，rank 128 → 8/128）。
- diffusers 命名的 `ff.net.0.proj` 輸出是 [value; gate]，載入時自動對調成原始 checkpoint 的 [gate; value]。
- 實測 864×480・56 幀：8 步加速＋首尾幀約 55 秒（原版 30 步約 144 秒）；文生影片 8 步約 42 秒（DiT 28 秒、影片解碼 9 秒）。
