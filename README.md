# Miroxen

專為 **Qwen-Image-2.1**（文生圖／編輯）與 **MiniMax-H3**（文字／圖片 → 影片 + 立體聲）
重新撰寫的 C++/CUDA 推論引擎，支援 GGUF 量化權重與社群 LoRA，提供 HTTP API 與簡易操作面板。

> 開發中。研究筆記：[docs/RESEARCH.md](docs/RESEARCH.md)；進度：[tasks/todo.md](tasks/todo.md)。

## 建置（Windows）

需求：Visual Studio 2022、CUDA 13.x、CMake ≥ 3.28、Ninja。

```bat
build.bat
ctest --test-dir build --output-on-failure
```

## 使用

```bat
build\miroxen.exe image -p "一隻在雪地裡的紅狐狸" -W 1024 -H 1024 --steps 20
build\miroxen.exe video -p "A red panda eating bamboo" -W 864 -H 480 --frames 56 --steps 30
build\miroxen.exe serve --port 8188   :: 開瀏覽器到 http://127.0.0.1:8188 使用操作面板
```

模型放在 `models/qwen-image-2.1/`、`models/minimax-h3/`，任何 GGUF 量化（Q2_K～Q8_0、IQ4、MXFP4、NVFP4）都能讀，
可用 `--qi-dit` / `--h3-dit` 指定檔案。API 說明見 [docs/API.md](docs/API.md)。

## 運算精度（`--compute`）

GGUF 量化只決定權重檔大小；`--compute` 決定 DiT 實際用哪種 tensor core 計算。fp8／fp4 用 Blackwell（RTX 50 系列）
的區塊縮放矩陣乘法，其他顯卡自動退回 bf16。RTX 5060 Ti 實測：

| 精度 | 矩陣乘法 | Qwen-Image 1024²・20 步 | H3 864×480・56 幀・30 步 | 畫質 |
|---|---|---|---|---|
| bf16 | 41 TFLOPS | 40.4 秒 | 267 秒 | 基準 |
| fp8（預設） | 152 TFLOPS | 16.5 秒 | 110 秒 | 幾乎相同（圖 31.1 dB） |
| fp4 | 293 TFLOPS | 11.9 秒 | 93.5 秒 | 乾淨，但構圖與提示詞遵循度會變 |

fp8／fp4 模式下注意力也改用 INT8（Q·K）＋ FP8（P·V）tensor core（SageAttention2 做法），比 bf16 注意力快 2.4–2.9 倍，
誤差約 0.4–1.4%。

H3 的 DiT（約 10 GiB）與 32B 文字編碼器（約 14 GiB）都是逐層直接從模型檔讀、送上 GPU 才轉格式，
不另外鎖住整份副本：整個程式只佔約 1.7 GB 記憶體，檔案內容交給作業系統快取（記憶體吃緊時可回收）。

## LoRA

LoRA 放在 `models/loras/`，命令列用 `--lora 檔名:強度`，API／面板用 `loras` 欄位。支援 diffusers／PEFT（`lora_A/lora_B`）
與 kohya（`lora_down/lora_up` + `alpha`）命名。fp8／fp4 模式下，換 LoRA 時會把它直接併進量化權重（約 2–3 秒），
之後出圖不會變慢；bf16 模式則每步即時計算。

實測：
- [Zenless Zone Zero Chibi](https://huggingface.co/wesjos/Qwen-Image2.1-ZenlessZoneZeroChibi-Lora)（畫風，rank 16）：192 層全部對應。
- [Turbo8](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1)（8 步加速，rank 128，含調變與時間嵌入層）：227 層全部對應。
  用法 `--steps 8 --shift-terminal 0`（API：`"steps":8,"shift_terminal":false`，不設 `cfg_scale`），1024² 約 8.4 秒。
