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
| fp8（預設） | 152 TFLOPS | 18.7 秒 | 142 秒 | 幾乎相同（圖 33.5 dB） |
| fp4 | 293 TFLOPS | 16.7 秒 | 108 秒 | 乾淨，但構圖與提示詞遵循度會變 |
