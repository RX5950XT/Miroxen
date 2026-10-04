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
