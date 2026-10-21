# AGENTS.md

Miroxen 是專為 Qwen-Image-2.1（文生圖／編輯）與 MiniMax-H3（影音生成）寫的 C++/CUDA 推論引擎，
附 HTTP API 與簡易操作面板。研究筆記見 `docs/RESEARCH.md`，進度見 `tasks/todo.md`。

## 建置與測試

- Windows：`build.bat [build目錄]`（呼叫 vcvars64 + CMake + Ninja；CUDA 13.3，預設架構 sm_120a;80;86;89;90，`a` 為 sm_120 專屬指令）。
- 測試：`ctest --test-dir build --output-on-failure`；每個 `tests/*_test.{cpp,cu}` 自動成為一個測試。
- `src/` 下新增 `.cpp/.cu` 不需改 CMakeLists（GLOB）。

## 規則

- 核心型別在 `src/core/`：`GType`（沿用 ggml 編號）、`DevBuf`/`PinnedBuf`/`Arena`、`TensorStore`。
- 張量 shape 一律 PyTorch 慣例（最外層在前）；線性層權重為 `[out, in]`。
- 錯誤用 `MX_CHECK` / `MX_CUDA` 丟例外；不要靜默吞錯。
- 大檔（模型權重）放 `models/`（已 gitignore），不要提交。
- 文件用繁體中文；程式識別字用英文。
