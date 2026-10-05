# 教訓

## CUDA 串流同步（2026-10-05）
- 症狀：H3 影片有殘留雜訊；第 0 步與參考完全吻合，第 1 步起差異巨大；輸出 std 偏低。
- 根因：Ctx 用 `cudaStreamNonBlocking` 串流，而讀回結果用 `cudaMemcpy`（預設串流），兩者不同步，
  最後讀到的是少算一步的 latent；除錯存檔也因此錯位，讓比對數據看起來像模型算錯。
- 規則：計算串流一律用一般（阻塞）串流；只有用 event 明確同步的輔助串流（權重串流的 copy stream）才用非阻塞。
- 除錯方法：「第 0 步對、之後錯」先懷疑同步／狀態，不要先懷疑數學。教師強制比對（把引擎第 i 步的輸入餵給參考）能分離誤差來源。

## 驗證標準
- bf16 參考經 50 層會放大誤差到 5–20%，不適合當裁判；用 fp32 計算（權重 bf16）的參考。
- 擴散取樣的逐像素差異會被多步放大，以「同雜訊下 latent 統計 + 解碼畫面」判斷，不只看相對誤差。

## Windows 開發環境
- Write 工具會寫 CRLF；倉庫已用 .gitattributes 強制 LF。Bash heredoc 裡的 `\n` 會被轉成真的換行，
  需要字面反斜線時用 `chr(92)` 或直接用 Edit 工具。
- CUDA 預設把 SM 多的 3060 Ti 排成 device 0；程式啟動時強制 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。

## 區塊縮放 GEMM（FP8／FP4，2026-10-05）
- cuBLASLt 的 `VEC32_UE8M0`（MXFP8）與 `VEC16_UE4M3`（NVFP4）在 sm_120 可用；`OUTER_VEC`、128 區塊縮放不行。
- 縮放張量要照 128×4 交錯排列，而且補齊的欄／列 cuBLASLt 也會讀：沒清零時 0xff 在 ue8m0 是 NaN（K 不是 128 倍數時才出現）。
- `__nv_cvt_float2_to_fp4x2` 的硬體指令只在 `sm_120a` 這類架構專屬目標才會產生，通用 sm_120 會退回軟體模擬（慢 2 倍以上）。
- 驗證排列：把量化結果在主機端照格式解回 f32 做 GEMM 比對，排列錯會直接對不上（tests/blockscale_test.cpp）。

## nvcc 與中文字串
- `.cu` 檔的 `MX_CHECK` 訊息含全形字元會被 nvcc 前端讀壞（missing closing quote），`.cu` 一律用英文訊息；
  測試若要中文訊息就寫成 `.cpp`。
