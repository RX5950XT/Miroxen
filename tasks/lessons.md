# 教訓

## 最佳化目標（2026-10-08）
- 以硬體利用、速度與模型表現的 CP 值評估，不把「最快」稱為「最優」。優先移除多餘計算／搬運；降低步數或精度須明講畫質取捨，微型測試加速不能當成完整生成加速。

## 單卡顯存與 WDDM（2026-10-11）
- Windows WDDM 顯存超用不會 OOM，而是溢出到系統記憶體、速度掉 10～20 倍（8 GB 卡影片取樣 47 秒／步、解碼 262 秒）。「單卡慢得離譜」先查 cudaMemGetInfo 與常駐元件總和，不要先懷疑核心。
- 單卡配置要主動卸載不在用的元件（VAE、視覺塔、暫存），並在 log 印出各階段可用顯存。

## LoRA 操作入口（2026-10-08）
- 加速 LoRA 的強度固定為作者訓練設定 1，只提供步數／模式選擇；強度滑桿只供已識別的畫風 LoRA。切換加速 LoRA 時同步套用對應步數與排程，不能留下上一顆的混合模式。

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
- `__nv_cvt_float2_to_fp4x2` 的硬體指令只在 `sm_120a` 這類架構專屬目標才會產生，通用 sm_120 會退回軟體模擬（慢 2 倍以上）；blockscale.cu 改用門檻比較編碼，兩種目標都快。
- 驗證排列：把量化結果在主機端照格式解回 f32 做 GEMM 比對，排列錯會直接對不上（tests/blockscale_test.cpp）。

## nvcc 與中文字串
- `.cu` 檔的 `MX_CHECK` 訊息含全形字元會被 nvcc 前端讀壞（missing closing quote），`.cu` 一律用英文訊息；
  測試若要中文訊息就寫成 `.cpp`。

## GeForce Blackwell tensor core 實測（5060 Ti，2026-10-05）
- `mma.sync` 速度：bf16（fp32 累加）44、FP8 e4m3（fp32 累加）90、INT8 180、`kind::mxf8f6f4.block_scale` FP8 180 TFLOPS。
  一般 FP8 MMA 只有半速；要全速就用區塊縮放版本並把縮放設成 2⁰（ue8m0 0x7f）。需要 `sm_120a` 編譯，專案已改成 `120a;86`。
- 自寫小測試量指令速度時記得選對卡：CUDA 預設 device 0 是 3060 Ti（見上），在 sm_86 上跑 sm_120 cubin 會報 no kernel image。
- FP8 注意力（SageAttention2 式）第一版 1 CTA/SM 只到 67 TFLOPS；換全速 MMA 到 82，`__launch_bounds__(256, 2)` 壓到 128 暫存器後 105。
  改 exp／int→float 轉換沒有幫助，瓶頸是 MMA 速率與佔用率。
- 後來發現 256 執行緒×2 CTA 壓到 128 暫存器時 D=128 會 spill；改 128 執行緒×3 CTA（168 暫存器）不 spill，再快 2–7%。
  調 `__launch_bounds__` 後用 `-Xptxas -v` 看暫存器與 spill。

## 建置
- 改了 class 成員或 struct（例如 BlockStreamer、ImageRequest）後出現「invalid resource handle」、0xc0000409、日誌數字
  每次不同這類記憶體被寫壞的症狀，根因是 ninja 沒記到標頭依賴：MSVC 的 `/showIncludes` 前綴是中文，編碼跟著主控台
  字碼頁走（PowerShell 7 是 UTF-8、git-bash 下的 cmd 是 950），和 CMake 設定時偵測到的不同就整個不記。
  build.bat 已固定 `chcp 65001`；檢查方式 `ninja -C build -t deps | grep -c "#deps 0"` 應為 0。（`VSLANG=1033` 沒效，沒裝英文語言包。）

## 主機記憶體
- 串流權重不要在主機留整份鎖頁副本（H3 曾鎖 30+ GB 導致系統記憶體不足被砍）：直接從 mmap 讀到兩塊鎖頁暫存即可，
  CPU 複製與 H2D 都藏在每層計算後面；量化放到 GPU copy stream 上做。

## LoRA 格式（2026-10-05）
- LightX2V 的 MiniMax-H3 加速 LoRA：key 為 diffusers 命名、中間帶 PEFT adapter 名（`x.lora_A.default.weight`），
  沒有逐層 alpha，縮放要用 safetensors metadata 的 `alpha`（8，rank 128 → 1/16）；用 1.0 會直接變雜訊。
- diffusers 的 `ff.net.0.proj` 是 [value; gate]，原始 checkpoint（GGUF）的 `mlp.fc1` 是 [gate; value]：LoRA 的 B 要前後對調。
  官方實作確認：diffusers `convert_minimax_h3_gated_ff`、LightX2V `value, gate = in_proj(...).chunk(2)`。
- MiniMax-H3 的 `steps` 是取樣點數，前向次數 = steps − 1；「8 步」加速 LoRA 用 `--steps 9 --shift 6`（768p 版）。

## 數值比對的門檻
- 和 fp32 參考差幾 % 時先拿官方 bf16 跑一次對照，再判斷是不是 bug：Qwen3-VL 視覺塔官方 bf16 對 fp32 本身就差 5.5%，
  圖片列有超大激活值通道（1838、2276）差到 15%。我們的數字和官方 bf16 同量級就是精度，不是錯。

## 效能量測（2026-10-05）
- 量速度前先 `nvidia-smi --query-compute-apps=pid,process_name --format=csv` 看 GPU0 有沒有遊戲、動態桌布、Broadcast 在跑。
  有的話只能做同時段「改前／改後交錯」A/B，數字不能和舊表比（H3 fp8 從 110 變 129 秒就是這樣，不是退步）。
- 沒有系統管理員權限也能量冷快取：`robocopy <來源> <目的> <檔名> /J` 以不經快取的 I/O 複製，新檔內容不在系統快取。
- 宣稱「輸出位元相同」前，先拿舊版同 seed 跑兩次：fp8 出圖本來就不是決定性的（平均每通道差約 0.8）。
- 子代理的優化：先看收益對複雜度。收益 <1% 或 <1 秒卻要幾百行的不合併，改從它的量測裡挑最小改法
  （LoRA 載入：Grok 240 行省 0.75 秒；只改 bf16 直接上傳 1 行就省 0.5 秒）。

## 儀表板（2026-10-06）
- 全域模型狀態與管理放在頂部連線狀態旁，精簡顯示、點開操作；API 入口同時提供可複製網址及其他區網電腦的設定說明。

## GGUF 模型替換（2026-10-06）
- 同架構的 GGUF 不保證張量名稱與融合方式相同；切換來源後須實際載入，對照 gate／proj 的順序、形狀與量化型別。
- 載入時接合權重也要保留原始來源，讓量化 LoRA 套用與移除後還原都可重建；只測無 LoRA 出圖不夠。
