# Miroxen 任務清單

## 計畫

- [x] 研究兩個模型、類似專案、本機硬體 → `docs/RESEARCH.md`
- [x] 存放庫骨架：CMake + CUDA（sm_120;86）、核心型別、介面標頭
- [x] 核心元件（平行分工）
  - [x] GGUF / safetensors 讀取 + GPU 反量化（codex）
  - [x] FlashAttention bf16（codex）
  - [x] Qwen2 BPE tokenizer（grok）
  - [x] HTTP API + 操作面板（grok，先接假引擎）
  - [x] 逐元素核心、GEMM、線性層（含 LoRA）、權重串流（主代理）
- [x] Qwen-Image-2.1
  - [x] Qwen3-VL 文字編碼器（純文字路徑）
  - [x] DiT + prefix KV cache
  - [x] VAE 解碼器
  - [x] 排程 + 端到端出圖；與 diffusers 做數值比對
  - [x] 編輯（視覺塔 + 參考圖）
  - [x] 自訂 sigma 節點（Viggle 6 步加速 LoRA）
- [x] MiniMax-H3
  - [x] 32B 文字編碼器（共用 Qwen3-VL 程式碼）
  - [x] DiT（AdaLN 預算 / pruned 表）
  - [x] 影片 VAE 解碼（ViT decoder）+ 音訊 VAE 解碼 + ffmpeg 封裝
  - [x] 端到端 T2VA；FL2VA
  - [x] Ref2VA 參考生成（圖片／影片／音訊）
  - [x] LightX2V 加速 LoRA（串流模式併入權重）
- [x] LoRA（diffusers / ComfyUI 命名；實測 ZZZ Chibi、Turbo8，fp8/fp4 下併入權重）
- [ ] 效能：與 sd.cpp 對照、雙卡分工
- [x] FP8／FP4 區塊縮放運算路徑（Qwen DiT 常駐、H3 串流）

## 狀態（2026-10-05）

- 完成並驗證：GGUF/safetensors、反量化、FlashAttention、tokenizer、Qwen3-VL 編碼器（對 transformers 1.5%）、
  Qwen-Image DiT（對 diffusers 2.4%）、Qwen-Image VAE（41 dB）、Qwen-Image 端到端出圖（含 RGBA）、HTTP API + 面板、
  H3 DiT（對 fp32 參考 2.0%/3.7%，版面座標完全一致）、H3 音訊 VAE（95 dB）、H3 影片 VAE（51.5 dB）、逐層權重串流。
- **已修正（2026-10-05）**：H3 殘留雜訊。根因是計算串流用了非阻塞串流，`cudaMemcpy` 讀回時最後一步還沒算完。
  修正後 864×480／56 幀／30 步輸出乾淨，教師強制比對每步與官方 fp32 參考差 0.4–1.7%（見 tasks/lessons.md）。
  另修 ffmpeg `-shortest` 因音訊略短而裁掉最後一幀。
- **Q4／低精度運算（2026-10-05）**：社群 Qwen-Image Q4_K（leejet）、Q4_K_M（unsloth）實測可用；反量化新增 GGUF MXFP4／NVFP4。
  DiT 新增 `--compute bf16|fp8|fp4`（cuBLASLt 區塊縮放 GEMM，MXFP8 152、NVFP4 293 TFLOPS），預設 fp8；H3 以 FP4/FP8 格式逐層串流。
  Qwen 1024²・20 步 40.4 → 18.7（fp8）／16.7 秒（fp4）；H3 864×480・56 幀 267 → 142／108 秒。API 與面板可逐請求選精度。
  INT8 GEMM／INT8 注意力的早期試作由區塊縮放 GEMM 與 FP8 注意力取代（試作分支已刪除）。
- **FP8 注意力（2026-10-05）**：INT8 Q·K + FP8 P·V（sm_120a 區塊縮放 MMA），16k token 105 TFLOPS（bf16 37），fp8/fp4 模式自動使用。
  Qwen 1024²・20 步 fp8 18.7 → 16.5 秒、fp4 16.7 → 11.9 秒；H3 864×480・56 幀 fp8 142 → 110 秒、fp4 108 → 93.5 秒。
- **串流不再鎖整份權重（2026-10-05）**：BlockStreamer 直接從 mmap 讀到 2 塊鎖頁暫存，量化在 GPU copy stream 上做；
  H3 程式記憶體約 30+ GB → 1.7 GB，畫質不變（對 bf16 20.3 dB，原 20.0）。
- **影像編輯（2026-10-05）**：最多 10 張參考圖，視覺塔 `mmproj-Qwen3VL-8B-Instruct-F16.gguf` 第一次編輯才載入；
  `size: "auto"` 依最後一張參考圖比例、約 1024² 像素。1024² 編輯 30 步 fp8 約 28 秒（含載入），「不改」指令能忠實還原原圖。
  對 diffusers：視覺塔 5.6%（官方 bf16 對 fp32 也是 5.5%）、帶參考圖的 DiT 1.7–1.9%。
  另加自訂 sigma 節點（`--sigmas`／API `sigmas`）：Viggle 6 步加速 LoRA 864×480 約 5.5 秒。
- **FL2VA 首尾幀（2026-10-05）**：`--first`／`--last`（可只給一張），沒給尺寸時依首幀比例用官方畫布（短邊 768）；
  關鍵幀經過 32B 編碼器 GGUF 內建的視覺塔。
- **Ref2VA 參考生成（2026-10-05）**：`--ref image=／video=／audio=`，圖 ≤9、影片 ≤3、音訊 ≤3、共 ≤12，音訊不能單獨，
  不能和首尾幀同用；影片自帶音軌一起參考。自動換 Ref2VA checkpoint（unsloth Q4_K）。參考圖預設短邊 2048（可調小），
  參考影片預設畫布不超過輸出畫布（快約一倍，可改回官方 768p）。864×480・56 幀、8 步加速 LoRA：一張參考圖約 75 秒；
  一段 56 幀參考影片約 151 秒（參考編碼 53 秒、DiT 78 秒）。
- **H3 加速 LoRA（2026-10-05）**：LightX2V fl2v 8 步（`--steps 9 --shift 6`）、4 步（`--steps 5`）、ref2v 8 步（`--steps 9 --shift 12`）。
  縮放用 metadata alpha（8/128）；diffusers 的 `ff.net.0.proj` 自動對調成 [gate; value]；fp8 串流時搬運權重順手併入。
  新增 `--shift`／API `video_shift`；H3 的 `steps` 是取樣點數（前向次數＝steps−1）。
  864×480・56 幀：8 步＋首尾幀約 55 秒（原版 30 步約 144 秒）；文生影片 8 步約 49 秒（DiT 30 秒）。
- **API／面板（2026-10-05）**：`size: "auto"`、寬高改為 32 的倍數、本文上限預設 256 MB；新欄位 `sigmas`、`references`、
  `video_shift`、`ref_image_short_edge`、`ref_native_canvas`；新錯誤碼 `invalid_sigmas`、`invalid_references`、`invalid_shift`、`invalid_media`。
  面板加「依參考圖比例」／「自動」尺寸、取樣排程進階區、參考生成清單（每列顯示提示詞標籤）、影片 shift 欄位；
  影片預設尺寸改成 864×480／1344×768（官方）／480×864／自動。
- **建置（2026-10-05）**：`build.bat` 固定 `chcp 65001`，否則 ninja 記不到中文 MSVC 的標頭依賴（改標頭不會重編）。

- **效能（2026-10-05 下午）**：
  - H3 串流 fp8/fp4 在 copy stream 上把 LoRA 併入權重：8 步加速 LoRA 的 DiT 34.4 → 30.0 秒（無 LoRA 25.8 秒）。
  - 影片 VAE 解碼器 FFN 改 FP8：PSNR 對 bf16 40.8 dB；注意力投影保留 bf16（全部 FP8 只再快 17%、PSNR 掉到 37.2）。
  - add_bias_bf16 改 2D grid（原本佔 H3 一次生成 GPU 時間 7%）。
  - 影片 VAE 3D 編碼器：im2col 向量化（原佔 52%）、時間 tap 合成一次 GEMM、GroupNorm 1024 執行緒：22 幀 3390 → 1665 ms。
  - Qwen VAE 解碼：自寫 implicit GEMM 卷積、分塊邊緣不白算、分塊 128：2048² 8.3 → 2.4 秒（PSNR 41.4 dB）。
  - 乾淨基準（5060 Ti）：Qwen 1024²・20 步 bf16/fp8/fp4 = 35.0/14.0/9.7 秒；H3 864×480・56 幀 30 步 110 秒、8 步加速 42 秒。
  - 影片 VAE 解碼器 bias 併進 SwiGLU／殘差相加（Codex）：124 幀 bf16 快 4%、fp8 快 8%，PSNR 不變。
  - FP8 注意力前處理（Grok 試做，未合併）：Q/K 量化已近記憶體頻寬上限，四個前處理 kernel 1.79 → 1.61 ms，端到端 <0.5%，不值得多 110 行。
  - FP8 注意力主 kernel 改 4 warp／CTA（Codex）：D=128 暫存器 128 → 168、不再 spill，注意力本身快 2–7%；H3 端到端 <1%（注意力只佔每步一小部分）。
  - LoRA 載入：bf16 張量直接上傳（省逐張量反量化＋同步），viggle r256 併入 3.5 → 3.0 秒。「冷快取 12 秒」重現不了（Grok 量到冷 3.7 秒，冷讀 1.3 GB 約 1 秒）；
    Grok 的多執行緒鎖頁上傳（2.35 秒、多 240 行）不值得，未合併。
  - 同 seed 同程式跑兩次 fp8 出圖並非位元相同（平均每通道差約 0.8），是既有的非決定性，不是 LoRA 造成。
  - nsys 分析：H3 DiT 每步的區塊縮放 GEMM 約 164 TFLOPS，已接近 MXFP8 實測上限，主幹再快要靠 fp4。
- **清理（2026-10-06）**：所有工作樹與未合併的試作分支已刪除，只剩 `main`。
- 紋理修正 VAE（madebyollin）改為預設（2026-10-06）：只改解碼器、編碼器相同；RGBA 已驗證背景透明度更乾淨。沒有檔案時退回官方 VAE。

## 待做

- [ ] H3 bf16／fp4 乾淨基準重量（2026-10-05 量時 GPU0 有其他程式在跑：fp8 129、fp4 102、bf16 317 秒，不能和表格比）
- [ ] 與 sd.cpp 做效能對照
- [ ] H3-Regenerate-2K（官方尚未開源）
- [ ] Qwen-Image PE 提示詞改寫模型（Qwen3.5 架構，尚未支援）
