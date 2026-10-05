# Miroxen 任務清單

## 計畫

- [x] 研究兩個模型、類似專案、本機硬體 → `docs/RESEARCH.md`
- [x] 存放庫骨架：CMake + CUDA（sm_120;86）、核心型別、介面標頭
- [ ] 核心元件（平行分工）
  - [ ] GGUF / safetensors 讀取 + GPU 反量化（codex）
  - [ ] FlashAttention bf16（codex）
  - [ ] Qwen2 BPE tokenizer（grok）
  - [ ] HTTP API + 操作面板（grok，先接假引擎）
  - [ ] 逐元素核心、GEMM、線性層（含 LoRA）、權重串流（主代理）
- [ ] Qwen-Image-2.1
  - [ ] Qwen3-VL 文字編碼器（純文字路徑）
  - [ ] DiT + prefix KV cache
  - [ ] VAE 解碼器
  - [ ] 排程 + 端到端出圖；與 diffusers 做數值比對
  - [ ] 編輯（視覺塔 + 參考圖）
- [ ] MiniMax-H3
  - [ ] 32B 文字編碼器（共用 Qwen3-VL 程式碼）
  - [ ] DiT（AdaLN 預算 / pruned 表）
  - [ ] 影片 VAE 解碼（ViT decoder）+ 音訊 VAE 解碼 + ffmpeg 封裝
  - [ ] 端到端 T2VA；FL2VA
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
  INT8 GEMM 計畫由區塊縮放 GEMM 取代（wt/i8gemm 不再需要）。
- **FP8 注意力（2026-10-05）**：INT8 Q·K + FP8 P·V（sm_120a 區塊縮放 MMA），16k token 105 TFLOPS（bf16 37），fp8/fp4 模式自動使用。
  Qwen 1024²・20 步 fp8 18.7 → 16.5 秒、fp4 16.7 → 11.9 秒；H3 尚未重測（測試時系統記憶體不足被中止）。
- 待做：H3 重測 FP8 注意力、FP8 注意力前處理（約占 10–15%）、正規化／silu 與量化融合、文字編碼器（GPU1 3060 Ti 不支援 FP8/FP4）、
  VAE 加速（Qwen 2048² 9.5 s、H3 124 幀 36 s）、FL2VA、影像編輯、sd.cpp 效能對照、H3 串流模式的 LoRA 併入。

