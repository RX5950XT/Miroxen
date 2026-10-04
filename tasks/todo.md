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
- [ ] LoRA（diffusers / ComfyUI 命名）
- [ ] 效能：與 sd.cpp 對照、雙卡分工、FP8 路徑

## 狀態（2026-10-05）

- 完成並驗證：GGUF/safetensors、反量化、FlashAttention、tokenizer、Qwen3-VL 編碼器（對 transformers 1.5%）、
  Qwen-Image DiT（對 diffusers 2.4%）、Qwen-Image VAE（41 dB）、Qwen-Image 端到端出圖（含 RGBA）、HTTP API + 面板、
  H3 DiT（對 fp32 參考 2.0%/3.7%，版面座標完全一致）、H3 音訊 VAE（95 dB）、H3 影片 VAE（51.5 dB）、逐層權重串流。
- **未解 bug**：H3 端到端影片是雜訊方塊（取樣未收斂）。已排除：32B 編碼器層數（50 層正確）、DiT 數值、版面、
  曲線表索引方向（sd.cpp 同樣用 t=1-sigma）。下一步：與 sd.cpp 同 prompt 對照、逐步比對 x0 預測、確認影片 VAE
  的 latents_mean/std 來源（unsloth 檔無 config）。
- 進行中被中斷：INT8 GEMM（wt/i8gemm）、INT8 注意力（wt/i8attn）子代理已停止，worktree 保留未合併。
- 待做：INT8 路徑接入 Linear、VAE 加速（Qwen 2048² 9.5 s、H3 124 幀 36 s）、FL2VA、影像編輯、LoRA 實測、sd.cpp 效能對照。

