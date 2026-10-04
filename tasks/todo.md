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

## Review

（完成後填寫）
