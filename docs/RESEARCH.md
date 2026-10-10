# 研究筆記：Qwen-Image-2.1 與 MiniMax-H3

資料來源：兩個模型的 HF 模型卡與 config、diffusers `main` 的參考實作
（`transformer_qwenimage21.py`、`pipeline_qwenimage21.py`、`transformer_minimax_h3.py`、
`modular_pipelines/minimax_h3/*`）、stable-diffusion.cpp（sd.cpp）的 `docs/` 與 `src/model/`，
以及社群 GGUF 的檔頭（用 HTTP Range 讀取）。日期：2026-10-04。

## 1. Qwen-Image-2.1（文生圖 + 編輯，7B DiT）

| 元件 | 規格 |
|---|---|
| 文字編碼器 | Qwen3-VL-8B（36 層、hidden 4096、GQA 32/8、head 128、mRoPE interleaved [24,20,20]）。取**最後一層輸出、未經最終 RMSNorm**，丟掉 system prompt 的 token。視覺塔 27 層 hidden 1152（編輯用）。 |
| DiT | 32 層 single-stream，hidden 4096（32 頭 × 128），SwiGLU MLP 12288，**無 bias**。 |
| 調變 | **全模型共用一組** modulation：`Linear(4096 → 4×4096)`，每層都用同一份 `[scale1, gate1, scale2, gate2]`（只有 scale 與 gate，無 shift），gate 先過 `tanh`。 |
| 正規化 | LayerNorm（無仿射）+ 調變；QK 是 per-head RMSNorm。txt_in 是 zero-centered RMSNorm（權重存 `scale-1`）→ Linear → GELU(tanh) → Linear。 |
| 注意力 | **block-causal**：整條序列因果，但每張圖內部雙向。文字 token 只看前面。 |
| causal_condition | 文字與參考圖 token 一律用 `t=0` 的調變 → 它們的啟用值與步數無關 → **prefix KV cache**：第一步算完整序列並存下 prefix 的 K/V，之後每步只算目標圖 token。 |
| RoPE | 3 軸 (frame 16, h 56, w 56)，complex 交錯式（相鄰兩通道一組），theta 10000；圖片的 h/w 以 0 為中心。 |
| 潛空間 | VAE 16× 壓縮、64 通道、patch_size 1（1 token = 16×16 像素）。1024² → 4096 token；2048² → 16384 token。 |
| VAE | `AutoencoderKLQwenImage21`：Wan 風格因果 3D 卷積 VAE（base 96、decoder base 144、z 64、4 通道 RGBA）。latents 需乘 std 加 mean。 |
| 排程 | FlowMatch Euler，sigmas = linspace(1, 1/N, N)，依 token 數做指數型 dynamic shift（base 0.5→max 0.9，256→8192 token）。預設 40 步、**不用 CFG**（true_cfg_scale=1）。 |
| 特色 | 原生 RGBA 透明圖（靠 prompt 觸發）、最多 10 張參考圖編輯。 |

社群權重：`unsloth/Qwen-Image-2.1-GGUF`、`leejet/Qwen-Image-2.1-GGUF`（ComfyUI 命名，`img_mlp.gate_up` 融合），
VAE 與 bf16/int8 權重在 `Comfy-Org/Qwen-Image-2.1`，文字編碼器 GGUF 用官方 `Qwen/Qwen3-VL-8B-Instruct-GGUF`。
LoRA 很多（Turbo/加速、風格、NSFW），格式以 diffusers/PEFT 與 ComfyUI 兩種命名為主。

## 2. MiniMax-H3（文字／圖片／參考 → 影片 + 立體聲音訊，33B DiT）

| 元件 | 規格 |
|---|---|
| 文字編碼器 | Qwen3-VL-32B（64 層、hidden 5120、head 128、GQA 64/8），**取第 50 層輸出**；社群 GGUF 已截成 50 層且去掉最終 norm。 |
| DiT | 50 層 dense single-stream，hidden 5376，注意力 56 頭 × 128 = 7168（比 hidden 大），SwiGLU FFN 14336，2 層 token refiner（給文字流）。 |
| 調變 | 每層 AdaLN 輸出 6×5376×3（shift/scale/gate × 2 組 × 3 種模態：影片/文字/音訊），**約 13B 參數在 AdaLN**。因為只依賴 timestep，可以**整段生成開始前一次算完所有步的調變表**，之後不用載入這 13B。社群 `pruned` GGUF 已把它壓成 `adaln_t_table[1025,8]` + 每層 `[8 → 96768]` 的低秩投影。 |
| 序列 | 一條打包序列：文字 + 條件影像 + 音訊 + 目標影片，**全注意力**（無遮罩）。 |
| RoPE | 3 軸 (t,h,w) 共用 16 個頻率，rotate-half，只轉 head 的前 96 通道。 |
| 影片 VAE | f16t4d24（空間 16×、時間 4×、24 通道），編碼器因果 3D 卷積，解碼器是 36 層 ViT。latent 再做 1×2×2 patch → 每 token 等效 32× 空間。 |
| 音訊 VAE | 32 kHz 立體聲，左右聲道共用編解碼器，40 Hz latent、32 通道（DAC/BigVGAN 類）。 |
| 排程 | 自家 flow 排程，影片 shift 12、音訊 shift 3；**CFG 已蒸餾**（一次前向一步）。 |
| 解析度 | 預設短邊 768、24 fps、最長 15 秒；幀數對齊 17k+5。 |

計算量估算（768p 16:9、10 秒）：約 61K 影片 token；每步 ≈ 2.4 PFLOP（線性層）+ 4 PFLOP（注意力），
**注意力占大宗**，所以 FlashAttention 的品質與 FP8/INT8 注意力是 H3 效能關鍵。

社群權重：`unsloth/MiniMax-H3-GGUF`、`leejet/MiniMax-H3-GGUF`（原始 MiniMax 命名：`blocks.N.attn.qkv_proj`、
`mlp.fc1` 融合 gate/up）、VAE 在 `Comfy-Org/MiniMax-H3`。Turbo 4/8 步 LoRA 很熱門。

## 3. 類似開源專案

| 專案 | 用途 | 參考價值 |
|---|---|---|
| **stable-diffusion.cpp**（MIT） | ggml 通用擴散推論，已支援兩個模型、GGUF、LoRA、prefix cache | 最接近的對照組；命名對應、VAE、tokenizer 細節；**效能基準** |
| diffusers | 官方參考實作 | 數值正確性的黃金標準（parity 測試） |
| ComfyUI / ComfyUI-GGUF | 圖形化工作流 | 社群 LoRA 與 GGUF 命名慣例 |
| SGLang / vLLM-Omni | 多卡伺服器部署 | Ulysses 序列並行、服務化設計 |
| SageAttention | INT8/FP8 注意力 | H3 長序列加速方向（sm_120 有 FP8 mma） |
| Strata（使用者啟發來源） | 單一模型專用 C++/CUDA 引擎 + Python API + Web 面板 | 架構範本：專用核心、權重串流、量測先行 |

## 4. 本機硬體與設計結論

硬體：RTX 5060 Ti 16GB（sm_120，PCIe x8）+ RTX 3060 Ti 8GB（sm_86）、Ryzen 7 5700X、96GB RAM。

1. **擴散模型是計算瓶頸，不是頻寬瓶頸**：量化權重在使用前逐層反量化成 bf16 暫存，再用 cuBLAS tensor core GEMM。
   反量化成本（每層數 ms）相對 GEMM 可忽略；任何 GGUF 量化都能跑。
2. **權重放鎖頁 RAM、逐層串流**：H3 長序列下每層計算 ~0.5 s，遠大於 PCIe 搬 400MB 的 ~30 ms，
   所以 16GB 卡也能跑 Q8 甚至 bf16 的 33B DiT，VRAM 留給啟用值。
3. **雙卡異質分工**：5060 Ti 跑 DiT；3060 Ti 跑文字編碼器與 VAE，伺服器模式下可與下一個請求重疊。
4. **模型專屬最佳化**：Qwen-Image prefix KV cache、H3 AdaLN 一次預算、融合 norm+調變、qk-norm+RoPE。
5. 後續：sm_120 FP8 GEMM／FP8 注意力（需量測畫質）。

## 5. Open-VC Attention 能否套用（2026-10-11）

來源：[MachGen 技術文章](https://www.machgen.ai/blog/open-vc-attn)、[GitHub](https://github.com/MachGen/open-vc-attention)（BSD-3）。

- **是什麼**：稠密（非稀疏）FP8 前向注意力，以 FlashAttention-4 的 CuTe DSL 實作。重點是 ExpCast（一個 FMA 直接算出 FP8 機率碼，略過 MUFU 指數與 FP32→FP8 轉換）、Q/K 每 128 列一個縮放、分母與分子用同一份解碼後機率、從對角線附近開始掃 key、選用的 V 殘差修補（ρ=0.5%）。
- **實測對象**：只有 MiniMax-H3（73,397 token、56 頭、D=128），B200 上注意力約為 BF16 FA4 的 2 倍，相對誤差約 3%。
- **不能直接用**：只支援 B200／B300（SM100／SM103）、Linux、Python + PyTorch cu130；快速路徑要 ≥32,768 token。本機 5060 Ti 是 SM120、3060 Ti 是 SM86，Windows 原生 C++ 引擎也不帶 PyTorch。
- **想法能不能移植**：Miroxen 已有同類路線的 FP8 注意力（INT8 Q·K + 區塊縮放 FP8 P·V，sm_120a，16k token 約 105 TFLOPS）。ExpCast 解決的是 B200「tensor core 遠快於指數單元」的瓶頸；5060 Ti 的 MUFU 指數吞吐換算約 700 TFLOPS 等效，遠高於 MMA 實測上限 180，先前也實測過改指數／整數轉換沒有收益（見 tasks/lessons.md），因此不移植 ExpCast。對角線起掃只是配合 ExpCast 沒有指數餘裕的設計，對一般 online softmax 無益。
- **可留意**：「分母用解碼後的 FP8 機率」與「V 殘差修補」是精度改善，不是加速；目前 FP8 注意力誤差在可接受範圍，沒有畫質問題需要它解決，暫不實作。若之後 720p／10 秒影片（約 7 萬 token、注意力占每步大半）出現 FP8 注意力的畫質問題，再從這兩點著手。

## 6. 單卡 8 GB／16 GB 配置（2026-10-11）

- `--gpus` 指定可見的卡；Qwen 圖片依剩餘顯存決定：雙卡維持 DiT 常駐＋副卡編碼器；單卡先把只跑一次的文字編碼器改逐層串流，仍放不下（8 GB）才把 DiT 也串流。串流沿用 H3 的 BlockStreamer，並支援分開存放的 gate/proj 在槽內接合。
- H3 影片：單卡時取樣前若剩餘顯存不足，先卸下影片／音訊 VAE、視覺塔與編碼器暫存，解碼時再載回；8 GB 以下的單卡另把 VAE 解碼器線性層存成 Q8_0（權重 4.8→2.5 GB），用到才還原 bf16。
- 教訓：Windows WDDM 顯存超用不會報 OOM，而是溢出到系統記憶體、速度掉 10～20 倍（8 GB 卡 864×480 影片解碼 262 秒 → 17 秒）。單卡配置要主動卸載，不能等錯誤。
