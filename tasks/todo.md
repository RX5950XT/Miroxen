# Miroxen 任務清單

## 計畫

### 提交、推送與垃圾清理（2026-10-07）
- [x] 檢查提交範圍、既有驗證與 staged diff，提交已完成的專案變更
- [ ] 確認 Git 遠端，推送並比對遠端 HEAD
- [ ] 清理確定可刪的暫存，保留模型、生成作品、驗證資料與運行服務

### 編碼／VAE 效能與分段量測（2026-10-07）
- [x] 加入圖片／影片分段計時，建立可重跑的 A/B 量測
- [x] 有界文字編碼快取，完整圖文內容比對與失效測試
- [x] 圖片 latent 在 GPU 轉換，減少 CPU 往返，驗證數值與實際輸出
- [x] 建置、受影響測試、真實圖片與影片量測，記錄收益及限制

Review：
- GPT-6 Luna 子代理分別檢查編碼快取與 VAE 搬運；主代理整合、審核 CUDA 轉置索引並執行 GPU 驗收。既有未提交工作保留，未 commit／push。
- 共用 Qwen3TextEncoder 精確比對完整 token IDs／視覺嵌入／DeepStack／位置；最近兩筆快取含輸出及 key 最多 128 MiB。模型重建自然清空；LoRA 只改 DiT，因此切換 LoRA 可保留文字快取。超過上限略過，`MX_DISABLE_ENCODER_CACHE=1` 可停用。
- `build.bat` 成功；`ctest --test-dir build -R "qwen3_encoder_cache_test|qwen3_text_parity_test|qwen3_vl_parity_test|server_test" --output-on-failure` 4/4，`ctest --test-dir build -R "qwen_latent_test|qwen_vae_test|h3_video_vae_test|h3_audio_vae_test|qwen_image_e2e_test" --output-on-failure` 5/5；最後建置後快取／latent／server 3/3 再驗。包含常駐與串流 TE CUDA 命中、逐位元輸出一致、VAE 參考比對與原版 Qwen 出圖。Python 語法及 `git diff --check` 通過。
- 真實 API：模型已暖機、FP8，三組不同提示詞各原樣重送，交錯比較。圖片 UC-Q8_0／Turbo8／736×1312／8 步：新提示詞中位 6.200 秒、重送 5.784 秒；每對差值中位省 0.452 秒。文字編碼中位 0.414 秒 → 0.00082 秒。影片 FL2VA Q4_K／LightX2V 4 步（API steps=5）／864×480／56 幀：26.640 → 24.650 秒，每對差值中位省 1.980 秒；TE 2.065 秒 → 0.00072 秒。這是重複提示詞的快取收益，不代表首次任意提示詞都會加速相同比例。
- 新提示詞階段占比（3 次總時間）：圖片 DiT 80.5%、VAE 解碼 10.2%、文字 6.3%、PNG 2.9%；影片 DiT 56.4%、影片 VAE 31.0%、文字 7.8%、音訊解碼／WAV 3.4%、ffmpeg 寫入／結束約 1.1%。階段內包含必要搬運；重疊 GPU／CPU 工作沒有拆成可相加的純核心時間，完整 ctest 未跑。
- Qwen latent layout／正規化在 GPU 做，移除參考編碼與生成解碼的 CPU 往返。同卡路徑與 N=37／1031 的 CPU 公式逐位元一致；decode 明確保留分步乘加。`build/qwen_latent_test.exe --benchmark` 暖機後交錯 20 次：N=3772 舊 0.9729 ms／GPU 0.0145 ms，每次省 1,931,264 bytes；N=16384 舊 3.9584 ms／GPU 0.0311 ms，省 8,388,608 bytes。這是微型測試，總生成只省毫秒，未宣稱整張快數十倍。
- 參考圖實際測試：原樣重送 hit、換另一張圖片 miss、移除 Turbo8 後仍 hit 且 40 步生成成功、卸載重載後 cache 回到 0 hits / 1 miss。參考快取最高 80.7 MiB；視覺塔與參考 VAE 本身仍會重算。停用快取對照三次皆記 disabled／bypass，文字階段重送仍需 0.284 秒。
- FP8 完整出圖並非逐像素決定性；三對圖片平均 RGB 每通道差約 1.2–1.6/255，沒有把整張 PNG 稱為位元相同。紅圍巾編輯輸出已看圖；ffprobe 確認影片 864×480、24 fps、56 幀、2.333333 秒且含音軌。沒有量 720p／10 秒影片，亦未改寫 H3 tile stitch 或驗證 Qwen 跨 GPU VAE 配置；桌面、NVIDIA Broadcast 與 Wallpaper Engine 持續執行，不能拿舊日期數字作直接 A/B。
- 原始量測／請求／階段耗時在 `out/perf-20261007/current.json`，摘要 `summary.json`；參考圖及失效驗證 `cache_validation.json`、停用對照 `cache-off.json`、日誌 `service.log`。可重跑工具 `tools/perf/pipeline_bench.py`；API 工作結果新增 `result.timings`，定義見 docs/API.md。
- 更新後服務已恢復 `0.0.0.0:8188`，輸出目錄 `out/perf-20261007/dashboard`；實際預設 UC-Q8_0／40 步／512×512 兩次成功，第二次文字階段 0.00084 秒，health ok。紀錄 `dashboard_validation.json`，測試 8189 服務已停止；既有生成檔保留。

### 新圖片模型、來源文件與目錄清理（2026-10-06）
- [x] 下載、SHA-256 校驗與切換 UC-Q8_0，實際生成驗證
- [x] 重寫 README，逐一標示模型／編碼器／VAE／LoRA 來源及原版步數
- [x] 整理 QA 輸出、隔離確定無用的暫存並驗證受影響範圍
- [ ] 刪除隔離暫存（自動安全審查拒絕刪除指令，未提供 policy 以外的詳細原因）


Review：
- UC-Q8_0 7,591,557,920 bytes，SHA-256 `cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8` 與 Hub LFS 一致；Q8_0 保留既有檔案精度，運算預設仍為 FP8。
- 新 GGUF 的 FFN 分開存放 gate_layer／proj，載入時在 GPU 接合成既有 gate_up 路徑；沒有改寫下載檔，也沒有增加每步 GEMM 次數。原版 fused 格式與 LoRA 還原路徑均保留。
- `build.bat` 成功；`ctest --test-dir build -R "weights_concat_test|qwen_dit_parity_test|server_test|model_file_test|h3_frames_test" --output-on-failure` 5/5 通過，含舊模型數值比對、接合列順序、輸入形狀驗證、FP8 LoRA 合併／還原。
- 真實 UC-Q8_0／FP8／512×512：40 步生成 7.20 秒、40 步編輯 11.69 秒、Turbo8 8 步 4.56 秒、移除 Turbo8 後 40 步 7.73 秒。首張請求含載入約 16.10 秒；模型狀態確認使用新檔。圖片生成與紅圍巾編輯已看圖確認，紀錄在 `out/model-swap-20261006/validation.json`。不是 720p 的效能數字。
- 瀏覽器 `tests/dashboard_check.js` 29/29 通過；新預設 UC-Q8_0 選項已確認。JS 語法與差異空白檢查通過；測試瀏覽器已關閉，服務保留執行。完整 ctest 與新模型的所有 LoRA 畫質組合未驗證。
- 原始模型、生成作品、官方 HF VAE、`models/ref/` 測試資料保留；37 個零散 QA 檔案集中至 `out/qa/2026-10-06/`。無用 `list/` 虛擬環境、Playwright 歷史與假引擎產物一併隔離；刪除指令遭自動審查拒絕，沒有宣稱已釋放空間。下面歷史紀錄的 `out/dashboard-*.js/.log/.png` 已移至該目錄，生成作品子目錄保持原址。


### 頂部模型管理與精簡 API 頁（2026-10-06）
- [x] 模型狀態與管理移到頂部原生 popover
- [x] API／硬體左右排列、網址複製及其他電腦使用說明
- [x] 明暗桌面／手機、模型操作與 LAN 複製驗證

Review：
- `playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js` 29/29 通過，含頂部模型管理展開／收起、與連線同列、桌面左右欄、手機、明暗對比與既有創作流程；JS 語法及 `git diff --check` 通過。
- `out/dashboard-model-control.js` 實際透過頂部選單載入 Qwen、切換 H3、卸載成功，顯存與 RAM 回收；`out/dashboard-lan-copy.js` 在 HTTP LAN 頁面複製 API 成功，讀回瀏覽器剪貼簿確認 Base URL，模型清單連結 HTTP 200，Escape 關閉選單，390px popover 無溢出。截圖在 `out/dashboard-compact-*` 與 `out/dashboard-top-model-*`。
- 只修改前端及說明，未重跑 C++ 建置／GPU 品質或效能測試；另一台實體電腦的連線仍未驗證，防火牆權限限制同前。


### API／硬體合併與模型管理（2026-10-06）
- [x] 合併 API 與硬體頁，顯示區域網路網址
- [x] 手動載入／切換／卸載模型，生成與管理操作互斥
- [x] 建置、API、瀏覽器與真實模型記憶體釋放驗證

Review：
- `build.bat` 成功；`ctest --test-dir build -R "server_test|model_file_test|h3_frames_test" --output-on-failure` 3/3 通過，涵蓋載入、切換、卸載、無效參數、生成中拒絕管理、金鑰驗證與 JSON 格式限制；JS 語法與 `git diff --check` 通過。
- `playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js` 23/23 通過；`out/dashboard-model-control.js` 實際點擊 Qwen → H3 → 卸載成功，桌面／390px 手機無溢出與 JS 錯誤。
- H3 卸載後程序 RAM 7.38 → 0.40 GiB，GPU0／GPU1 釋放約 5.75／2.93 GiB；卸載後真實圖片工作自動重新載入並成功輸出。服務保持執行，模型最後已卸載；測試瀏覽器關閉。
- 已由本機透過 `http://192.168.2.176:8188/v1/network` 驗證 LAN 位址回應；另一台裝置尚未驗證。沒有管理員權限，無法設定 Windows 防火牆入站規則。


### 儀表板與監控（2026-10-06）
- [x] 對齊模型檔、自訂尺寸、圖片任務、影片秒數、參考素材與 LoRA 設定
- [x] 整理 UI／UX、API 狀態及 CPU／RAM／GPU 監控
- [x] 建置、API 驗證、桌面／手機瀏覽器流程與實際生成驗證

Review：
- `build.bat` 成功；`ctest --test-dir build -R "server_test|model_file_test|h3_frames_test" --output-on-failure`，3/3 通過；JS 語法與 `git diff --check` 通過。
- `npx --package @playwright/cli playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js`，21 項通過：LoRA 排程／任務篩選、素材互斥、模型檔、尺寸、秒數、監控、API 預覽、手機版、明暗按鈕對比與零 JavaScript 錯誤。
- 真實引擎：Qwen Q4_K＋官方 VAE 出圖、切回 Q8_0＋紋理修正 VAE 成功；透明圖／去背 PNG 有透明區域，雙圖遮罩修改成功；FL2VA 與 Ref2VA 均生成含音軌的 24 fps 影片。生成中 `/v1/models` 回應 2 ms；Ref2VA 狀態顯示實際 DiT。結果與紀錄在 `out/dashboard-*`。
- 金鑰模式：假引擎連線、取消、生成、下載通過，金鑰未寫入 browser storage；已關閉測試服務並刪除臨時金鑰。
- 修正最短影片：輸入 5 幀會對齊至 22 幀，避免 VAE 解碼失敗；影片 1 個取樣點改為立即拒絕。這輪真實生成用小尺寸檢查流程，沒有重跑 720p／10 秒效能或逐項畫質評分。


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
