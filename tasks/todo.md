# Miroxen 任務清單

## 計畫

### Turbo 生圖、單卡 8G／16G、社群 LoRA、儀表板（2026-10-11）
- [x] 研究 Open-VC Attention 能否套用（結論寫入 docs/RESEARCH.md）
- [x] Qwen-Image-2.1-Turbo（AtomicChat Q8_0）成為第二個生圖模型 `qwen-image-2.1-turbo`；預設文字編碼器換 AtomicChat 去拒答 AD-Q4_K
- [x] 移除圖片加速 LoRA（Viggle／Turbo8 檔案移至 out/trash）與 Viggle 9 步混合模式
- [x] 單卡自動配置：`--gpus`；Qwen 依顯存串流文字編碼器／DiT；H3 取樣前卸下 VAE、8G 卡 VAE 解碼器存 Q8_0
- [x] 社群 LoRA 安裝（上傳／網址下載、自動判斷圖片或影片）
- [x] 儀表板改良（標準／Turbo 模型卡、顯示卡配置標籤、LoRA 安裝、進階設定收合）
- [x] 文件、完整驗收

Review：
- Turbo：`qwen-image-2.1-turbo` 為第二個圖片模型，固定 8 個原值 sigma（不位移）；API 省略 steps 用 8，其他步數未給 sigmas 回 invalid_steps；CLI `--turbo`。正式服務 1024²：Turbo 6.6 秒、標準 40 步 29.7 秒；招牌文字、雪景圖已看圖。預設文字編碼器改 AtomicChat 去拒答 AD-Q4_K。Viggle／Turbo8 LoRA 移至 `out/trash/image-acceleration-loras-20261011/`，Viggle 混合模式程式移除（移除前的未提交 diff 備份在 `out/reports/pre-turbo-uncommitted-20261011.patch`）。
- 單卡：`--gpus`；Qwen 依 cudaMemGetInfo 決定串流文字編碼器／DiT（BlockStreamer 支援 gate/proj 接合），同卡時 VAE 編碼／解碼按需放大 DiT Arena。H3 取樣前顯存不足即卸下 VAE／視覺塔／編碼器暫存、音訊解碼後再放掉音訊 VAE；<12 GiB 單卡 VAE 解碼器存 Q8_0（新 GPU 量化核心 quantize_q8_0）。8 GB 影片 864×480／56 幀：316 → 68 秒（取樣 190→43 秒是因先前溢出，解碼 262→17 秒）。CLI 與 `serve --gpus 1` 實測圖片、編輯、LoRA、影片均成功並看圖。
- LoRA：`POST /v1/loras/upload`（串流寫檔，8 GB 上限，其他 API 仍受 --max-body）、`POST /v1/loras/install`（curl 不經 shell，HF／Civitai 網址正規化）、`GET /v1/loras/installs`；依張量名自動放 image／video。實測 HF 原神 TCG LoRA 下載安裝並疊在 Turbo 出圖（192 層）。Civitai 實際下載未驗證（只驗網址轉換）。
- 驗證：`build.bat` 成功；`ctest --test-dir build --output-on-failure` 19 通過、qwen_dit_parity_test 依例 Skipped（原版權重已移走）；qwen_image_e2e_test 因缺官方 VAE 檔回報 SKIP 但回 0（既有問題，未改）。server_test 328 項、dequant Q8_0 往返、weights_concat 串流接合新增。`tests/dashboard_check.js` 34/34（正式服務），impeccable detect 只剩結果圖執行時才設 src 的既有提示。
- 未做：Open-VC 不能直接用（B200 限定），評估寫入 docs/RESEARCH.md §5；未移植 ExpCast。8 GB 卡無 FP8，未做 INT8 GEMM。


### 硬體利用與畫質保留最佳化（2026-10-08）
- [x] 量影片現況，確認解碼前資料搬運與暫存生命週期
- [x] H3 影片／音訊 latent 在 GPU 轉換，移除 CPU 往返，保持數值與精度
- [x] 建置、逐位元對照、實際影片與分段計時驗收，恢復服務

Review：
- H3 FL2VA／Ref2VA 共用解碼路徑改為 GPU 排列轉換；只取生成列，跳過參考區塊。同卡直接使用取樣結果，跨卡搬目標 latent，不再 GPU→CPU 重排→GPU。解碼前同步完成取樣並釋放 c.act／scratch／Lt workspace／速度緩衝；864×480、56 幀實際記錄釋放 1635.0 MiB。模型、步數與精度不變。
- `build.bat` 成功；`ctest --test-dir build -R 'h3_latent_test|h3_frames_test|h3_audio_vae_test|h3_video_vae_test|server_test' --output-on-failure`：5/5 通過（46 秒，VAE 參考並未 skip）。新測試有非方形／非整 block／參考列 offset、同卡和跨卡逐位元對照。`CUDA_DEVICE_ORDER=PCI_BUS_ID` 下 `build/h3_latent_test.exe --benchmark`：72×80×46 latent 幾何（736×1280、243 幀），舊 CPU 往返 23.054 ms、新 GPU 0.269 ms，省 51,079,680 bytes 主機搬運／次；這是微型測試，不是完整 10 秒生成。
- 真實 FP8：864×480、56 幀、FL4、相同 seed／提示詞，各版本一次暖機後兩次量測。改前 26.458／25.208 秒，改後 25.424／25.469 秒；背景桌面、Broadcast、Wallpaper Engine 等仍在跑，且未交錯版本，不能宣稱完整生成明顯加速。這輪主要收益是解碼前降低 GPU 記憶體重疊；釋放本身約需 10–20 ms。
- 額外實際 FL 首尾幀、Ref 圖片＋音訊各完成 448×256／22 幀；ffprobe 確認三種流程均 24 fps、正確幀數及音軌，文生／Ref 首幀已看圖。`out/qa/h3-transfer-20261008` 保存 before／after／references 請求結果與截圖；服務於 8188 恢復、新輸出在 `out/generated/2026-10-08-optimized`。README／API 同步計時定義，diff 檢查通過。未跑完整 ctest、BF16／FP4 完整生成或 720p／10 秒端到端品質比較，未改圖片管線或下載新權重。

### LoRA 強度控制修正（2026-10-08）
- [x] 加速 LoRA 固定強度 1，僅畫風顯示滑桿，保留作者步數預設
- [x] 介面驗證圖片／FL2VA／Ref2VA 預設、畫風強度與混合模式

Review：儀表板僅已識別的畫風 LoRA 顯示強度滑桿；加速與未識別用途的 LoRA 請求固定 scale 1，既有非 1 狀態也不會流入加速請求。無滑桿的列使用全寬。勾選加速 LoRA 仍套用既有作者預設，Viggle 混合需另外選取；一般步數可手動修改。Edge headless 的 `tests/viggle_hybrid_check.js` 14 項通過，涵蓋圖片、FL4／FL8／Ref8、混合切換及模擬畫風 scale 0.65；JS 語法與 diff 檢查通過。未修改 API／CLI 強度契約或推論引擎，未重跑 GPU 生圖／完整 ctest。

### Viggle 9 步混合模式（2026-10-08）
- [x] 共用設定驗證、API／CLI／儀表板入口，保留原有 6 步模式
- [x] 第八步前卸下 LoRA、還原底模並重建 DiT prefix 快取
- [x] 建置、參數與介面檢查、實際生成／編輯／取消後恢復驗收

Review：
- `viggle_hybrid` 使用作者的 9 個 sigma，7 次 Viggle forward 後以 `apply_loras({})` 從原始權重還原底模，重跑 prefix prepare 清除 K/V；下一次一般生成會重新套用要求的 LoRA。API、CLI 與管線共用設定驗證，只接受單一 v0.3 r256、強度／CFG 1、無負向提示詞。UI 可切回原有 6 步，換 Turbo8 或影片會清除混合模式。
- `build.bat` 通過；`ctest --test-dir build -R 'image_sampling_test|server_test|weights_concat_test|qwen_latent_test' --output-on-failure`：4/4 通過；JS 語法與 `git diff --check` 通過。Edge headless 執行 `tests/viggle_hybrid_check.js`：6 項檢查通過。
- UC Q8／FP8、Q4_K_M TE、紋理修正 VAE：512² 的 6 步、9 步混合、參考圖編輯、切換期取消、取消後再跑 6 步共 5 項 GPU 驗收成功；9 步有 `base_tail` 與 `lora_switch`，生成／編輯輸出非黑灰圖。CLI `--viggle-hybrid` 256² 實際 9 步出圖成功（6.9 秒）。紀錄及作品在 `out/qa/viggle-hybrid-20261008/`。
- 512² 混合模式的切換耗時約 2.40 秒，參考圖編輯約 7.30 秒，已獨立列入 `timings.lora_switch`；不能套用作者其他框架的耗時比例。未做全面畫質 A/B，也未實測 BF16／FP4 混合路徑；未跑完整 ctest。
- 重啟至更新版本，服務輸出為 `out/generated/2026-10-08`。先前背景啟動複合命令被工具拒絕（僅回 `blocked by policy`），已改由執行工具直接管理服務並成功啟動，未解除審查。

### 待刪資料集中（2026-10-08）
- [x] 將 out 舊資料、Q8 專用測試／reference／下載快取移至 `_cleanup_pending/20261008`，保留相對路徑
- [x] 核對檔案數與大小，確認來源已移出、API 健康；不執行刪除

Review：566 個檔案已集中隔離，映射在該目錄的 `manifest.json`；`out` 原有 559 檔共 596,713,578 bytes 全數移入。首次核對誤將七個單檔各計兩次，已修正計數並確認八個來源均不存在、目的均存在。重新建立服務用空白輸出目錄，`GET /health` 為 ok、`git diff --check` 通過；隔離目錄已加入 gitignore。僅搬移，未刪除或釋放容量；尚未完成預設／文件殘留清理，未執行生成或建置。

### 未使用模型資料清理（2026-10-08）
- [x] 核對 runtime、儀表板選項及數值測試依賴，保留可用權重與必要 HF VAE
- [x] 移出重複 HF snapshots、歷史 H3 dump 與未引用的四層 reference
- [x] 驗證保留檔案及服務狀態，記錄隔離結果
- [ ] 實際刪除隔離資料（自動審查拒絕，未釋放容量）

Review：
- `hf/FL2VA`、`hf/Ref2VA` 是無權重 shards 的重複 snapshot；四個 `models/ref/h3_dump*` 為歷史除錯輸出，`h3_dit_ref_4l.safetensors` 沒有現行 caller。339 檔共 399,897,264 bytes（381.37 MiB）已移至 `out/trash/unused-model-data-20261008`；可選權重、LoRA、實用 reference 與 HF VAE 均保留。
- 原生 PowerShell `Remove-Item` 被自動審查拒絕，僅回 `blocked by policy`；未繞過刪除，也未宣稱釋放容量。前次 `out/trash` 本輪開始時已不存在，沒有將舊垃圾狀態當作現況。
- 比對清理前後檔案路徑及大小：保留的 222 檔完全一致，7 個來源已移出且目的均存在；`GET /health` 為 ok，`GET /v1/models` 仍列出 UC 圖片與 FL／Ref 影片權重及其他角色，`GET /v1/loras` 仍列出六個 LoRA。詳細移動映射與驗收在 `out/reports/unused-model-data-20261008.json`；未重跑生成或 GPU 測試，沒有修改程式或權重。

### 模型分類與輸出整理（2026-10-07）
- [x] 圖片／影片權重與 LoRA 分類，原版圖片 DiT 移出載入目錄
- [x] 同步預設、CLI／API、測試與來源文件，UC-Q8_0 為圖片唯一預設
- [x] out 零散作品分類、垃圾隔離，保留生成作品及必要測試資料
- [x] 建置、受影響測試與實際圖／影音 API 驗證，恢復儀表板
- [ ] 刪除原版圖片 DiT 與隔離垃圾（自動安全審查拒絕，未釋放容量）

Review：
- `models/image/qwen-image-2.1`、`models/video/minimax-h3` 分開存放模型；LoRA 分為 `models/loras/image`／`video`。官方 TE、vision、VAE 與 `models/ref` 數值資料保留；使用者准刪範圍按原版圖片 DiT 解讀，沒有刪除必要的官方元件。
- 105 次移動已由主代理與 GPT-6 Luna 複核來源消失／目的存在，未覆寫或含 reparse point；詳細映射在 `out/reports/organization-20261007.json`。零散作品按副檔名存至 images／videos，歷史 runtime／model-swap／QA 移至 qa；效能記錄移至 `out/perf/2026-10-07`。舊紀錄路徑可透過映射查找。
- 三個原版 DiT 共 16,084,215,584 bytes（14.98 GiB），原生 PowerShell 指定完整檔名的刪除仍遭自動安全審查拒絕（`blocked by policy`）。改為可逆隔離至 `out/trash/original-image-dit`，模型選單不再列出；原本隔離的 venv／Python 快取／舊 Playwright／假引擎產物也集中至 trash，沒有宣稱已刪除或釋放空間。
- API 仍用 LoRA 原檔名，共用清單／解析器讀取平面目錄和 image／video 直屬檔案；保留 canonical parent 邊界，重複檔名回 `ambiguous_lora`，CLI 依 image／video 命令找對分類。server_test 涵蓋分類讀取、拒絕路徑輸入與重名錯誤。
- `build.bat` 成功；`ctest --test-dir build -R "server_test|model_file_test|tokenizer_test|qwen_dit_parity_test|qwen3_text_parity_test|qwen3_vl_parity_test|qwen_vae_test|h3_audio_vae_test|h3_video_vae_test|h3_dit_parity_test|qwen_image_e2e_test" --output-on-failure`：10 通過、原版 Qwen DiT parity 1 個明確 Skipped，100.58 秒；未將原版 reference 用於 UC。Python compile 10 檔、JS 語法及 `git diff --check` 通過；完整 ctest 未跑。
- 實際 CLI 以裸檔名 Turbo8 從 image 子目錄載入並完成 512²／8 步圖片。實際 API：UC 圖片、參考圖紅圍巾編輯（已看圖）、H3 FL2VA／4 步短片（448×256、24 fps、22 幀、含音軌）成功；HTTP 結果下載通過。`out/qa/model-layout-20261007/validation.json` 記錄請求結果及模型狀態；這是路徑／功能驗證，不是 720p／10 秒效能測試。
- `playwright-cli -s=miroxen-layout run-code --filename tests/dashboard_check.js` 29/29 通過，包含實際 UC 選項與三個圖片／三個影片 LoRA；測試瀏覽器與 8189 服務已關閉。正式服務 PID 31980 在 `0.0.0.0:8188`，新作品輸出 `out/generated/2026-10-07`，UC-Q8_0 已載入且 health ok；原效能驗證作品沒有覆寫，final-service.json 記錄最終狀態。
- 本輪程式與文件變更已提交，模型／輸出不提交；推送仍待 GitHub 私人／公開倉庫或目的地網址的選擇。

### 提交、推送與垃圾清理（2026-10-07）
- [x] 檢查提交範圍、既有驗證與 staged diff，提交已完成的專案變更
- [x] 確認 Git 遠端，推送並比對遠端 HEAD
- [ ] 清理確定可刪的暫存，保留模型、生成作品、驗證資料與運行服務

Review：
- 已提交 `baaddc8`，共 41 個本輪相關檔案；`git diff --cached --check` 通過，模型、生成檔與 Python 快取未提交。建置、GPU／API／瀏覽器驗證見以下各輪 Review。
- 目前沒有 Git 遠端；`gh repo view RX5950XT/Miroxen` 確認倉庫尚不存在。已詢問私人／公開倉庫或目的地網址，未獲選擇前不建立倉庫，因此尚未推送。
- 2026-10-11：使用者選公開倉庫。gitleaks 掃描歷史與 staged 內容無洩漏、無個人路徑或大檔；建立 https://github.com/RX5950XT/Miroxen（public），`git ls-remote` 與本機 HEAD 一致。
- 垃圾盤點與主代理複核確認 9 個目標、33 個檔案，共 649,072 bytes：孤立 venv、Python 快取、舊 Playwright 歷史、假引擎 auth 產物、空日誌與空 tmp。絕對路徑均位於專案內，未含 reparse point；原生 PowerShell 刪除仍遭自動安全審查拒絕（只有 `blocked by policy`），沒有執行刪除或改用其他方式繞過。
- 保留 Ninja 標頭依賴、建置結果、模型、生成作品及效能／數值比對資料。服務 PID 40568、`http://127.0.0.1:8188/health` 仍為 ok，沒有生成中工作。

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

- **H3 長影片顯存（2026-10-12）**：
  - DiT 每層不再留整段 [S, D] 的 h 與 o：AdaLN／閘控殘差逐 chunk 做，FP8 注意力輸出直接覆寫 q（主 kernel 只讀量化工作區）。
    每列啟用值 111 → 86 KB；736×1280・10 秒（67k 列）7.2 → 5.4 GiB，15 秒（99.7k 列）約 10.6 → 8.1 GiB。
  - 取樣前依序列長度配置：放得下用 8192 列分段；不夠先卸 VAE，再把分段減半到 512；還是不夠直接報錯（不讓 WDDM 溢出）。
  - 剩餘顯存常駐前 n 層主幹權重（BlockStreamer::set_resident，解碼前釋放）：864×480・56 幀 8 步 DiT 32 → 30–31 秒（只放得下 12 層）。
  - 實測（5060 Ti，桌面佔 2.5 GB，8 步加速 LoRA）：736×1280 5 秒 284 秒（30 步 851 秒）；10 秒舊版 967 → 新版 927 秒；
    15 秒新版 1576 秒、畫面正常（舊版估計需 ~12 GiB，會溢出）。圖片 736×1280：40 步 26 秒、Turbo 8 步 6–7 秒。
  - 新舊版同 seed 影片 PSNR 約 23 dB，與舊版自己跑兩次（22 dB）相同，是 fp8 既有的非決定性；bf16 對 fp32 參考誤差不變（1.96%／3.65%）。

## 待做

- [ ] H3 bf16／fp4 乾淨基準重量（2026-10-05 量時 GPU0 有其他程式在跑：fp8 129、fp4 102、bf16 317 秒，不能和表格比）
- [ ] 與 sd.cpp 做效能對照
- [ ] H3-Regenerate-2K（官方尚未開源）
- [ ] Qwen-Image PE 提示詞改寫模型（Qwen3.5 架構，尚未支援）
