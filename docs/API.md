# Miroxen HTTP API

本機推論服務。預設只聽 `127.0.0.1:8188`。JSON 使用 UTF-8。時間戳是 Unix 秒。

面板在 `GET /`，靜態資源在 `/web/`，生成檔在 `/files/`。這三條與 `GET /health` 不需要金鑰。`api_key` 不是空字串時，所有 `/v1/*` 都要帶：

```
Authorization: Bearer <api_key>
```

比對的是整段標頭，沒有多餘空白。失敗回 `401`。

## 錯誤

HTTP 狀態搭配這個物件，成功的回應沒有 `error` 包裹：

```json
{"error": {"message": "說明", "type": "invalid_request_error", "code": "invalid_size"}}
```

| 狀態 | 何時 |
|---|---|
| 400 | 參數不合法 |
| 401 | 缺金鑰或金鑰不對 |
| 404 | 找不到路徑、工作或檔案 |
| 413 | 本文超過 `max_body_mb`（預設 64；0 表示不限制） |
| 499 | 同步圖像生成被取消 |
| 500 | 引擎失敗 |
| 503 | 伺服器正在停止，同步請求還沒完成 |

常見 `code`：`invalid_json`、`missing_field`、`missing_model`、`model_not_found`、`model_kind`、`model_not_loaded`、`empty_prompt`、`prompt_too_long`、`invalid_size`、`invalid_steps`、`invalid_n`、`invalid_cfg`、`invalid_frames`、`invalid_format`、`invalid_compute`、`invalid_type`、`invalid_lora`、`lora_not_found`、`invalid_image`、`unauthorized`、`payload_too_large`、`job_not_found`、`not_found`、`cancelled`、`engine_error`、`shutting_down`。

每一筆請求在紀錄裡留一行：方法、路徑、狀態碼。

## 通用查詢

`GET /health` → `{"status":"ok"}`

`GET /v1/models` 為 OpenAI 風格清單。`kind` 是 `image` 或 `video`，`loaded` 表示能不能用。

```json
{"object":"list","data":[{"id":"qwen-image-2.1","object":"model","owned_by":"miroxen","kind":"image","loaded":true,"detail":"..."}]}
```

`GET /v1/system`

```json
{"gpus":[{"index":0,"name":"NVIDIA GeForce RTX 5060 Ti","total":17179869184,"free":14000000000}],"queue_length":1}
```

`total` 與 `free` 是位元組。`queue_length` 是尚未完成的工作數（`queued` 加 `running`）。

`GET /v1/loras` 只列 `lora_dir` 直屬的 `*.safetensors`（不遞迴），依名稱排序。

```json
{"data":[{"name":"detail.safetensors","size":12345}]}
```

## 生成參數

`model` 必填，而且必須已載入、種類相符。`prompt` 去掉前後空白後不可為空，最多 8000 個 Unicode 碼位。`negative_prompt` 同樣上限，可省略。`n` 若出現必須是整數 `1`。

`size` 是 `"寬x高"`（`x` 或 `X`，兩邊都是數字）。也可以改傳整數 `width` 與 `height`。寬與高都要是 16 的倍數。圖像是 256 到 2752，影片是 256 到 1920。

`steps` 是 1 到 200。省略時圖像 40、影片 30。`cfg_scale` 若出現必須是 0 到 20 的有限數字；只有圖像會把它交給引擎，預設 1。`seed` 是整數，省略或 `-1` 表示由引擎抽一個。

`loras` 是 `[{"name":"檔名.safetensors","scale":1.0}]`。`name` 只能是目錄裡的檔名：拒絕 `..`、絕對路徑、斜線、反斜線與磁碟機冒號。`scale` 預設 1。

`images`、`first_frame`、`last_frame` 是 base64，或 `data:image/...;base64,...`。內容必須是能解出來的 PNG 或 JPEG。參考圖最多 10 張，會先存到 `out_dir/tmp`，工作結束後刪除。影片的首幀、尾幀各一張，可省略。

影片幀數欄位是 `num_frames`（也接受 `frames`），5 到 361，預設 56。

`compute` 是 DiT 的運算精度，可省略（用伺服器啟動時的 `--compute`，預設 `fp8`）：`bf16`（最準）、`fp8`（MXFP8，約快 2 倍、畫質幾乎不變）、`fp4`（NVFP4，最快）。`fp8`／`fp4` 需要 Blackwell 以上的顯卡，不支援時自動退回 `bf16`。和目前載入的精度不同時會重新載入模型（Qwen-Image 約 10 秒、H3 約 40 秒）。

## 圖像

`POST /v1/images/generations` 會排隊，並等到這件工作結束才回應。同一時間只有一個工作在跑。

`response_format` 預設 `b64_json`，也可設 `url`。`url` 指到 `/files/<工作 id>.png`。

```json
{"created":1710000000,"data":[{"b64_json":"...","seed":42}]}
```

`url` 格式時該筆是 `{"url":"/files/job_00000001.png","seed":42}`，沒有 `b64_json`。

## 影片與非同步工作

`POST /v1/videos/generations` 立刻回工作物件，狀態是 `queued`。

`POST /v1/jobs` 的本文多一個 `type`：`image` 或 `video`，其餘欄位相同。非同步結果一律是網址，不內嵌 base64。

工作物件：

```json
{
  "id": "job_0000002a",
  "object": "image.job",
  "type": "image",
  "status": "running",
  "created": 1710000000,
  "prompt": "雨夜的書桌",
  "progress": {"stage": "denoise", "step": 3, "total": 40},
  "result": null,
  "error": null
}
```

`object` 為 `image.job` 或 `video.job`。`status` 是 `queued`、`running`、`succeeded`、`failed`、`cancelled`。`stage` 常見 `encode`、`denoise`、`decode`。

成功時：

```json
"result": {"url": "/files/job_0000002a.png", "seed": 42, "seconds": 1.25}
```

失敗或取消時 `error` 為 `{"message","type","code"}`。取消的訊息是「已取消」。

`GET /v1/jobs` 回最近 50 筆，新的在前：`{"object":"list","data":[...]}`。已結束的工作最多留 200 筆。

`GET /v1/jobs/{id}` 回單一工作。`DELETE /v1/jobs/{id}` 要求取消：還在排隊的會立刻變成 `cancelled` 並移出佇列；正在跑的會設取消旗標，引擎在下一步結束時停下。

`GET /v1/jobs/{id}/events` 是 Server-Sent Events。每個事件是一行 `data: ` 加上工作 JSON，進度有變就送，結束狀態送出後關閉。閒置時約每 15 秒一行 `: ping`。

## 靜態檔

`GET /` 回 `web/index.html`。`GET /web/<檔名>` 與 `GET /files/<檔名>` 只提供該目錄直屬的檔案。檔名限 ASCII 的字母、數字、`.`、`_`、`-`，不能以 `.` 開頭，也不能含 `..`。

## 面板

`web/` 是不需建置步驟的頁面，複製在繁體中文。它呼叫 `/v1/models`、`/v1/system`、`/v1/loras`、`/v1/jobs`，並用 EventSource 訂進度。瀏覽器的 EventSource 不能加 `Authorization`，所以開金鑰時請用其他客戶端，或把金鑰留空給本機面板。

## 程式進入點

伺服器本體是 `mx::run_server(engine, opt)`。在 `main.cpp` 接上 `miroxen serve` 之前，可從測試程式啟動假引擎：

```
build\server_test.exe serve 8188
```

工作目錄要在專案根目錄，這樣才找得到 `web/`。
