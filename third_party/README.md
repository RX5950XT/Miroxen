# 第三方標頭

只收單檔標頭，建置時由 `CMakeLists.txt` 的 `third_party` include 路徑帶入。

| 檔案 | 版本 | 授權 | 來源 |
|---|---|---|---|
| `httplib.h` | cpp-httplib 0.59.0 | MIT，Copyright (c) 2026 Yuji Hirose | https://github.com/yhirose/cpp-httplib/releases/tag/v0.59.0 |
| `nlohmann/json.hpp` | nlohmann/json 3.12.0 | MIT，Copyright (c) 2013-2025 Niels Lohmann | https://github.com/nlohmann/json/releases/tag/v3.12.0 |
| `stb_image.h` | nothings/stb v2.30 | 公有領域（Sean Barrett） | https://github.com/nothings/stb |
| `stb_image_write.h` | nothings/stb v1.16 | 公有領域（Sean Barrett） | https://github.com/nothings/stb |

stb 兩份標頭取自 2026-10-04 的 master。檔案開頭的版本註解即為準。
