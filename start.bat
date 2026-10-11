@echo off
rem 一鍵啟動：沒建置就先建置，服務沒在跑就在背景啟動（無視窗，日誌寫到 out\logs），最後打開工作台
chcp 65001 >nul
cd /d "%~dp0"
set "URL=http://127.0.0.1:8188"

curl.exe -s -m 2 %URL%/health >nul 2>&1 && goto open

if not exist build\miroxen.exe (
  echo 第一次使用，正在建置...
  call build.bat || (echo 建置失敗 & pause & exit /b 1)
)

if not exist out\logs mkdir out\logs
rem shortcut: 額外參數原樣接在 serve 後面，含引號的參數不支援；需要時改寫成參數陣列
powershell -NoProfile -Command "Start-Process -FilePath 'build\miroxen.exe' -ArgumentList 'serve --port 8188 %*' -WindowStyle Hidden -RedirectStandardOutput 'out\logs\server.log' -RedirectStandardError 'out\logs\server-error.log'"
echo 等待服務啟動...
for /l %%i in (1,1,60) do (
  curl.exe -s -m 2 %URL%/health >nul 2>&1 && goto open
  timeout /t 1 /nobreak >nul
)
echo 服務 60 秒內沒有回應，請看 out\logs\server-error.log。
pause
exit /b 1

:open
start "" %URL%
