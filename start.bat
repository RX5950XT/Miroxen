@echo off
rem 一鍵啟動：沒建置就先建置，服務沒在跑就開新視窗啟動，最後打開工作台
chcp 65001 >nul
cd /d "%~dp0"
set "URL=http://127.0.0.1:8188"

curl.exe -s -m 2 %URL%/health >nul 2>&1 && goto open

if not exist build\miroxen.exe (
  echo 第一次使用，正在建置...
  call build.bat || (echo 建置失敗 & pause & exit /b 1)
)

start "Miroxen 服務（關閉此視窗即停止）" build\miroxen.exe serve --port 8188 %*
echo 等待服務啟動...
for /l %%i in (1,1,60) do (
  curl.exe -s -m 2 %URL%/health >nul 2>&1 && goto open
  timeout /t 1 /nobreak >nul
)
echo 服務 60 秒內沒有回應，請看服務視窗的錯誤訊息。
pause
exit /b 1

:open
start "" %URL%
