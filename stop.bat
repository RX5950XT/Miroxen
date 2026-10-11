@echo off
rem 停止背景執行的 Miroxen 服務
chcp 65001 >nul
taskkill /im miroxen.exe /f >nul 2>&1 && echo Miroxen 服務已停止。 || echo Miroxen 服務沒有在執行。
timeout /t 2 /nobreak >nul
