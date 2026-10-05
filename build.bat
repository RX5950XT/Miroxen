@echo off
rem 固定 UTF-8 字碼頁：cl 的 /showIncludes 前綴（中文）編碼才會和 CMake 設定時偵測到的一致，ninja 才記得到標頭依賴
chcp 65001 >NUL
rem Usage: build.bat [build_dir=build] [extra cmake args...]
setlocal
set BUILD_DIR=%1
if "%BUILD_DIR%"=="" set BUILD_DIR=build
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL || exit /b 1
cd /d "%~dp0"
rem msys ninja fails under MSVC env; use the uv-installed one.
set NINJA=%USERPROFILE%\.local\bin\ninja.exe
if not exist "%BUILD_DIR%\build.ninja" (
  cmake -S . -B "%BUILD_DIR%" -G Ninja "-DCMAKE_MAKE_PROGRAM=%NINJA%" -DCMAKE_BUILD_TYPE=Release %2 %3 %4 %5 || exit /b 1
)
cmake --build "%BUILD_DIR%" -j 16 || exit /b 1
