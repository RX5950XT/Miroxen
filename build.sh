#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-build}"
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$ROOT_DIR/$BUILD_DIR"; fi

if [[ -n "${CUDACXX:-}" ]]; then
  NVCC="$CUDACXX"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
else
  echo "nvcc not found; install CUDA Toolkit or set CUDACXX" >&2
  exit 1
fi

# CUDA 12.8／13.0 的 nvcc 會把「取 half 位元的低位元組」錯編成數值轉整數（Q8_0 縮放值壞掉），13.3 起正常。
ver="$("$NVCC" --version | grep -o 'release [0-9]*\.[0-9]*' | cut -d' ' -f2)"
major="${ver%%.*}" minor="${ver#*.}"
if [[ -z "$ver" ]] || (( major < 13 || (major == 13 && minor < 3) )); then
  echo "nvcc ${ver:-?} is known to miscompile this project; use CUDA >= 13.3 (set CUDACXX)" >&2
  exit 1
fi

CUDA_ARCHS="${CUDA_ARCHS:-120a;80;86;89;90}"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-$(nproc)}"
