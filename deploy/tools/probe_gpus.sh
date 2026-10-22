#!/usr/bin/env bash
# 逐一嘗試開各種 GPU，看帳號有哪些權限；開成功就立刻關掉。COLAB_AUTH 例如 "--auth adc"。
for g in ${@:-G4 H100 A100 L4 T4}; do
  echo "== $g"
  timeout 240 colab $COLAB_AUTH new -s probe --gpu "$g" 2>&1 | tail -2
  colab $COLAB_AUTH status -s probe 2>&1 | head -12
  colab $COLAB_AUTH stop -s probe >/dev/null 2>&1
done
