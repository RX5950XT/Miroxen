#!/usr/bin/env bash
# VM 自己用 colab CLI 把 serve.py 送進自己的 kernel（kernel 忙碌＋有連線的 client，VM 才不會被回收）。
# colab exec 斷線後會印「Connection was lost」然後卡住不結束（例如有人從別處 colab exec 換掉了 kernel），
# 所以看到這行就砍掉重送；serve.py 會接手已在跑的伺服器。
# 參數：$1 session 名稱；環境：MX_COLAB_AUTH、MX_COMPUTE。
CUR=/content/cell.cur
while :; do
  colab ${MX_COLAB_AUTH:-} exec -s "$1" --timeout 2592000 --env MX_COMPUTE="${MX_COMPUTE:-fp8}" \
    -f /content/mx/serve.py < /dev/null > "$CUR" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    grep -aq "Connection was lost" "$CUR" && { kill "$pid"; break; }
    sleep 10
  done
  wait "$pid" 2>/dev/null
  { cat "$CUR"; echo "[$(date +%T)] cell 連線結束，10 秒後重送"; } >> /content/cell.log
  sleep 10
done
