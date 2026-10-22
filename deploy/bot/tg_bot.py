#!/usr/bin/env python3
"""Telegram bot：/up 自動搶 GPU（依序 G4 → A100 → L4）開 Miroxen 並回傳公開網址，沒人用就自動關機。
在 WSL 執行（要用 mx-colab）：TG_TOKEN=... TG_CHAT_ID=123 python3 tg_bot.py
  TG_CHAT_ID  允許操作的 chat id（逗號分隔）；不設時 bot 只回你的 chat id，方便第一次設定
  MX_GPUS     搶機順序，預設 "G4 A100 L4"
  MX_IDLE_MIN 連續幾分鐘沒有新工作、佇列也空就關機，預設 30
指令：/up /status /down /id
"""
import json, os, subprocess, sys, threading, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # deploy/，mx-colab 在這裡
CONF = os.environ.get("MX_CONF", os.path.expanduser("~/.config/miroxen-colab"))
EP = os.path.join(CONF, f"endpoint-{os.environ.get('MX_SESSION', 'mx')}")
GPUS = os.environ.get("MX_GPUS", "G4 A100 L4").split()
IDLE = float(os.environ.get("MX_IDLE_MIN", "30")) * 60
ALLOWED = {c.strip() for c in os.environ.get("TG_CHAT_ID", "").split(",") if c.strip()}
lock = threading.Lock()  # 同時只跑一個 up/down


def tg(method, **params):
    url = f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/{method}"
    with urllib.request.urlopen(url, urllib.parse.urlencode(params).encode(), timeout=70) as r:
        return json.load(r)["result"]


def say(chat, text):
    try:
        tg("sendMessage", chat_id=chat, text=text, disable_web_page_preview="true")
    except Exception as e:
        print("送訊息失敗：", e, file=sys.stderr)


def mx(*args, timeout=1800):
    p = subprocess.run([os.path.join(HERE, "mx-colab"), *args], capture_output=True, text=True, timeout=timeout)
    print(p.stdout[-2000:], p.stderr[-2000:], sep="\n", flush=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def endpoint():
    try:
        return open(EP).read().strip()
    except OSError:
        return ""


def jobs(url):
    key = open(os.path.join(CONF, "api_key")).read().strip()
    req = urllib.request.Request(f"{url}/v1/jobs", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def activity(prev, data):
    """回傳 (指紋, 是否有人在用)：有排隊／執行中的工作，或最新工作 id 變了，就算有人在用。"""
    items = data.get("data", [])
    fp = items[0].get("id") if items else None
    busy = any(j.get("status") in ("queued", "running") for j in items)
    return fp, busy or fp != prev


def do_up(chat):
    if not lock.acquire(blocking=False):
        return say(chat, "正在開機或關機中，請稍等")
    try:
        if endpoint():
            return say(chat, f"已經在線上：{endpoint()}")
        for gpu in GPUS:
            say(chat, f"搶 {gpu} 中…")
            try:
                rc, out = mx("up", "--gpu", gpu)
            except subprocess.TimeoutExpired:
                rc, out = 1, "逾時"
            if rc == 0 and endpoint():
                return say(chat, f"上線了（{gpu}）：{endpoint()}\n閒置 {IDLE / 60:.0f} 分鐘自動關機")
            say(chat, f"{gpu} 失敗：{out.splitlines()[-1] if out else '?'}")
            mx("down", timeout=300)  # 收掉開到一半的 VM，免得空燒點數
        say(chat, "全部 GPU 都搶不到，晚點再 /up")
    finally:
        lock.release()


def do_down(chat, why=""):
    with lock:
        rc, out = mx("down", timeout=600)
        if os.path.exists(EP):
            os.remove(EP)  # VM 已被回收時 mx-colab down 會提早結束，網址檔要自己清
        say(chat, f"{why}已關機" + ("" if rc == 0 else f"（mx-colab down 回報：{out.splitlines()[-1] if out else rc}）"))


def watchdog(chat_of):
    prev, last, fails = None, time.time(), 0
    while True:
        time.sleep(60)
        url = endpoint()
        if not url or lock.locked():
            prev, last, fails = None, time.time(), 0
            continue
        try:
            prev, active = activity(prev, jobs(url))
            fails = 0
            if active:
                last = time.time()
        except Exception as e:
            fails += 1
            print("查工作失敗：", e, file=sys.stderr)
        if fails >= 5:
            do_down(chat_of(), "連續 5 分鐘連不上 VM（可能被回收），")
        elif time.time() - last > IDLE:
            do_down(chat_of(), f"閒置 {IDLE / 60:.0f} 分鐘，")


def main():
    owner = [next(iter(ALLOWED), None)]
    threading.Thread(target=watchdog, args=(lambda: owner[0],), daemon=True).start()
    offset = 0
    while True:
        try:
            updates = tg("getUpdates", offset=offset, timeout=50)
        except Exception as e:
            print("getUpdates 失敗：", e, file=sys.stderr)
            time.sleep(5)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or {}
            chat, text = str(msg.get("chat", {}).get("id", "")), (msg.get("text") or "").split("@")[0].strip()
            if not chat:
                continue
            if chat not in ALLOWED:
                say(chat, f"你的 chat id 是 {chat}，把它設進 TG_CHAT_ID 才能操作")
                continue
            owner[0] = chat
            if text == "/up":
                threading.Thread(target=do_up, args=(chat,), daemon=True).start()
            elif text == "/down":
                threading.Thread(target=do_down, args=(chat,), daemon=True).start()
            elif text == "/status":
                say(chat, f"網址：{endpoint() or '（沒開）'}\n" + mx("status", timeout=60)[1][-1500:])
            else:
                say(chat, "指令：/up 開機（G4→A100→L4）、/status、/down")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        assert activity(None, {"data": []}) == (None, False)
        assert activity(None, {"data": [{"id": "a", "status": "done"}]}) == ("a", True)  # 新工作
        assert activity("a", {"data": [{"id": "a", "status": "done"}]}) == ("a", False)  # 沒變化 → 閒置
        assert activity("a", {"data": [{"id": "a", "status": "running"}]}) == ("a", True)  # 還在跑
        print("selftest ok")
    else:
        main()
