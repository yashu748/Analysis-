"""
SMC trap alert bot -> Telegram.
  python smc_alert.py        # PAIRS scan karo, signal ho to Telegram par bhejo
  python smc_alert.py test   # sirf test message bhejo
Secrets (environment): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID. Optional: PAIRS="BTC_USDT,ETH_USDT"
"""
import os
import sys

import requests

from smc_trap import P, latest_signal, load_gate

PAIRS = [x.strip() for x in os.getenv("PAIRS", "BTC_USDT,ETH_USDT,SOL_USDT").split(",") if x.strip()]
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")


def send(text):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text}, timeout=20)
    r.raise_for_status()


def fmt(pair, s):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    return (f"{icon} {s['side']} {pair} (5m, SMC trap)\n"
            f"Entry: {s['entry']:.6g}\nSL: {s['sl']:.6g}\nTarget: {s['tp']:.6g}\n"
            f"RR 1:{P().rr:g}\nCandle: {s['time']} UTC")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send("✅ SMC alert bot: Telegram connect ho gaya")
        return
    for pair in PAIRS:
        try:
            df = load_gate(pair)
            for s in latest_signal(df):
                if s["kind"] == "SIGNAL":
                    send(fmt(pair, s))
        except Exception as e:  # ek pair fail ho to baaki chalte rahein
            print(pair, "error:", e)


if __name__ == "__main__":
    main()
