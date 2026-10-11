"""
Live bot: team_scanner ko baar-baar chalata hai (default har 5 minute) aur
Telegram par SIRF naya ya badla hua setup bhejta hai (spam nahi).
sol_bot.py aur team_scanner.py ke saath ek hi folder me rakho.

ENV (optional):
  RUN_MINUTES  kitni der loop chale (default 345 = 5h45m, GitHub limit 6h)
  INTERVAL     scans ke beech seconds (default 300)
  SEND_FIRST   1 = pehle scan ke setups bhi bhejo, 0 = sirf baad ke badlav (default 1)
  TOP_N, CONSENSUS  team_scanner wale hi
Sirf alerts, advice nahi.
"""
import os
import time

import sol_bot as bot
import team_scanner as ts

RUN_MIN = float(os.getenv("RUN_MINUTES", "345"))
INTERVAL = int(os.getenv("INTERVAL", "300"))
SEND_FIRST = os.getenv("SEND_FIRST", "1") == "1"
MAX_IN_MSG = 6


def main():
    end = time.time() + RUN_MIN * 60
    state = {}  # pair -> last bias
    first = True
    while time.time() < end:
        t0 = time.time()
        try:
            res = ts.scan()
        except Exception as e:
            print("Scan error:", e)
            res = []

        new, dropped = [], []
        for r in res:
            prev = state.get(r["pair"])
            state[r["pair"]] = r["bias"]
            if r["bias"] != "WAIT" and r["bias"] != prev:
                new.append(r)
            elif prev in ("LONG", "SHORT") and r["bias"] == "WAIT":
                dropped.append(r["pair"])

        if first and not SEND_FIRST:
            new, dropped = [], []
        first = False

        if new or dropped:
            new.sort(key=lambda r: max(r["bull"], r["bear"]) / r["total"],
                     reverse=True)
            parts = []
            if new:
                parts.append(f"LIVE: {len(new)} naya/badla setup\n\n" +
                             "\n\n".join(ts.fmt(r) for r in new[:MAX_IN_MSG]))
            if dropped:
                parts.append("Consensus toota (setup khatam): " +
                             ", ".join(dropped[:15]))
            parts.append("SL = 1.5xATR(1h), T1 = 1.5R, T2 = 3R. "
                         "Advice nahi, SL ke bina trade mat lo.")
            msg = "\n\n".join(parts)
            print(msg)
            bot.send_telegram(msg)
        else:
            print("Koi naya badlav nahi.")

        sleep_for = max(30, INTERVAL - (time.time() - t0))
        if time.time() + sleep_for >= end:
            break
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
  
