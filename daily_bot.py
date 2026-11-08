#!/usr/bin/env python3
"""
daily_bot.py - DAILY trend alert bot (spot, sirf LONG). backtest3 ke 4 jeetne wale tareeke:
  donch20 / donch55 : daily close pichhle 20 / 55 din ke high ke upar + close > EMA100.
                      SL = 3 ATR neeche, phir har din trailing (sabse ooncha close - 3 ATR). Max 40 din.
  mom3              : 20 din ke return me top-3 coin + close > EMA50. 5 din hold, SL 3 ATR.

Ye bot roz ek baar chalta hai (daily candle band hone ke baad, ~00:10 UTC) aur bhejta hai:
  * naye entry alerts, * aaj ke exits, * khule positions ka summary (naya trailing SL).
State file nahi chahiye: har baar pichhle 120 din ka strategy replay karke pata karta hai ki kya open hai.
(Backtest me ek coin me overlap trades allowed the, yahan ek coin+strategy me ek hi position.)

Run:  python daily_bot.py           (Telegram bhejta hai)
Test: python daily_bot.py --dry     (sirf print)
Secrets: TG_TOKEN, TG_CHAT_ID
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import chart_bot as b

DRY = "--dry" in sys.argv
TOP_N = int(os.getenv("DAILY_PAIRS", "40"))
FEE = 0.004                 # round trip (Gate spot 0.2% har taraf)
SL_ATR = 3.0
MAXHOLD_DONCH = 40
HOLD_MOM = 5
COOLDOWN = 5
WARM = 110
REPLAY = 120
RISK_PCT_TARGET = 1.0       # position size tip: ek trade me capital ka itna % risk
MIN_RISK, MAX_RISK = 0.004, 0.20   # SL distance 0.4%-20% ke andar ho (backtest me bhi yahi limit thi)


def prep(df):
    c = df.c
    return dict(
        o=df.o.values, h=df.h.values, l=df.l.values, c=c.values, t=df.t.values, atr=b.atr(df).values,
        ema50=b.ema(c, 50).values, ema100=b.ema(c, 100).values,
        dh20=df.h.rolling(20).max().shift(1).values, dh55=df.h.rolling(55).max().shift(1).values,
    )


def mask(m):
    m = np.nan_to_num(np.asarray(m, dtype=float), nan=0.0) > 0
    m[:WARM] = False
    return m


def replay(F, sig, kind):
    """Pichhle REPLAY din ka strategy chalao. Return (events_last_candle, open_position_or_None, stop_yesterday)."""
    n = len(F["c"])
    pos, last_sig = None, -10 ** 9
    events, prev_stop = [], None
    for i in range(max(WARM, n - REPLAY), n):
        if pos is not None:
            prev_stop = pos["stop"] if i == n - 1 else prev_stop
            ex = None
            if F["l"][i] <= pos["stop"]:
                ex, why = min(pos["stop"], F["o"][i]), "SL hit"
            else:
                pos["best"] = max(pos["best"], F["c"][i])
                if kind == "donch":
                    pos["stop"] = max(pos["stop"], pos["best"] - SL_ATR * F["atr"][i])
                    if i - pos["i"] >= MAXHOLD_DONCH:
                        ex, why = F["c"][i], "40 din poore"
                elif i - pos["i"] >= HOLD_MOM:
                    ex, why = F["c"][i], "5 din poore"
            if ex is not None:
                r = (ex - pos["entry"]) / pos["risk"] - FEE * pos["entry"] / pos["risk"]
                if i == n - 1:
                    events.append(dict(type="exit", why=why, price=ex, r=r, entry=pos["entry"]))
                pos = None
        if pos is None and sig[i] and i - last_sig >= COOLDOWN and F["atr"][i] > 0:
            entry = F["c"][i]
            risk = SL_ATR * F["atr"][i]
            last_sig = i
            if MIN_RISK <= risk / entry <= MAX_RISK:      # bahut door SL wale (pump) coin skip
                pos = dict(i=i, entry=entry, stop=entry - risk, risk=risk, best=entry)
                if i == n - 1:
                    events.append(dict(type="entry", price=entry, stop=pos["stop"]))
    return events, pos, prev_stop


def fmt(x):
    return b.fp(x)


def current_price(pair, fallback):
    r = b.get(f"{b.BASE}/spot/tickers", {"currency_pair": pair})
    try:
        return float(r[0]["last"])
    except Exception:
        return fallback


def main():
    t0 = time.time()
    pairs = [p for p in b.coin_list() if p != "BTC_USDT"][:TOP_N]
    print(f"Daily bot: {len(pairs)} coins", flush=True)
    DATA = {}
    for p in pairs:
        df = b.fetch(p, "1d", 300)
        if df is not None and len(df) > WARM + 30:
            DATA[p] = df
    PRE = {p: prep(df) for p, df in DATA.items()}
    closes = pd.DataFrame({p: DATA[p].set_index("t").c for p in DATA})
    rank = closes.pct_change(20).rank(axis=1, ascending=False)

    new_msgs, exit_msgs = [], []
    open_map = {}
    for p, F in PRE.items():
        mom_rank = rank[p].reindex(DATA[p].t.values).values
        strategies = [
            ("donch20", "donch", mask((F["c"] > F["dh20"]) & (F["c"] > F["ema100"]))),
            ("donch55", "donch", mask((F["c"] > F["dh55"]) & (F["c"] > F["ema100"]))),
            ("mom3", "mom", mask((mom_rank <= 3) & (F["c"] > F["ema50"]))),
        ]
        for name, kind, sig in strategies:
            events, pos, prev_stop = replay(F, sig, kind)
            nm = p.replace("_", "/")
            for ev in events:
                if ev["type"] == "entry":
                    cur = current_price(p, ev["price"])
                    risk_pct = (ev["price"] - ev["stop"]) / ev["price"] * 100
                    size = RISK_PCT_TARGET / risk_pct * 100
                    ext = "Exit: SL lage (SL roz upar khisakta hai) ya 40 din." if kind == "donch" \
                        else "Exit: 5 din baad ya SL lage."
                    warn = "\n\u26a0\ufe0f Price entry se 5% se upar hai, chase mat karo." if cur > ev["price"] * 1.05 else ""
                    new_msgs.append(
                        f"\U0001f7e2 DAILY LONG | {nm} | {name}\n"
                        f"Entry ref (kal ka close): {fmt(ev['price'])}\n"
                        f"Abhi price: {fmt(cur)} ({(cur / ev['price'] - 1) * 100:+.1f}%)\n"
                        f"SL: {fmt(ev['stop'])} (-{risk_pct:.1f}%)\n"
                        f"{ext}\n"
                        f"Size tip: {RISK_PCT_TARGET:.0f}% risk ke liye position = capital ka {size:.0f}%{warn}\n"
                        f"Alert hai, advice nahi. SL zaroor lagao.")
                else:
                    icon = "\u2705" if ev["r"] > 0 else "\u274c"
                    exit_msgs.append(f"{icon} EXIT {nm} | {name} | {ev['why']}\n"
                                     f"Entry {fmt(ev['entry'])} -> Exit {fmt(ev['price'])}  "
                                     f"({ev['r']:+.2f}R, fees ke baad)")
            if pos is not None and pos["i"] != len(F["c"]) - 1:
                key = (nm, pos["entry"], round(pos["stop"], 10))
                if key in open_map:
                    open_map[key]["names"].append(name)          # donch20 aur donch55 same trade ho to ek line
                else:
                    last = F["c"][-1]
                    chg = (last / pos["entry"] - 1) * 100
                    moved = " (SL upar gaya)" if prev_stop is not None and pos["stop"] > prev_stop * 1.0001 else ""
                    open_map[key] = dict(names=[name], text=f"entry {fmt(pos['entry'])}, ab {fmt(last)} ({chg:+.1f}%), "
                                                            f"SL {fmt(pos['stop'])}{moved}")

    day = time.strftime("%Y-%m-%d", time.gmtime(float(list(DATA.values())[0].t.iloc[-1]))) if DATA else "?"
    msgs = new_msgs + exit_msgs
    n_open = len(open_map)
    open_lines = [f"{k[0]} {'+'.join(v['names'])}: {v['text']}" for k, v in open_map.items()]
    if open_lines:
        msgs.append(f"OPEN POSITIONS ({n_open}), candle {day}\n" + "\n".join(open_lines[:25]))
    if not msgs:
        msgs.append(f"Daily bot ({day}): aaj koi naya signal ya exit nahi. Open positions: 0")
    for m in msgs:
        if DRY:
            print("\n" + m)
        else:
            b.tg_text(m[:3900])
            time.sleep(0.5)
    print(f"Done: {len(new_msgs)} entry, {len(exit_msgs)} exit, {n_open} open ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
  
