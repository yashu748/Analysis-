"""
backtest_algo.py - algo_signal ko Gate.io ke asli purane candles par check karta hai.
Batata hai: kitne trades, win-rate, average R, profit factor (fees ke baad).
Run: python backtest_algo.py        (ENV: TF=1h, COINS=60, PAGES=3, FEE=0.001)
"""
import os
import time
import requests
import pandas as pd

from algo_signal import add_signals

BASE = "https://api.gateio.ws/api/v4"
TF = os.getenv("TF", "1h")
COINS = int(os.getenv("COINS", "60"))
PAGES = int(os.getenv("PAGES", "3"))           # 1 page = 1000 candles
FEE = float(os.getenv("FEE", "0.001"))          # har taraf 0.1%
SEC = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}[TF]
MAX_HOLD = 48


def top_pairs():
    r = requests.get(f"{BASE}/spot/tickers", timeout=20).json()
    rows = []
    for t in r:
        p = t["currency_pair"]
        if not p.endswith("_USDT") or any(x in p for x in ("3L_", "3S_", "5L_", "5S_")):
            continue
        rows.append((p, float(t.get("quote_volume") or 0)))
    rows.sort(key=lambda x: -x[1])
    return [p for p, _ in rows[:COINS]]


def candles(pair):
    out, to = [], int(time.time())
    for _ in range(PAGES):
        r = requests.get(f"{BASE}/spot/candlesticks",
                         params={"currency_pair": pair, "interval": TF, "limit": 1000, "to": to},
                         timeout=20).json()
        if not isinstance(r, list) or not r:
            break
        out = r + out
        to = int(r[0][0]) - 1
        time.sleep(0.12)
    if len(out) < 200:
        return None
    df = pd.DataFrame(out, columns=["t", "qv", "c", "h", "l", "o", "v", "w"])
    for col in ["c", "h", "l", "o", "v"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("t").sort_values("t")
    if str(df["w"].iloc[-1]).lower() == "false":
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def simulate(df, rr):
    """Return list of R results. Entry = agli candle ka open. SL aur TP dono ek candle me aaye to SL maana."""
    res = []
    n = len(df)
    i = 60
    while i < n - 2:
        s = df.sig.iat[i]
        if s == 0:
            i += 1
            continue
        entry = df.o.iat[i + 1]
        sl = df.st_line.iat[i]
        risk = (entry - sl) * s
        if risk <= 0 or risk / entry < 0.003 or risk / entry > 0.12:
            i += 1
            continue
        tp = entry + s * rr * risk
        r_out, end = None, min(n - 1, i + 1 + MAX_HOLD)
        for j in range(i + 1, end + 1):
            lo, hi = df.l.iat[j], df.h.iat[j]
            hit_sl = lo <= sl if s == 1 else hi >= sl
            hit_tp = hi >= tp if s == 1 else lo <= tp
            if hit_sl:
                r_out = -1.0
                break
            if hit_tp:
                r_out = rr
                break
        if r_out is None:
            px = df.c.iat[end]
            r_out = (px - entry) * s / risk
            j = end
        cost = 2 * FEE * entry / risk          # fees ko R me badla
        res.append(r_out - cost)
        i = j + 1
    return res


def report(name, rs):
    if not rs:
        print(f"{name}: koi trade nahi")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    print(f"{name}: trades={len(rs)} win%={100*len(wins)/len(rs):.1f} "
          f"avgR={sum(rs)/len(rs):+.2f} profitFactor={pf:.2f}")


def main():
    pairs = top_pairs()
    print(f"TF={TF} coins={len(pairs)} pages={PAGES} fee/side={FEE}")
    data = {}
    for p in pairs:
        try:
            df = candles(p)
            if df is not None:
                data[p] = df
        except Exception as e:
            print("skip", p, repr(e)[:80])
    print(f"{len(data)} coins ka data mila\n")
    for filt in (False, True):
        print("=== FILTERS", "ON (EMA+RSI+Volume)" if filt else "OFF (sirf Supertrend flip)")
        sigs = {p: add_signals(df, use_filters=filt) for p, df in data.items()}
        for rr in (1.0, 1.5, 2.0, 3.0):
            allr = []
            for df in sigs.values():
                allr += simulate(df, rr)
            report(f"  TP={rr}R", allr)
        print()
    print("Dhyan: ye sirf purana data hai. Win% aur avgR dono dekho. avgR > 0 hi asli profit hai.")


if __name__ == "__main__":
    main()
