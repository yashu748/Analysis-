"""
SMC "Smart Money ka Trap" strategy  (5-min timeframe)  -- tans_view reel

Reel ka idea (retail ko trap karna):
  1. Price trendline ko 3 baar touch karta hai -> retail 3rd touch par BUY karta hai,
     unka stop loss trendline ke neeche padha rehta hai (= liquidity).
  2. Smart money price ko trendline ke NEECHE le jaakar wo SL hit karta hai (liquidity sweep).
  3. Phir price tezi se wapas upar jaakar pichhla high todta hai (structure break).
  4. Jahan se reversal shuru hua wo DEMAND zone hai -> price zone me wapas aaye to BUY.
     SL = zone ke neeche, target = RR x risk.
  5. SELL side bilkul ulta (descending trendline, supply zone). Code SELL ke liye
     price ko ulta (invert) karke wahi BUY logic chalata hai.

Chalane ke tarike:
  python smc_trap.py demo              # banaye hue data par test
  python smc_trap.py data.csv          # CSV: datetime,open,high,low,close (5-min)
  python smc_trap.py gate BTC_USDT     # Gate.io public API se 5m candles (login nahi chahiye)

Bot me use karne ke liye:  from smc_trap import load_gate, latest_signal
"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class P:
    pivot: int = 3            # swing high/low: dono side kitni candles
    touch_tol: float = 0.30   # trendline touch tolerance (x ATR)
    sweep_atr: float = 0.20   # trendline kitni todni hai sweep ke liye (x ATR)
    max_span: int = 200       # pehle aur 3rd touch ke beech max candles
    wait_sweep: int = 60      # 3rd touch ke baad sweep ka intezaar (candles)
    wait_mss: int = 40        # sweep ke baad structure break ka intezaar
    wait_retest: int = 40     # structure break ke baad zone retest ka intezaar
    sl_atr: float = 0.10      # SL buffer (x ATR)
    rr: float = 2.0           # target = rr x risk
    max_hold: int = 60        # trade max kitni candles tak chalega
    atr_len: int = 14


def atr_series(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def invert(df):
    """Price ko ulta kar do: BUY logic se SELL setups mil jate hain."""
    return pd.DataFrame({"open": -df["open"], "high": -df["low"],
                         "low": -df["high"], "close": -df["close"]}, index=df.index)


def find_long(df, p):
    """BUY setups dhundta hai. Return: (setups, pending_zone_ya_None)."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr = atr_series(df, p.atr_len).to_numpy()
    n, k = len(df), p.pivot
    piv, setups, st = [], [], None

    for t in range(n):
        # swing low tab confirm hota hai jab uske baad k candles aa jayein (lookahead nahi)
        i, new_piv = t - k, False
        if i >= k and l[i] == l[i - k:i + k + 1].min() and (not piv or i - piv[-1] > k):
            piv.append(i)
            new_piv = True

        # --- Step 1: ascending trendline, 3 touches ---
        if st is None:
            if new_piv and len(piv) >= 3:
                a, b, c3 = piv[-3:]
                if l[a] < l[b] < l[c3] and c3 - a <= p.max_span:
                    slope = (l[c3] - l[a]) / (c3 - a)
                    tol = p.touch_tol * atr[t]
                    mid_ok = abs(l[b] - (l[a] + slope * (b - a))) <= tol
                    respected = all(l[x] >= l[a] + slope * (x - a) - tol for x in range(a, t + 1))
                    if mid_ok and respected:
                        st = dict(stage="sweep", a=a, slope=slope, t3=c3)
            continue

        line_t = l[st["a"]] + st["slope"] * (t - st["a"])

        # --- Step 2: liquidity sweep (trendline ke neeche) ---
        if st["stage"] == "sweep":
            if t - st["t3"] > p.wait_sweep:
                st = None
            elif l[t] < line_t - p.sweep_atr * atr[t]:
                st.update(stage="mss", sweep_idx=t, struct_hi=h[st["t3"]:t + 1].max(),
                          low=l[t], low_idx=t)

        # --- Step 3: structure break (pichhla high todna) ---
        elif st["stage"] == "mss":
            if l[t] < st["low"]:
                st["low"], st["low_idx"] = l[t], t
            if t - st["sweep_idx"] > p.wait_mss:
                st = None
            elif c[t] > st["struct_hi"]:
                j = st["low_idx"]                       # sweep ka sabse neecha candle
                zl = l[j]
                zh = min(max(o[j], c[j], zl + 0.3 * atr[j]), zl + 1.5 * atr[j])
                st.update(stage="retest", mss_idx=t, zl=zl, zh=zh)

        # --- Step 4: demand zone retest -> entry ---
        elif st["stage"] == "retest":
            if t - st["mss_idx"] > p.wait_retest or c[t] < st["zl"]:
                st = None                               # time out ya zone toot gaya
            elif l[t] <= st["zh"]:
                entry = min(st["zh"], o[t])
                sl = st["zl"] - p.sl_atr * atr[t]
                if entry > sl:
                    setups.append(dict(entry_idx=t, entry=entry, sl=sl,
                                       tp=entry + p.rr * (entry - sl),
                                       zl=st["zl"], zh=st["zh"],
                                       sweep_idx=st["sweep_idx"], mss_idx=st["mss_idx"]))
                st = None

    pending = st if st is not None and st["stage"] == "retest" else None
    return setups, pending


def simulate(df, s, p):
    """Entry ke baad SL/TP check. Ek hi candle me dono lage to SL pehle maana (conservative)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n, i0 = len(df), s["entry_idx"]
    end = min(n, i0 + p.max_hold)
    for j in range(i0, end):
        if l[j] <= s["sl"]:
            return j, s["sl"], "SL"
        if h[j] >= s["tp"]:
            return j, s["tp"], "TP"
    return end - 1, c[end - 1], ("OPEN" if i0 + p.max_hold > n else "TIME")


def run(df, p=P()):
    """Poora backtest: BUY + SELL dono side."""
    rows = []
    for side, sg in (("BUY", 1), ("SELL", -1)):
        d = df if sg == 1 else invert(df)
        setups, _ = find_long(d, p)
        for s in setups:
            j, px, why = simulate(d, s, p)
            rows.append(dict(side=side, time=df.index[s["entry_idx"]],
                             entry=sg * s["entry"], sl=sg * s["sl"], tp=sg * s["tp"],
                             exit_time=df.index[j], exit=sg * px, result=why,
                             pnl=px - s["entry"]))
    cols = ["side", "time", "entry", "sl", "tp", "exit_time", "exit", "result", "pnl"]
    return pd.DataFrame(rows, columns=cols).sort_values("time").reset_index(drop=True)


def latest_signal(df, p=P()):
    """Live bot ke liye: sirf aakhri band candle par bane signals + zone jahan wait chal raha hai."""
    out, n = [], len(df)
    for side, sg in (("BUY", 1), ("SELL", -1)):
        d = df if sg == 1 else invert(df)
        setups, pend = find_long(d, p)
        for s in setups:
            if s["entry_idx"] == n - 1:
                out.append(dict(kind="SIGNAL", side=side, time=df.index[-1],
                                entry=sg * s["entry"], sl=sg * s["sl"], tp=sg * s["tp"]))
        if pend:
            lo, hi = sorted((sg * pend["zl"], sg * pend["zh"]))
            out.append(dict(kind="WATCH", side=side, zone_low=lo, zone_high=hi))
    return out


# ---------------- data loaders ----------------
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [x.lower().strip() for x in df.columns]
    tcol = next(x for x in ("datetime", "date", "time", "timestamp") if x in df.columns)
    df.index = pd.to_datetime(df[tcol])
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def load_gate(pair="BTC_USDT", interval="5m", limit=1000):
    import requests
    r = requests.get("https://api.gateio.ws/api/v4/spot/candlesticks",
                     params=dict(currency_pair=pair, interval=interval, limit=limit), timeout=20)
    r.raise_for_status()
    rows = [x for x in r.json() if str(x[7]).lower() == "true"]   # sirf band candles
    df = pd.DataFrame(rows, columns=["t", "qv", "close", "high", "low", "open", "bv", "closed"])
    df.index = pd.to_datetime(df["t"].astype(int), unit="s")
    return df[["open", "high", "low", "close"]].astype(float)


def demo_df(seed=3):
    """Reel jaisa banaya hua scenario: 3 touch -> sweep -> structure break -> zone retest."""
    rng = np.random.default_rng(seed)
    pts = [(0, 110), (10, 100), (25, 110), (40, 103), (55, 112), (70, 106), (85, 113),
           (105, 104), (122, 114.5), (150, 104.3), (190, 120)]
    x = np.arange(pts[-1][0] + 1)
    c = np.interp(x, [a for a, _ in pts], [b for _, b in pts]) + rng.normal(0, 0.05, len(x))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 0.15, len(x))
    l = np.minimum(o, c) - rng.uniform(0, 0.15, len(x))
    idx = pd.date_range("2026-11-24 09:15", periods=len(x), freq="5min")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


if __name__ == "__main__":
    a = sys.argv[1:] or ["demo"]
    if a[0] == "demo":
        df = demo_df()
    elif a[0] == "gate":
        df = load_gate(a[1] if len(a) > 1 else "BTC_USDT")
    else:
        df = load_csv(a[0])

    tr = run(df)
    if tr.empty:
        print("Koi setup nahi mila.")
    else:
        done = tr[tr["result"] != "OPEN"]
        print(f"Trades: {len(tr)} | Win%: {(done['pnl'] > 0).mean() * 100:.1f} | "
              f"Total pts: {tr['pnl'].sum():.2f} | Avg pts: {tr['pnl'].mean():.2f}")
        print(tr.tail(15).to_string(index=False))
    for s in latest_signal(df):
        print("LIVE:", s)
