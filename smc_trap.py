
#!/usr/bin/env python3
"""
SMC Smart Money Trap Strategy
15-Minute Timeframe

Features:
1. Three trendline touches
2. Liquidity sweep
3. Market Structure Shift (MSS)
4. Demand/Supply zone retest
5. BUY and SELL signals
6. Gate.io candle loader
7. CSV loader
8. Backtesting
9. Live signal detection

Import:
from smc_trap import load_gate, latest_signal
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================================================
# STRATEGY SETTINGS
# =========================================================

@dataclass
class P:
    pivot: int = 3
    touch_tol: float = 0.30
    sweep_atr: float = 0.20
    max_span: int = 200

    wait_sweep: int = 60
    wait_mss: int = 40
    wait_retest: int = 40

    sl_atr: float = 0.10
    min_rr: float = 2.0

    max_hold: int = 96
    atr_len: int = 14


# =========================================================
# DATA VALIDATION
# =========================================================

def prepare_df(df):

    if df is None or df.empty:
        raise ValueError("OHLC data empty hai.")

    df = df.copy()

    df.columns = [str(x).lower().strip() for x in df.columns]

    required = ["open", "high", "low", "close"]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    df = df[required].astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 20:
        raise ValueError("Strategy ke liye kam se kam 20 candles chahiye.")

    if (df["high"] < df["low"]).any():
        raise ValueError("Invalid candle: High, Low se chhota hai.")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("Invalid candle: High OHLC se match nahi karta.")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("Invalid candle: Low OHLC se match nahi karta.")

    return df


# =========================================================
# ATR INDICATOR
# =========================================================

def atr_series(df, n=14):

    pc = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(n, min_periods=1).mean()


# =========================================================
# SELL SIDE PRICE INVERSION
# =========================================================

def invert(df):

    return pd.DataFrame(
        {
            "open": -df["open"],
            "high": -df["low"],
            "low": -df["high"],
            "close": -df["close"]
        },
        index=df.index
    )


# =========================================================
# BUY SETUP DETECTION
# =========================================================

def find_long(df, p):

    df = prepare_df(df)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    atr = atr_series(df, p.atr_len).to_numpy()

    n = len(df)
    k = p.pivot

    piv = []
    setups = []
    st = None

    for t in range(n):

        # -------------------------------------------------
        # STEP 1: CONFIRM SWING LOW
        # -------------------------------------------------

        i = t - k
        new_piv = False

        if i >= k:

            window = l[i - k:i + k + 1]

            if (
                l[i] == window.min()
                and (not piv or i - piv[-1] > k)
            ):
                piv.append(i)
                new_piv = True

        # -------------------------------------------------
        # STEP 2: THREE TOUCH TRENDLINE
        # -------------------------------------------------

        if st is None:

            if new_piv and len(piv) >= 3:

                a, b, c3 = piv[-3:]

                if (
                    l[a] < l[b] < l[c3]
                    and c3 - a <= p.max_span
                ):

                    slope = (l[c3] - l[a]) / (c3 - a)

                    tol = p.touch_tol * atr[t]

                    expected_b = l[a] + slope * (b - a)

                    mid_ok = abs(l[b] - expected_b) <= tol

                    respected = all(
                        l[x] >= l[a] + slope * (x - a) - tol
                        for x in range(a, t + 1)
                    )

                    if mid_ok and respected:

                        st = {
                            "stage": "sweep",
                            "a": a,
                            "b": b,
                            "base": l[a],
                            "slope": slope,
                            "t3": c3
                        }

            continue

        line_t = (
            l[st["a"]]
            + st["slope"] * (t - st["a"])
        )

        # -------------------------------------------------
        # STEP 3: LIQUIDITY SWEEP
        # -------------------------------------------------

        if st["stage"] == "sweep":

            if t - st["t3"] > p.wait_sweep:

                st = None

            elif l[t] < line_t - p.sweep_atr * atr[t]:

                st.update(
                    {
                        "stage": "mss",
                        "sweep_idx": t,
                        "struct_hi": h[st["t3"]:t + 1].max(),
                        "low": l[t],
                        "low_idx": t
                    }
                )

        # -------------------------------------------------
        # STEP 4: MARKET STRUCTURE SHIFT
        # -------------------------------------------------

        elif st["stage"] == "mss":

            if l[t] < st["low"]:

                st["low"] = l[t]
                st["low_idx"] = t

            if t - st["sweep_idx"] > p.wait_mss:

                st = None

            elif c[t] > st["struct_hi"]:

                j = st["low_idx"]

                zl = l[j]

                zh = min(
                    max(
                        o[j],
                        c[j],
                        zl + 0.3 * atr[j]
                    ),
                    zl + 1.5 * atr[j]
                )

                st.update(
                    {
                        "stage": "retest",
                        "mss_idx": t,
                        "zl": zl,
                        "zh": zh
                    }
                )

        # -------------------------------------------------
        # STEP 5: DEMAND ZONE RETEST
        # -------------------------------------------------

        elif st["stage"] == "retest":

            if (
                t - st["mss_idx"] > p.wait_retest
                or c[t] < st["zl"]
            ):

                st = None

            elif l[t] <= st["zh"]:

                entry = min(st["zh"], o[t])

                sl = st["zl"] - p.sl_atr * atr[t]

                tp = h[st["mss_idx"]:t + 1].max()

                risk = entry - sl
                reward = tp - entry

                if (
                    risk > 0
                    and reward >= p.min_rr * risk
                ):

                    setups.append(
                        {
                            "entry_idx": t,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "zl": st["zl"],
                            "zh": st["zh"],
                            "sweep_idx": st["sweep_idx"],
                            "mss_idx": st["mss_idx"],
                            "a": st["a"],
                            "b": st["b"],
                            "t3": st["t3"],
                            "base": st["base"],
                            "slope": st["slope"],
                            "struct_hi": st["struct_hi"],
                            "low_idx": st["low_idx"]
                        }
                    )

                st = None

    pending = (
        st
        if st is not None and st["stage"] == "retest"
        else None
    )

    return setups, pending


# =========================================================
# TRADE SIMULATION
# =========================================================

def simulate(df, s, p):

    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    n = len(df)
    i0 = s["entry_idx"]

    end = min(n, i0 + p.max_hold)

    for j in range(i0, end):

        # Conservative: SL first if both hit in same candle

        if l[j] <= s["sl"]:

            return j, s["sl"], "SL"

        if h[j] >= s["tp"]:

            return j, s["tp"], "TP"

    if end <= i0:

        return i0, c[i0], "OPEN"

    result = "OPEN" if i0 + p.max_hold > n else "TIME"

    return end - 1, c[end - 1], result


# =========================================================
# COMPLETE BACKTEST BUY + SELL
# =========================================================

def run(df, p=None):

    if p is None:
        p = P()

    df = prepare_df(df)

    rows = []

    for side, sg in (
        ("BUY", 1),
        ("SELL", -1)
    ):

        d = df if sg == 1 else invert(df)

        setups, _ = find_long(d, p)

        for s in setups:

            j, px, why = simulate(d, s, p)

            pnl = px - s["entry"]

            rows.append(
                {
                    "side": side,
                    "time": df.index[s["entry_idx"]],
                    "entry": sg * s["entry"],
                    "sl": sg * s["sl"],
                    "tp": sg * s["tp"],
                    "exit_time": df.index[j],
                    "exit": sg * px,
                    "result": why,
                    "pnl": pnl
                }
            )

    cols = [
        "side",
        "time",
        "entry",
        "sl",
        "tp",
        "exit_time",
        "exit",
        "result",
        "pnl"
    ]

    result = pd.DataFrame(rows, columns=cols)

    return result.sort_values("time").reset_index(drop=True)


# =========================================================
# LIVE SIGNAL DETECTION
# =========================================================

def latest_signal(df, p=None):

    if p is None:
        p = P()

    df = prepare_df(df)

    out = []
    n = len(df)

    for side, sg in (
        ("BUY", 1),
        ("SELL", -1)
    ):

        d = df if sg == 1 else invert(df)

        setups, pend = find_long(d, p)

        for s in setups:

            if s["entry_idx"] == n - 1:

                out.append(
                    {
                        "kind": "SIGNAL",
                        "side": side,
                        "time": df.index[-1],
                        "entry": sg * s["entry"],
                        "sl": sg * s["sl"],
                        "tp": sg * s["tp"],
                        "detail": s,
                        "sg": sg
                    }
                )

        if pend:

            lo, hi = sorted(
                (
                    sg * pend["zl"],
                    sg * pend["zh"]
                )
            )

            out.append(
                {
                    "kind": "WATCH",
                    "side": side,
                    "zone_low": lo,
                    "zone_high": hi
                }
            )

    return out


# =========================================================
# CSV DATA LOADER
# =========================================================

def load_csv(path):

    df = pd.read_csv(path)

    df.columns = [
        str(x).lower().strip()
        for x in df.columns
    ]

    tcol = next(
        (
            x for x in (
                "datetime",
                "date",
                "time",
                "timestamp"
            )
            if x in df.columns
        ),
        None
    )

    if tcol is None:
        raise ValueError(
            "CSV mein datetime/date/time/timestamp column nahi mila."
        )

    df.index = pd.to_datetime(
        df[tcol],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=[tcol] if tcol in df.columns else [])

    return prepare_df(df)


# =========================================================
# GATE.IO DATA LOADER
# =========================================================

def load_gate(
    pair="BTC_USDT",
    interval="15m",
    limit=1000
):

    import requests

    url = (
        "https://api.gateio.ws/api/v4/"
        "spot/candlesticks"
    )

    response = requests.get(
        url,
        params={
            "currency_pair": pair,
            "interval": interval,
            "limit": min(int(limit), 1000)
        },
        timeout=20
    )

    response.raise_for_status()

    rows = response.json()

    if not rows:

        raise ValueError(
            "Gate.io se candle data nahi mila."
        )

    # Gate.io spot candle format:
    # timestamp, volume, close, high, low, open, quote_volume

    df = pd.DataFrame(
        rows,
        columns=[
            "t",
            "qv",
            "close",
            "high",
            "low",
            "open",
            "bv"
        ]
    )

    df.index = pd.to_datetime(
        df["t"].astype(int),
        unit="s",
        utc=True
    )

    df = df[
        ["open", "high", "low", "close"]
    ].astype(float)

    # Current incomplete candle remove

    now = pd.Timestamp.now(tz="UTC")

    candle_duration = pd.Timedelta(interval)

    df = df[
        df.index + candle_duration <= now
    ]

    return prepare_df(df)


# =========================================================
# DEMO DATA
# =========================================================

def demo_df(seed=3):

    rng = np.random.default_rng(seed)

    pts = [
        (0, 110),
        (10, 100),
        (25, 110),
        (40, 103),
        (55, 112),
        (70, 106),
        (85, 113),
        (105, 104),
        (122, 114.5),
        (150, 104.3),
        (190, 120)
    ]

    x = np.arange(
        pts[-1][0] + 1
    )

    c = np.interp(
        x,
        [a for a, _ in pts],
        [b for _, b in pts]
    )

    c += rng.normal(
        0,
        0.05,
        len(x)
    )

    o = np.r_[
        c[0],
        c[:-1]
    ]

    h = np.maximum(o, c) + rng.uniform(
        0,
        0.15,
        len(x)
    )

    l = np.minimum(o, c) - rng.uniform(
        0,
        0.15,
        len(x)
    )

    idx = pd.date_range(
        "2026-11-24 09:15",
        periods=len(x),
        freq="15min"
    )

    return pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": l,
            "close": c
        },
        index=idx
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    args = sys.argv[1:] or ["demo"]

    try:

        if args[0] == "demo":

            df = demo_df()

        elif args[0] == "gate":

            pair = (
                args[1]
                if len(args) > 1
                else "BTC_USDT"
            )

            df = load_gate(
                pair=pair,
                interval="15m",
                limit=1000
            )

        else:

            df = load_csv(args[0])

        print("\n========== SMC STRATEGY ==========")

        print("Candles:", len(df))
        print("Symbol:", args[1] if args[0] == "gate" and len(args) > 1 else args[0])

        trades = run(df)

        if trades.empty:

            print("\nKoi setup nahi mila.")

        else:

            done = trades[
                trades["result"].isin(["SL", "TP"])
            ]

            wins = (
                done["pnl"] > 0
            ).sum()

            win_rate = (
                wins / len(done) * 100
                if len(done) > 0
                else 0
            )

            print("\n========== BACKTEST ==========")

            print("Total trades:", len(trades))
            print("Completed trades:", len(done))
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Total points: {trades['pnl'].sum():.4f}")
            print(f"Average points: {trades['pnl'].mean():.4f}")

            print("\n========== LAST 15 TRADES ==========")

            print(
                trades.tail(15).to_string(
                    index=False
                )
            )

        print("\n========== LIVE SIGNALS ==========")

        signals = latest_signal(df)

        if not signals:

            print("Abhi koi LIVE signal ya WATCH zone nahi hai.")

        for signal in signals:

            print("\nLIVE:", signal)

        print("\n========== COMPLETE ==========")

    except Exception as e:

        print("\nERROR:", str(e))

        raise
              
