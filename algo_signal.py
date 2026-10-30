"""
algo_signal.py - "Algo Advance" jaisa LONG / SHORT / EXIT signal (Supertrend + EMA + RSI + Volume + ATR).
Sirf BAND (closed) candles use hote hain, isliye signal repaint nahi karta.
Signal candle ke close par banta hai, entry agli candle ke open par maani jaati hai.

Use:  df = add_signals(df)   # df columns: o h l c v
      df['sig'] = +1 LONG, -1 SHORT, 0 kuch nahi
"""
import numpy as np
import pandas as pd

ST_LEN = 10
ST_MULT = 3.0
EMA_TREND = 50
VOL_MULT = 1.2


def _atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=ST_LEN, mult=ST_MULT):
    atr = _atr(df, n).values
    h, l, c = df.h.values, df.l.values, df.c.values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1 and c[i] < fl[i]:
            d[i] = -1
        elif d[i - 1] == -1 and c[i] > fu[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    line = np.where(d == 1, fl, fu)
    return d, line


def add_signals(df, use_filters=True):
    df = df.copy().reset_index(drop=True)
    df["ema"] = df.c.ewm(span=EMA_TREND, adjust=False).mean()
    dlt = df.c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, 1e-9))
    df["atr"] = _atr(df)
    df["vavg"] = df.v.rolling(20).mean().shift(1)
    d, line = supertrend(df)
    df["st_dir"], df["st_line"] = d, line
    flip_up = (df.st_dir == 1) & (df.st_dir.shift() == -1)
    flip_dn = (df.st_dir == -1) & (df.st_dir.shift() == 1)
    long_ok = flip_up
    short_ok = flip_dn
    if use_filters:
        long_ok = long_ok & (df.c > df.ema) & (df.rsi.between(50, 72)) & (df.v > VOL_MULT * df.vavg)
        short_ok = short_ok & (df.c < df.ema) & (df.rsi.between(28, 50)) & (df.v > VOL_MULT * df.vavg)
    df["sig"] = 0
    df.loc[long_ok, "sig"] = 1
    df.loc[short_ok, "sig"] = -1
    return df


def trade_levels(row, side, rr1=1.5, rr2=3.0):
    """Entry = signal candle close (live me next open), SL = Supertrend line, T1/T2 = R multiples."""
    entry = row.c
    sl = row.st_line
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    if side == 1:
        return entry, sl, entry + rr1 * risk, entry + rr2 * risk
    return entry, sl, entry - rr1 * risk, entry - rr2 * risk
