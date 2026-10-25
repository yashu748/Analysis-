import numpy as np

def _pivots(df, order=4):
    hi, lo = df["high"].values, df["low"].values
    ph, pl = [], []
    for i in range(order, len(df) - order):
        if hi[i] == hi[i-order:i+order+1].max(): ph.append(i)
        if lo[i] == lo[i-order:i+order+1].min(): pl.append(i)
    return ph, pl

def detect_pattern(df, order=4, tol=0.04, min_depth=0.05, lookback=80):
    df = df.tail(lookback).reset_index(drop=True)
    ph, pl = _pivots(df, order)
    close = df["close"].values
    last = len(df) - 1

    # Double Top (M) -> bearish breakout
    if len(ph) >= 2:
        a, b = ph[-2], ph[-1]
        h1, h2 = df["high"][a], df["high"][b]
        if abs(h1 - h2) / max(h1, h2) <= tol and b - a >= 5:
            neck_i = a + int(np.argmin(df["low"].values[a:b+1]))
            neck = df["low"][neck_i]
            if (min(h1, h2) - neck) / min(h1, h2) >= min_depth:
                broke = close[last] < neck and last > b
                return dict(type="DOUBLE TOP (M)", side="SELL", a=a, b=b, neck_i=neck_i,
                            p1=h1, p2=h2, neck=neck, broke=bool(broke), df=df,
                            sl=max(h1, h2), target=neck - (max(h1, h2) - neck))

    # Double Bottom (W) -> bullish breakout
    if len(pl) >= 2:
        a, b = pl[-2], pl[-1]
        l1, l2 = df["low"][a], df["low"][b]
        if abs(l1 - l2) / max(l1, l2) <= tol and b - a >= 5:
            neck_i = a + int(np.argmax(df["high"].values[a:b+1]))
            neck = df["high"][neck_i]
            if (neck - max(l1, l2)) / neck >= min_depth:
                broke = close[last] > neck and last > b
                return dict(type="DOUBLE BOTTOM (W)", side="BUY", a=a, b=b, neck_i=neck_i,
                            p1=l1, p2=l2, neck=neck, broke=bool(broke), df=df,
                            sl=min(l1, l2), target=neck + (neck - min(l1, l2)))
    return None

def draw_pattern(ax, pat, offset=0):
    """offset = len(full_df) - len(pat['df']) agar chart full df par bana hai"""
    a, b, n = pat["a"]+offset, pat["b"]+offset, pat["neck_i"]+offset
    c = "red" if pat["side"] == "SELL" else "lime"
    ax.plot([a, n, b], [pat["p1"], pat["neck"], pat["p2"]], color=c, lw=1.5)
    ax.axhline(pat["neck"], color=c, ls="--", lw=1)
    ax.text(a, pat["neck"], " Neckline", color=c, fontsize=8, va="bottom")

def pattern_text(pat):
    if not pat: return ""
    status = "✅ BREAKOUT HUA" if pat["broke"] else "⏳ Forming (breakout pending)"
    return (f"\n📐 Pattern: {pat['type']} — {status}\n"
            f"Neckline: {pat['neck']:.4f}\nPattern SL: {pat['sl']:.4f}\nPattern Target: {pat['target']:.4f}")
