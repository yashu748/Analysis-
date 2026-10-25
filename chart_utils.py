"""
chart_utils.py - Telegram alert ke liye pro-style chart
Panels: Candles + EMA 9/26 (cross arrows + blue +), Entry/SL/TP boxes, Liq line,
        Volume, RSI(14)
DataFrame me ye columns chahiye: o, h, l, c, v  (optional: t = unix time)
"""
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def chart_png(df, pair, entry, sl, tp, tf="1h", leverage=30, bars=90, long=True):
    d = df.copy().reset_index(drop=True)
    d["e9"], d["e26"] = _ema(d.c, 9), _ema(d.c, 26)
    d["rsi"] = _rsi(d.c)
    d["up"] = (d.e9 > d.e26) & (d.e9.shift(1) <= d.e26.shift(1))
    d["dn"] = (d.e9 < d.e26) & (d.e9.shift(1) >= d.e26.shift(1))
    d = d.tail(bars).reset_index(drop=True)
    n = len(d)
    x = np.arange(n)

    bg, grid = "#0b0e14", "#1c212b"
    green, red, blue, orange = "#26a69a", "#ef5350", "#4a90ff", "#ff9800"
    fig = plt.figure(figsize=(8, 7), facecolor=bg)
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 1.2, 1.6], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    axr = fig.add_subplot(gs[2], sharex=ax)
    for a in (ax, axv, axr):
        a.set_facecolor(bg)
        a.tick_params(colors="#aaaaaa", labelsize=8)
        a.grid(color=grid, lw=0.5)
        for s in a.spines.values():
            s.set_color("#333333")
        a.yaxis.tick_right()

    # ---- Candles ----
    col = np.where(d.c >= d.o, green, red)
    ax.vlines(x, d.l, d.h, color=col, lw=1)
    body = np.maximum((d.c - d.o).abs(), (d.h - d.l) * 0.03)
    ax.bar(x, body, bottom=np.minimum(d.o, d.c), color=col, width=0.7)

    # ---- EMA 9 / 26 + cross markers ----
    ax.plot(x, d.e9, color=orange, lw=1.3, label="EMA 9")
    ax.plot(x, d.e26, color="#4caf50", lw=1.3, label="EMA 26")
    cross = d.up | d.dn
    ax.scatter(x[cross], d.e9[cross], marker="+", s=130, color=blue, lw=2.5, zorder=6)
    ax.scatter(x[d.up], d.l[d.up] - (d.h - d.l).mean() * 0.6, marker=r"$\uparrow$",
               s=170, color=green, zorder=6)
    ax.scatter(x[d.dn], d.h[d.dn] + (d.h - d.l).mean() * 0.6, marker=r"$\downarrow$",
               s=170, color=red, zorder=6)

    # ---- Entry / SL / TP boxes (position tool jaisa) ----
    right = n + 14
    ax.set_xlim(-1, right)
    ax.fill_between([n - 1, right], entry, tp, color=green, alpha=0.18)
    ax.fill_between([n - 1, right], entry, sl, color=red, alpha=0.18)
    ax.axhline(entry, color=blue, ls="--", lw=1)
    ax.axhline(tp, color=green, ls="--", lw=1)
    ax.axhline(sl, color=red, ls="--", lw=1)
    ax.text(right - 0.5, tp, f"TP {tp:.6g}", color=green, ha="right", va="bottom", fontsize=8)
    ax.text(right - 0.5, entry, f"Entry {entry:.6g}", color=blue, ha="right", va="bottom", fontsize=8)
    ax.text(right - 0.5, sl, f"SL {sl:.6g}", color=red, ha="right", va="top", fontsize=8)

    # ---- Liquidation line ----
    liq = entry * (1 - 0.99 / leverage) if long else entry * (1 + 0.99 / leverage)
    ax.axhline(liq, color="#f4a6a6", lw=1.2)
    ax.text(right - 0.5, liq, f"Liq {liq:.6g} ({leverage}x)", color="#f4a6a6",
            ha="right", va="top", fontsize=8)

    lo = min(d.l.min(), sl, liq)
    hi = max(d.h.max(), tp)
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    risk, reward = abs(entry - sl), abs(tp - entry)
    ax.set_title(f"{pair}  {tf}   R:R 1:{reward / risk:.1f}   "
                 f"SL {abs(sl / entry - 1) * 100:.1f}%  TP {abs(tp / entry - 1) * 100:.1f}%",
                 color="white", fontsize=10, loc="left")
    ax.legend(loc="upper left", facecolor=bg, labelcolor="white", fontsize=8, framealpha=0.6)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---- Volume ----
    axv.bar(x, d.v, color=col, width=0.7, alpha=0.8)
    axv.set_ylabel("Vol", color="#aaaaaa", fontsize=8)
    plt.setp(axv.get_xticklabels(), visible=False)

    # ---- RSI ----
    axr.plot(x, d.rsi, color=orange, lw=1.2)
    axr.axhline(70, color=red, ls="--", lw=0.8)
    axr.axhline(50, color="#888888", ls="--", lw=0.8)
    axr.axhline(30, color=green, ls="--", lw=0.8)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI 14", color="#aaaaaa", fontsize=8)
    axr.text(n - 1, d.rsi.iloc[-1], f" {d.rsi.iloc[-1]:.0f}", color=orange, fontsize=8, va="center")
    if "t" in d.columns:
        ts = pd.to_datetime(d.t, unit="s").dt.strftime("%d %b %H:%M")
        pos = x[::15]
        axr.set_xticks(pos)
        axr.set_xticklabels(ts.iloc[pos], rotation=0, fontsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
  
