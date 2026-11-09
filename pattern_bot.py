#!/usr/bin/env python3
"""
pattern_bot.py - 1h chart analysis alert bot (screenshot jaisa chart + SAKHT pattern check)

Chart me: candles, EMA9/26, pattern ki lines + naam, candlestick pattern markers (blue +),
          Entry / SL / T1 / T2 zones, Liquidation (30x) line, Volume, RSI.
Caption : pair, side, pattern, 24h change, entry/SL/T1/T2, indicators, trend, patterns, candles, leverage warning.

"FAKE PATTERN HARD" - ye rules lage hain (sab aasaani se badal sakte ho, CONFIG dekho):
  * Swing point tabhi maana jata hai jab wo kam se kam 1.5 ATR ka asli swing ho (chhoti hilne-dulne nahi).
  * Double/Triple Top-Bottom: dono top ATR ke hisaab se barabar, beech me kam se kam 2.5 ATR ka dip,
    pehle 3 ATR ka trend, aur 8-70 candle ka gap.
  * Pattern tabhi valid jab neckline/trendline ke bahar CANDLE CLOSE ho chuka ho (breakout confirm).
    "forming" (abhi bana nahi) pattern ignore hota hai - pehle wali galtiyan yahin se aati thi.
  * Breakout 6 candle se purana ho ya price wapas neckline ke andar aa gaya ho to pattern fail.
  * Candlestick pattern (Evening/Morning Star, Engulfing, Hammer, Shooting Star) tabhi jab pehle ka trend ho.
  * Ulta CONFIRMED pattern ho to alert block.

Run:  python pattern_bot.py          (RUN_MINUTES tak loop, har 5 min scan)
Test: python pattern_bot.py --dry    (Telegram nahi bhejta)
Secrets: TG_TOKEN, TG_CHAT_ID
"""
import io
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import chart_bot as b

# ================= CONFIG =================
DRY = "--dry" in sys.argv
ALERT_TF = "1h"
TF_SEC = 3600
SCAN_EVERY = 300
RUN_MINUTES = float(os.getenv("RUN_MINUTES", "350"))
MAX_COINS = 120
MIN_CONFIRM = 4                    # 5 me se itne indicators
VOL_MULT = 1.5
MAX_ALERTS_PER_DAY = 6
MAX_ALERTS_PER_SCAN = 2
COOLDOWN_MIN = 720
MAX_SIGNAL_AGE_MIN = 15            # signal wali candle ko band hue 15 min se zyada na hua ho (restart par purane alert dobara nahi)
ATR_SL = 1.5
MAX_RISK_PCT = 10.0
T1_R, T2_R = 1.5, 3.0
LEV = 30                           # chart par Liq line kis leverage ki
MIN_QUALITY = 55
REQUIRE_PATTERN = False            # True = same side ka confirmed pattern ya candle pattern zaroori
BLOCK_OPPOSITE = True              # ulta confirmed pattern ho to alert nahi

# --- sakht pattern settings ---
SWING_K, SWING_PROM = 3, 1.5       # pivot ke dono taraf 3 candle, aur kam se kam 1.5 ATR ka swing
TOL_ATR, TOL_PCT = 0.6, 0.01       # do top/bottom barabar: 0.6 ATR aur 1% ke andar
SEP_MIN, SEP_MAX = 8, 70           # do top ke beech candles
DEPTH_ATR = 2.5                    # top aur neckline ke beech kam se kam itna ATR
PRIOR_ATR = 3.0                    # pattern se pehle itna ATR ka trend (30 candle me)
CONFIRM_ATR = 0.15                 # neckline ke bahar itne ATR par CLOSE = breakout
MAX_AGE = 6                        # breakout 6 candle se purana nahi
MIN_PATTERN_SCORE = 60

NAMES = {"Double Top": "Double Bottom", "Triple Top": "Triple Bottom",
         "Head & Shoulders": "Inverse H&S", "Descending Triangle": "Ascending Triangle",
         "Rising Wedge": "Falling Wedge"}

seen, last_alert, stats = {}, {}, {}
open_trades, tally, day_alerts = [], {"W": 0, "L": 0, "R": 0.0}, []


def rej(why):
    stats[why] = stats.get(why, 0) + 1
    return None


# ================= SWINGS =================
def swing_points(df, a, k=SWING_K, prom=SWING_PROM, w=10):
    h, l, n = df.h.values, df.l.values, len(df)
    hi, lo = [], []
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max() and h[i] - l[max(0, i - w):i + w + 1].min() >= prom * a[i]:
            hi.append((i, float(h[i])))
        if l[i] == l[i - k:i + k + 1].min() and h[max(0, i - w):i + w + 1].max() - l[i] >= prom * a[i]:
            lo.append((i, float(l[i])))

    def dedupe(p, is_hi):
        out = []
        for q in p:
            if out and q[0] - out[-1][0] <= k:
                if (q[1] > out[-1][1]) == is_hi:
                    out[-1] = q
            else:
                out.append(q)
        return out
    return dedupe(hi, True), dedupe(lo, False)


def fit(pts):
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    s, i0 = np.polyfit(x, y, 1)
    return s, i0, float(np.abs(y - (s * x + i0)).max())


# ================= BEARISH DETECTORS (view me; bullish = mirror) =================
def finish(v, name, line_fn, start, height, pts, lines, base):
    """Breakout CLOSE neeche ho chuka ho, recent ho, aur price wapas andar na aaya ho."""
    n, c, a = v["n"], v["c"], v["a"]
    j = None
    for k in range(start + 1, n):
        if c[k] < line_fn(k) - CONFIRM_ATR * a[k]:
            j = k
            break
    if j is None:                              # abhi forming - ignore
        return None
    if n - 1 - j > MAX_AGE:                    # purana breakout
        return None
    if c[-1] > line_fn(n - 1):                 # price wapas neckline ke andar aa gaya
        return None
    return dict(name=name, status="confirmed", age=n - 1 - j, score=base + 20,
                target=line_fn(j) - height, pts=pts, lines=lines)


def vol_at(v, i):
    return float(v["v"][max(0, i - 1):i + 2].mean())


def det_tops(v, P, T):
    out, n, a = [], v["n"], v["a"]
    if len(P) >= 3 and n - P[-1][0] <= 45:
        (i1, h1), (i2, h2), (i3, h3) = P[-3:]
        tops = [h1, h2, h3]
        if (min(i2 - i1, i3 - i2) >= SEP_MIN and (i3 - i1) <= SEP_MAX * 2
                and max(tops) - min(tops) <= TOL_ATR * a[i3] and (max(tops) - min(tops)) / max(tops) <= TOL_PCT):
            neck = float(v["l"][i1:i3 + 1].min())
            top = max(tops)
            prior = v["c"][i1] - v["c"][max(0, i1 - 30)]
            if top - neck >= DEPTH_ATR * a[i3] and prior >= PRIOR_ATR * a[i3]:
                r = finish(v, "Triple Top", lambda j, nk=neck: nk, i3, top - neck,
                           [(i1, h1), (i2, h2), (i3, h3)], [((i1, neck), (n - 1, neck))],
                           50 + (10 if vol_at(v, i3) <= vol_at(v, i1) * 1.2 else 0) + (10 if prior >= 5 * a[i3] else 0))
                if r:
                    return [r]
    if len(P) >= 2 and n - P[-1][0] <= 45:
        (i1, h1), (i2, h2) = P[-2:]
        top = max(h1, h2)
        if SEP_MIN <= i2 - i1 <= SEP_MAX and abs(h1 - h2) <= TOL_ATR * a[i2] and abs(h1 - h2) / top <= TOL_PCT:
            neck = float(v["l"][i1:i2 + 1].min())
            depth = top - neck
            prior = v["c"][i1] - v["c"][max(0, i1 - 30)]
            if depth >= DEPTH_ATR * a[i2] and prior >= PRIOR_ATR * a[i2]:
                r = finish(v, "Double Top", lambda j, nk=neck: nk, i2, depth,
                           [(i1, h1), (i2, h2)], [((i1, neck), (n - 1, neck))],
                           50 + (10 if depth >= 4 * a[i2] else 0) + (10 if vol_at(v, i2) <= vol_at(v, i1) * 1.2 else 0)
                           + (10 if prior >= 5 * a[i2] else 0))
                if r:
                    out.append(r)
    return out


def det_hs(v, P, T):
    n, a = v["n"], v["a"]
    if len(P) < 3 or n - P[-1][0] > 45:
        return []
    (iL, hL), (iH, hH), (iR, hR) = P[-3:]
    if min(iH - iL, iR - iH) < 5 or (iR - iL) > SEP_MAX * 2:
        return []
    if hH - max(hL, hR) < 1.0 * a[iR] or abs(hL - hR) > 1.0 * a[iR]:
        return []
    l = v["l"]
    t1 = iL + int(np.argmin(l[iL:iH + 1]))
    t2 = iH + int(np.argmin(l[iH:iR + 1]))
    slope = (l[t2] - l[t1]) / max(1, (t2 - t1))
    line = lambda j, t1=t1, s=slope: l[t1] + s * (j - t1)
    if hH - (l[t1] + l[t2]) / 2 < DEPTH_ATR * a[iR]:
        return []
    if v["c"][iL] - v["c"][max(0, iL - 30)] < 2.0 * a[iR]:
        return []
    r = finish(v, "Head & Shoulders", line, iR, hH - (l[t1] + l[t2]) / 2,
               [(iL, hL), (iH, hH), (iR, hR)], [((t1, float(l[t1])), (n - 1, float(line(n - 1))))],
               55 + (10 if vol_at(v, iR) <= vol_at(v, iH) else 0) + (10 if hH - max(hL, hR) >= 2 * a[iR] else 0))
    return [r] if r else []


def det_desc_tri(v, P, T):
    n, a = v["n"], v["a"]
    if len(P) < 3 or len(T) < 3:
        return []
    p3, t3 = P[-3:], T[-3:]
    first, last = min(p3[0][0], t3[0][0]), max(p3[-1][0], t3[-1][0])
    if not (20 <= last - first <= 90) or n - last > 30:
        return []
    lows = [q[1] for q in t3]
    if max(lows) - min(lows) > 0.8 * a[last] or not (p3[0][1] > p3[1][1] > p3[2][1]):
        return []
    if p3[0][1] - p3[2][1] < 1.5 * a[last]:
        return []
    if fit(p3)[2] > 0.8 * a[last]:
        return []
    neck = min(lows)
    r = finish(v, "Descending Triangle", lambda j, nk=neck: nk, last, p3[0][1] - neck, p3,
               [((t3[0][0], neck), (n - 1, neck))], 55)
    return [r] if r else []


def det_rising_wedge(v, P, T):
    n, a = v["n"], v["a"]
    if len(P) < 3 or len(T) < 3:
        return []
    p3, t3 = P[-3:], T[-3:]
    first, last = min(p3[0][0], t3[0][0]), max(p3[-1][0], t3[-1][0])
    if not (20 <= last - first <= 90) or n - last > 30:
        return []
    if not (p3[0][1] < p3[1][1] < p3[2][1] and t3[0][1] < t3[1][1] < t3[2][1]):
        return []
    sp, ip, rp = fit(p3)
    st, it, rt = fit(t3)
    if max(rp, rt) > 0.8 * a[last] or not (st > sp > 0):          # neeche ki line zyada tez chadhe = converge
        return []
    width0 = (sp * first + ip) - (st * first + it)
    if width0 < (sp * last + ip - (st * last + it)) * 1.2:
        return []
    r = finish(v, "Rising Wedge", lambda j: st * j + it, last, width0, p3 + [], 
               [((first, st * first + it), (n - 1, st * (n - 1) + it)),
                ((first, sp * first + ip), (n - 1, sp * (n - 1) + ip))], 55)
    return [r] if r else []


BEAR = (det_tops, det_hs, det_desc_tri, det_rising_wedge)


def make_view(df, a, hi, lo, mirror):
    if not mirror:
        return (dict(h=df.h.values, l=df.l.values, c=df.c.values, v=df.v.values, a=a, n=len(df)), list(hi), list(lo))
    return (dict(h=-df.l.values, l=-df.h.values, c=-df.c.values, v=df.v.values, a=a, n=len(df)),
            [(i, -p) for i, p in lo], [(i, -p) for i, p in hi])


def unmirror(p):
    q = dict(p)
    q["name"] = NAMES.get(p["name"], p["name"])
    q["side"] = "LONG"
    q["target"] = -p["target"]
    q["pts"] = [(i, -x) for i, x in p["pts"]]
    q["lines"] = [((i1, -y1), (i2, -y2)) for (i1, y1), (i2, y2) in p["lines"]]
    return q


# ================= NEUTRAL PATTERNS (breakout kisi bhi taraf) =================
def det_neutral(df, a, hi, lo):
    out, n = [], len(df)
    if len(hi) < 3 or len(lo) < 3:
        return out
    p3, t3 = hi[-3:], lo[-3:]
    first, last = min(p3[0][0], t3[0][0]), max(p3[-1][0], t3[-1][0])
    if not (20 <= last - first <= 90) or n - last > 30:
        return out
    sp, ip, rp = fit(p3)
    st, it, rt = fit(t3)
    if max(rp, rt) > 0.8 * a[last]:
        return out
    up = lambda j: sp * j + ip
    dn = lambda j: st * j + it
    width0, width1 = up(first) - dn(first), up(last) - dn(last)
    name = None
    if p3[0][1] > p3[1][1] > p3[2][1] and t3[0][1] < t3[1][1] < t3[2][1] and p3[0][1] - p3[2][1] >= a[last] \
            and t3[2][1] - t3[0][1] >= a[last] and width1 > 0 and width0 > width1 * 1.2:
        name = "Symmetrical Triangle"
    elif p3[0][1] < p3[1][1] < p3[2][1] and t3[0][1] > t3[1][1] > t3[2][1] and p3[2][1] - p3[0][1] >= a[last] \
            and t3[0][1] - t3[2][1] >= a[last] and width1 > width0 * 1.25:
        name = "Broadening (Megaphone)"
    if not name:
        return out
    c = df.c.values
    for j in range(last + 1, n):
        side = None
        if c[j] > up(j) + CONFIRM_ATR * a[j]:
            side = "LONG"
        elif c[j] < dn(j) - CONFIRM_ATR * a[j]:
            side = "SHORT"
        if side:
            if n - 1 - j > MAX_AGE:
                return out
            if (side == "LONG" and c[-1] < up(n - 1)) or (side == "SHORT" and c[-1] > dn(n - 1)):
                return out
            h0 = max(width0, width1)
            tgt = up(j) + h0 if side == "LONG" else dn(j) - h0
            out.append(dict(name=name, side=side, status="confirmed", age=n - 1 - j, score=60 + 20, target=tgt,
                            pts=p3 + t3, lines=[((first, up(first)), (n - 1, up(n - 1))),
                                                ((first, dn(first)), (n - 1, dn(n - 1)))]))
            return out
    return out


def detect_patterns(df, a):
    hi, lo = swing_points(df, a)
    res = []
    for mirror in (False, True):
        v, P, T = make_view(df, a, hi, lo, mirror)
        for fn in BEAR:
            for p in fn(v, P, T):
                p = unmirror(p) if mirror else dict(p, side="SHORT")
                res.append(p)
    res += det_neutral(df, a, hi, lo)
    res = [p for p in res if p["score"] >= MIN_PATTERN_SCORE]
    res.sort(key=lambda p: -p["score"])
    return res


# ================= CANDLESTICK PATTERNS =================
def candle_patterns(df, a):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    n, out = len(df), []
    body = np.abs(c - o)
    for i in range(10, n):
        at = a[i]
        up = c[i - 1] - c[i - 6] >= 1.5 * at
        dn = c[i - 6] - c[i - 1] >= 1.5 * at
        g = lambda k: c[k] > o[k]
        r = lambda k: c[k] < o[k]
        if up and g(i - 2) and body[i - 2] >= 0.8 * at and body[i - 1] <= 0.35 * body[i - 2] and r(i) \
                and c[i] < (o[i - 2] + c[i - 2]) / 2:
            out.append((i, "Evening Star", "SHORT"))
        if dn and r(i - 2) and body[i - 2] >= 0.8 * at and body[i - 1] <= 0.35 * body[i - 2] and g(i) \
                and c[i] > (o[i - 2] + c[i - 2]) / 2:
            out.append((i, "Morning Star", "LONG"))
        if up and g(i - 1) and r(i) and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] >= 1.0 * at:
            out.append((i, "Bearish Engulfing", "SHORT"))
        if dn and r(i - 1) and g(i) and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] >= 1.0 * at:
            out.append((i, "Bullish Engulfing", "LONG"))
        rng = h[i] - l[i]
        uw, lw = h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]
        if up and rng >= 1.0 * at and uw >= 2 * max(body[i], 1e-12) and lw <= 0.3 * max(body[i], 1e-12) + 0.1 * rng:
            out.append((i, "Shooting Star", "SHORT"))
        if dn and rng >= 1.0 * at and lw >= 2 * max(body[i], 1e-12) and uw <= 0.3 * max(body[i], 1e-12) + 0.1 * rng:
            out.append((i, "Hammer", "LONG"))
    return out


# ================= ANALYSIS =================
def analyze(pair):
    df = b.fetch(pair, ALERT_TF, 300)
    if df is None or len(df) < 150:
        return None
    ts = df.t.iloc[-1]
    if seen.get(pair) == ts:
        return None
    seen[pair] = ts
    if time.time() - (ts + TF_SEC) > MAX_SIGNAL_AGE_MIN * 60:
        return None
    c = df.c
    e9, e26 = b.ema(c, 9), b.ema(c, 26)
    st = b.supertrend(df)
    r = b.rsi(c)
    atr_s = b.atr(df)
    a = atr_s.values
    ml = b.ema(c, 12) - b.ema(c, 26)
    hist = ml - b.ema(ml, 9)
    vavg = df.v.iloc[-21:-1].mean()
    vr = df.v.iloc[-1] / vavg if vavg > 0 else 0
    R, price, at = r.iloc[-1], c.iloc[-1], a[-1]
    if np.isnan(R) or not at > 0:
        return None
    L = dict(EMA=e9.iloc[-1] > e26.iloc[-1] and price > e26.iloc[-1], Supertrend=st[-1] == 1,
             RSI=50 <= R <= 70, MACD=hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2], Volume=vr > VOL_MULT)
    S = dict(EMA=e9.iloc[-1] < e26.iloc[-1] and price < e26.iloc[-1], Supertrend=st[-1] == -1,
             RSI=30 <= R <= 50, MACD=hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2], Volume=vr > VOL_MULT)
    nl, ns = sum(L.values()), sum(S.values())
    side = None
    if nl >= MIN_CONFIRM and nl > ns:
        side, ind, conf = "LONG", L, nl
    elif ns >= MIN_CONFIRM and ns > nl:
        side, ind, conf = "SHORT", S, ns
    if not side:
        return None
    stats["[candidate] setup mila"] = stats.get("[candidate] setup mila", 0) + 1
    if time.time() - last_alert.get((pair, side), 0) < COOLDOWN_MIN * 60:
        return None
    want = 1 if side == "LONG" else -1
    tr = {tf: b.htf_trend(b.htf_df(pair, tf)) for tf in ("4h", "1d")}
    if tr["4h"] == -want and tr["1d"] == -want:
        return rej("4h aur 1d dono trend against")

    pats = detect_patterns(df, a)
    all_c = candle_patterns(df, a)
    cands = [x for x in all_c if x[0] >= len(df) - 3]
    opp = [p for p in pats if p["side"] != side]
    if BLOCK_OPPOSITE and opp:
        return rej(f"ulta confirmed pattern: {opp[0]['name']}")
    same = [p for p in pats if p["side"] == side]
    same_c = [x for x in cands if x[2] == side]
    if REQUIRE_PATTERN and not (same or same_c):
        return rej("same side pattern nahi mila")

    sw = df.tail(10)
    if side == "LONG":
        sl = min(price - ATR_SL * at, sw.l.min() - 0.25 * at)
    else:
        sl = max(price + ATR_SL * at, sw.h.max() + 0.25 * at)
    risk = abs(price - sl)
    if risk > price * MAX_RISK_PCT / 100:
        return rej("SL bahut door")
    if risk < price * 0.002:
        return rej("SL bahut paas")
    sgn = 1 if side == "LONG" else -1
    t1, t2 = price + sgn * T1_R * risk, price + sgn * T2_R * risk
    liq = price * (1 - 1 / LEV) if side == "LONG" else price * (1 + 1 / LEV)

    q = 40 + 10 * (conf == 5) + min(20, 10 * len(same)) + (5 if same_c else 0)
    q += 6 * sum(1 for v_ in tr.values() if v_ == want) + (5 if 2 <= vr <= 5 else 0)
    if q < MIN_QUALITY:
        return rej("quality kam")
    chg24 = (price / c.iloc[-25] - 1) * 100 if len(c) > 25 else 0.0
    return dict(pair=pair, side=side, price=price, sl=sl, t1=t1, t2=t2, liq=liq, risk_pct=risk / price * 100,
                ind=ind, conf=conf, rsi=R, tr=tr, pats=pats, cands=cands, all_cands=all_c,
                df=df, e9=e9, e26=e26, rsi_s=r, q=q, chg24=chg24, vr=vr)


# ================= CHART =================
def make_chart(s):
    df = s["df"]
    n_all = len(df)
    W = 100
    off = max(0, n_all - W)
    d = df.iloc[off:].reset_index(drop=True)
    n = len(d)
    bg = "#0d1117"
    fig, (ax, av, ar) = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor=bg,
                                     gridspec_kw={"height_ratios": [5, 1.2, 1.5], "hspace": 0.05})
    for a_ in (ax, av, ar):
        a_.set_facecolor(bg)
        a_.tick_params(colors="#9ca3af", labelsize=8)
        for sp in a_.spines.values():
            sp.set_color("#374151")
    for i, row in d.iterrows():
        col = "#26a69a" if row.c >= row.o else "#ef5350"
        ax.vlines(i, row.l, row.h, color=col, lw=1)
        ax.bar(i, max(abs(row.c - row.o), row.c * 1e-5), bottom=min(row.o, row.c), color=col, width=0.65)
        av.bar(i, row.v, color=col, width=0.65)
    ax.plot(range(n), s["e9"].values[off:], color="#f59e0b", lw=1, label="EMA 9")
    ax.plot(range(n), s["e26"].values[off:], color="#22c55e", lw=1, label="EMA 26")

    # candlestick pattern markers (blue +)
    for i, nm, sd in s["all_cands"]:
        x = i - off
        if 0 <= x < n:
            y = d.h.iloc[x] if sd == "SHORT" else d.l.iloc[x]
            ax.plot(x, y, marker="P", color="#60a5fa", ms=7, ls="none")
            if i >= n_all - 3:
                ax.text(x, y, " " + nm, color="#93c5fd", fontsize=7,
                        va="bottom" if sd == "SHORT" else "top")

    # chart patterns
    ymax = d.h.max()
    for k, p in enumerate(s["pats"][:3]):
        col = "#ef5350" if p["side"] == "SHORT" else "#26a69a"
        for (i1, y1), (i2, y2) in p["lines"]:
            ax.plot([i1 - off, i2 - off], [y1, y2], color=col, lw=1.2, alpha=0.9)
        xs = [i - off for i, _ in p["pts"] if i - off >= 0]
        ys = [y for i, y in p["pts"] if i - off >= 0]
        ax.plot(xs, ys, color=col, lw=0.9, ls=":", marker="o", ms=3)
        if xs:
            ax.text(min(xs), ymax * (1 + 0.004 * k), f"{p['name']} \u2714", color="#fbbf24", fontsize=8, va="bottom")

    lv = [("Entry", s["price"], "#60a5fa"), ("SL", s["sl"], "#ef5350"),
          ("T1", s["t1"], "#2dd4bf"), ("T2", s["t2"], "#2dd4bf"), (f"Liq {LEV}x", s["liq"], "#fb923c")]
    for name, val, col in lv:
        ax.axhline(val, color=col, ls="--", lw=1)
        ax.text(n + 0.5, val, f"{name} {b.fp(val)}", color=col, fontsize=8, va="center")
    ax.axhspan(s["price"], s["t2"], xmin=0.93, xmax=0.97, color="#26a69a", alpha=0.25)
    ax.axhspan(s["price"], s["sl"], xmin=0.93, xmax=0.97, color="#ef5350", alpha=0.25)
    lo_ = min(d.l.min(), s["sl"], s["t2"], s["liq"])
    hi_ = max(d.h.max(), s["sl"], s["t2"], s["liq"])
    pad = (hi_ - lo_) * 0.05
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.set_xlim(-1, n + 14)
    ax.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor="#9ca3af", edgecolor="#374151",
              bbox_to_anchor=(0, 0.93))
    ax.set_title(f"{s['pair']}  {ALERT_TF}  {s['side']}  confirm {s['conf']}/5   T1 {T1_R}R  T2 {T2_R}R",
                 color="#e5e7eb", fontsize=11, loc="left")
    av.set_ylabel("Vol", color="#9ca3af", fontsize=8)
    ar.plot(range(n), s["rsi_s"].values[off:], color="#fbbf24", lw=1)
    for y, col in ((30, "#2dd4bf"), (70, "#ef5350")):
        ar.axhline(y, color=col, ls="--", lw=0.8)
    ar.set_ylim(0, 100)
    ar.set_ylabel("RSI 14", color="#9ca3af", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def caption(s):
    ok = lambda k: "\u2705" if s["ind"][k] else "\u274c"
    arrow = {1: "\u2b06\ufe0f", -1: "\u2b07\ufe0f", 0: "\u2796"}
    top = s["pats"][0]["name"] if s["pats"] else (s["cands"][0][1] if s["cands"] else "indicators")
    sgn = "-" if s["side"] == "LONG" else "+"
    maxlev = max(1, int(0.8 / (s["risk_pct"] / 100)))
    t = [f"{s['pair']}  {s['side']} setup  (pattern: {top}, 24h {s['chg24']:+.1f}%)",
         f"Entry ~{b.fp(s['price'])}",
         f"SL {b.fp(s['sl'])}  ({sgn}{s['risk_pct']:.1f}%)",
         f"T1 {b.fp(s['t1'])} ({T1_R}R)  |  T2 {b.fp(s['t2'])} ({T2_R}R)", "",
         "Indicators: " + " ".join(f"{ok(k)}{k}" for k in s["ind"]) + f"  (RSI {s['rsi']:.0f})",
         "Trend: " + "  ".join(f"{tf} {arrow[v]}" for tf, v in s["tr"].items()) + f"   Quality {s['q']}/100"]
    if s["pats"]:
        t += ["", "\U0001f4d0 Patterns (breakout confirm):"]
        for p in s["pats"][:3]:
            dot = "\U0001f7e2" if p["side"] == "LONG" else "\U0001f534"
            t.append(f"{dot} {p['name']} - tgt {b.fp(p['target'])} ({p['age']} candle pehle break)")
    if s["cands"]:
        t += ["", "\U0001f56f Candle: " + ", ".join(f"{nm}" for _, nm, _ in s["cands"])]
    liq_before_sl = (s["side"] == "SHORT" and s["liq"] < s["sl"]) or (s["side"] == "LONG" and s["liq"] > s["sl"])
    t += ["", f"Leverage: SL se pehle liquidation na ho isliye max ~{maxlev}x."]
    if liq_before_sl:
        t.append(f"\u26a0\ufe0f {LEV}x par Liq {b.fp(s['liq'])} SL se pehle aa jayega, {LEV}x mat lena.")
    t += ["", "T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."]
    if s["side"] == "SHORT":
        t[-1] += " Short ke liye futures/perp chahiye."
    return "\n".join(t)


def send(s):
    cap = caption(s)
    if DRY or not b.TG_TOKEN or not b.TG_CHAT:
        print("[DRY]\n" + cap + "\n")
        return
    api = f"https://api.telegram.org/bot{b.TG_TOKEN}"
    try:
        img = make_chart(s)
        r = requests.post(f"{api}/sendPhoto", data={"chat_id": b.TG_CHAT, "caption": cap[:1024]},
                          files={"photo": ("chart.png", img)}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(r.text)
        if len(cap) > 1024:
            b.tg_text(cap)
    except Exception as e:
        print("photo failed:", e)
        b.tg_text(cap)


# ================= RESULT TRACKER =================
def track_results():
    now = time.time()
    for tr_ in list(open_trades):
        df = b.fetch(tr_["pair"], ALERT_TF, 120)
        if df is None:
            continue
        res = b.evaluate(tr_, df[df.t >= tr_["since"]])
        if res is None and now - tr_["since"] > 48 * 3600:
            res = ("expire", 0.0)
        if res is None:
            continue
        open_trades.remove(tr_)
        name, rv = res
        if name == "expire":
            b.tg_text(f"Expire: {tr_['pair']} {tr_['side']} - 48 ghante me na SL na T1 hit hua")
            continue
        tally["R"] += rv
        tally["W" if rv > 0 else "L"] += 1
        tot = tally["W"] + tally["L"]
        icon = "\u274c" if name == "SL" else "\u2705"
        b.tg_text(f"{icon} Result: {tr_['pair']} {tr_['side']} - {name} ({rv:+.2f}R)\n"
                  f"Total: {tally['W']} win / {tally['L']} loss ({tally['W'] / tot * 100:.0f}%), net {tally['R']:+.2f}R")


# ================= SCAN LOOP =================
def scan():
    stats.clear()
    pairs = b.coin_list()[:MAX_COINS]

    def work(p):
        try:
            return analyze(p)
        except Exception:
            traceback.print_exc()
            return None
    with ThreadPoolExecutor(6) as ex:
        res = [s for s in ex.map(work, pairs) if s]
    res.sort(key=lambda s: s["q"], reverse=True)
    now = time.time()
    day_alerts[:] = [t for t in day_alerts if now - t < 24 * 3600]
    room = max(0, MAX_ALERTS_PER_DAY - len(day_alerts))
    if res and room == 0:
        stats["daily limit poori"] = len(res)
    sent = 0
    for s in res[:min(MAX_ALERTS_PER_SCAN, room)]:
        last_alert[(s["pair"], s["side"])] = now
        day_alerts.append(now)
        send(s)
        sent += 1
        open_trades.append(dict(pair=s["pair"], side=s["side"], entry=s["price"], stop=s["sl"], t1=s["t1"],
                                t2=s["t2"], since=float(s["df"].t.iloc[-1]) + TF_SEC, t1_hit=False))
    if open_trades:
        try:
            track_results()
        except Exception:
            traceback.print_exc()
    blocked = ", ".join(f"{k}: {v}" for k, v in sorted(stats.items(), key=lambda x: -x[1]))
    print(time.strftime("%H:%M:%S"), f"scanned {len(pairs)} coins, alerts {sent} (24h: {len(day_alerts)}/{MAX_ALERTS_PER_DAY}),"
          f" open {len(open_trades)}" + (f" | {blocked}" if blocked else ""), flush=True)


def main():
    end = time.time() + (0.1 if DRY else RUN_MINUTES) * 60
    print(f"Pattern bot start: {ALERT_TF}, sakht pattern check, runs {RUN_MINUTES:.0f} min", flush=True)
    while True:
        t0 = time.time()
        try:
            scan()
        except Exception:
            traceback.print_exc()
        if time.time() >= end:
            break
        time.sleep(max(5, SCAN_EVERY - (time.time() - t0)))


if __name__ == "__main__":
    main()
