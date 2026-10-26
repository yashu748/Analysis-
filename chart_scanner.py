"""
Chart scanner: Gate.io coins jinme khareedari + trend upar ki taraf hai, unka analysis karke
chart ki photo ke saath Telegram alert bhejta hai. Ab saare chart + candlestick patterns bhi.
buy_pressure_scanner.py, sol_bot.py, team_scanner.py ke saath ek hi folder me rakho.
(pattern_detector.py alag se nahi chahiye, sab is file me hai.)

Pipeline:
 1) buy_pressure_scanner ka screening (liquid coins, volume spike, CMF, OBV, trend,
    taker buying %, order book)
 2) Chart structure se levels: SL swing low ke neeche (ATR ke andar clamp), targets agle
    resistance levels par (nahi mile to 1.5R / 3R)
 3) Resistance bilkul upar ho to setup skip (R:R kharab)
 4) Saare patterns detect (Double/Triple Top-Bottom, H&S, Triangles, Wedges, Flags,
    Cup&Handle, Channels, Candlesticks). Bearish breakdown ho chuka ho to long skip.
 5) Alag se pattern scan: bullish breakout -> LONG, bearish breakdown -> SHORT alert
 6) Chart image (candles, EMA 9/26 cross arrows, Entry/SL/T1/T2 boxes, Liq line,
    volume, RSI, patterns) + caption Telegram par

ENV (optional): MIN_SCORE (6), MIN_BUY_RATIO (0.58), COOLDOWN_H (6), MAX_ALERTS (5),
                LEVERAGE (30), PATTERN_ONLY (1 = buy-pressure alerts me bhi bullish pattern zaroori),
                MAX_PAT_ALERTS (5), MIN_CONFIRM_PAT (3), PAT_BREAKOUT_ONLY (1), PAT_SCAN_LIMIT (200),
                HTF_FILTER (1), BTC_FILTER (1), EXT_ATR (2.5)  <- ye 3 fail-trades kam karte hain
Ye rule-based analysis hai, live AI model nahi. Advice nahi.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402

import sol_bot as bot  # noqa: E402
import buy_pressure_scanner as bp  # noqa: E402

MAX_ALERTS = int(os.getenv("MAX_ALERTS", "5"))
STATE_FILE = "alert_state_chart.json"
CHART_BARS = 80
MIN_RISK_PCT = float(os.getenv("MIN_RISK_PCT", "0.6")) / 100   # SL isse chhota ho to fees kha jati hain
MIN_CONFIRM = int(os.getenv("MIN_CONFIRM", "4"))
LEVERAGE = float(os.getenv("LEVERAGE", "30"))                  # chart par Liq line ke liye
PATTERN_ONLY = os.getenv("PATTERN_ONLY", "0") == "1"
MAX_PAT_ALERTS = int(os.getenv("MAX_PAT_ALERTS", "5"))        # pattern wale LONG+SHORT alerts
MIN_CONFIRM_PAT = int(os.getenv("MIN_CONFIRM_PAT", "3"))      # pattern trade ke liye indicator confirm
PAT_BREAKOUT_ONLY = os.getenv("PAT_BREAKOUT_ONLY", "1") == "1"  # 1 = sirf breakout ho chuke pattern
PAT_SCAN_LIMIT = int(os.getenv("PAT_SCAN_LIMIT", "200"))      # kitne coins pattern ke liye scan
HTF_FILTER = os.getenv("HTF_FILTER", "1") == "1"              # 4h trend ulta ho to alert nahi
BTC_FILTER = os.getenv("BTC_FILTER", "1") == "1"              # BTC 1h trend ulta ho to alert nahi
EXT_ATR = float(os.getenv("EXT_ATR", "2.5"))                  # price EMA20 se itne ATR door = late entry, skip

# ================= PATTERN DETECTION =================
def _pivots(h, l, order=3):
    ph, pl = [], []
    for i in range(order, len(h) - order):
        if h[i] == max(h[i - order:i + order + 1]) and h[i] > max(h[i - order:i]):
            ph.append(i)
        if l[i] == min(l[i - order:i + order + 1]) and l[i] < min(l[i - order:i]):
            pl.append(i)
    return ph, pl


def _fit(pts):
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    d = n * sxx - sx * sx
    m = (n * sxy - sx * sy) / d if d else 0.0
    return m, (sy - m * sx) / n


def _pat(name, bias, broke, lines, neck=None, target=None, kind="chart"):
    return dict(name=name, bias=bias, broke=bool(broke), lines=lines,
                neck=neck, target=target, kind=kind)


def _argmin(seq, a, b):
    return a + min(range(b - a + 1), key=lambda k: seq[a + k])


# ---------- "top" type patterns (bearish). Bottoms = inverted data se milte hain ----------
def _top_patterns(H, L, C, ph, last, tol, depth):
    out = []
    # Triple Top
    triple = False
    if len(ph) >= 3:
        a, b, c3 = ph[-3:]
        hs = [H[a], H[b], H[c3]]
        if (max(hs) - min(hs) <= tol * max(hs) and b - a >= 4 and c3 - b >= 4
                and last - c3 <= 25):
            t1, t2 = _argmin(L, a, b), _argmin(L, b, c3)
            neck, top = min(L[t1], L[t2]), max(hs)
            if (min(hs) - neck) / min(hs) >= depth:
                triple = True
                out.append(_pat("Triple Top", "BEAR", C[-1] < neck,
                                [(a, H[a], t1, L[t1], 0), (t1, L[t1], b, H[b], 0),
                                 (b, H[b], t2, L[t2], 0), (t2, L[t2], c3, H[c3], 0),
                                 (a, neck, last + 2, neck, 1)],
                                neck, neck - (top - neck)))
    # Double Top
    if len(ph) >= 2 and not triple:
        a, b = ph[-2:]
        h1, h2 = H[a], H[b]
        if b - a >= 5 and last - b <= 25 and abs(h1 - h2) / max(h1, h2) <= tol:
            t = _argmin(L, a, b)
            neck, top = L[t], max(h1, h2)
            if (min(h1, h2) - neck) / min(h1, h2) >= depth:
                out.append(_pat("Double Top", "BEAR", C[-1] < neck,
                                [(a, h1, t, neck, 0), (t, neck, b, h2, 0),
                                 (a, neck, last + 2, neck, 1)],
                                neck, neck - (top - neck)))
    # Head & Shoulders
    if len(ph) >= 3:
        a, b, c3 = ph[-3:]
        if (H[b] > H[a] * 1.02 and H[b] > H[c3] * 1.02 and b - a >= 3 and c3 - b >= 3
                and last - c3 <= 25
                and abs(H[a] - H[c3]) / max(H[a], H[c3]) <= 0.06):
            t1, t2 = _argmin(L, a, b), _argmin(L, b, c3)
            if t2 != t1:
                m = (L[t2] - L[t1]) / (t2 - t1)

                def ny(x):
                    return L[t1] + m * (x - t1)
                hgt = H[b] - ny(b)
                if hgt / H[b] >= 0.04:
                    now = ny(last)
                    out.append(_pat("Head & Shoulders", "BEAR", C[-1] < now,
                                    [(a, H[a], t1, L[t1], 0), (t1, L[t1], b, H[b], 0),
                                     (b, H[b], t2, L[t2], 0), (t2, L[t2], c3, H[c3], 0),
                                     (t1, L[t1], last + 2, ny(last + 2), 1)],
                                    now, now - hgt))
    # Bear Flag (pole neeche, phir halka upar-side consolidation)
    for f in range(6, 15):
        if len(C) < f + 12:
            break
        p0, p1 = C[-f - 11], C[-f - 1]
        pole = (p0 - p1) / p0
        sh, sl = max(H[-f:]), min(L[-f:])
        if (pole >= 0.08 and (sh - sl) / p1 <= pole * 0.6
                and p1 * 0.99 <= C[-1] <= p1 + 0.6 * (p0 - p1)):
            lo_prev = min(L[-f:-1])
            out.append(_pat("Bear Flag", "BEAR", C[-1] < lo_prev,
                            [(last - f - 10, p0, last - f, p1, 0),
                             (last - f + 1, sh, last + 2, sh, 0),
                             (last - f + 1, sl, last + 2, sl, 0)],
                            lo_prev, lo_prev - (p0 - p1)))
            break
    return out


_FLIP = {"Triple Top": "Triple Bottom", "Double Top": "Double Bottom",
         "Head & Shoulders": "Inverse Head & Shoulders", "Bear Flag": "Bull Flag"}


def _flip(p, K):
    p["name"] = _FLIP[p["name"]]
    p["bias"] = "BULL"
    p["lines"] = [(x1, K - y1, x2, K - y2, d) for x1, y1, x2, y2, d in p["lines"]]
    p["neck"] = K - p["neck"]
    p["target"] = K - p["target"]
    return p


# ---------- triangles / wedges / channels / rectangle ----------
def _trend_shapes(H, L, C, ph, pl, last, flat_th=0.10):
    uh = [i for i in ph if i >= last - 60][-3:]
    lh = [i for i in pl if i >= last - 60][-3:]
    if len(uh) < 2 or len(lh) < 2 or len(uh) + len(lh) < 5:
        return []
    x0 = min(uh[0], lh[0])
    if last - x0 < 12:
        return []
    mu, bu = _fit([(i, H[i]) for i in uh])
    ml, bl = _fit([(i, L[i]) for i in lh])

    def up(x):
        return mu * x + bu

    def lo(x):
        return ml * x + bl
    gap0, gap1 = up(x0) - lo(x0), up(last) - lo(last)
    if gap0 <= 0 or gap1 <= 0:
        return []
    if C[-1] > up(last) * 1.05 or C[-1] < lo(last) * 0.95:
        return []
    pm = sum(C[-30:]) / len(C[-30:])
    su, sl_ = mu / pm * 100, ml / pm * 100

    def tag(s):
        return "up" if s > flat_th else ("dn" if s < -flat_th else "flat")
    tu, tl = tag(su), tag(sl_)
    conv, div = gap1 < 0.8 * gap0, gap1 > 1.2 * gap0
    name = bias = None
    if conv:
        if tu == "flat" and tl == "up":
            name, bias = "Ascending Triangle", "BULL"
        elif tu == "dn" and tl == "flat":
            name, bias = "Descending Triangle", "BEAR"
        elif tu == "dn" and tl == "up":
            name, bias = "Symmetrical Triangle", "NEUTRAL"
        elif tu == "up" and tl == "up":
            name, bias = "Rising Wedge", "BEAR"
        elif tu == "dn" and tl == "dn":
            name, bias = "Falling Wedge", "BULL"
    elif div and tu == "up" and tl == "dn":
        name, bias = "Broadening (Megaphone)", "NEUTRAL"
    else:
        if tu == "flat" and tl == "flat":
            name, bias = "Rectangle (Range)", "NEUTRAL"
        elif tu == "up" and tl == "up":
            name, bias = "Up Channel", "BULL"
        elif tu == "dn" and tl == "dn":
            name, bias = "Down Channel", "BEAR"
    if not name:
        return []
    bu_, bd_ = C[-1] > up(last), C[-1] < lo(last)
    broke = bu_ or bd_
    if broke:
        bias = "BULL" if bu_ else "BEAR"
    neck = lo(last) if bd_ else up(last)
    if bu_ or (not broke and bias == "BULL"):
        target = up(last) + gap0
    elif bd_ or (not broke and bias == "BEAR"):
        target = lo(last) - gap0
    else:
        target = None
    if name.endswith("Channel"):
        target = None
    lines = [(x0, up(x0), last + 2, up(last + 2), 0),
             (x0, lo(x0), last + 2, lo(last + 2), 0)]
    return [_pat(name, bias, broke, lines, neck, target)]


# ---------- Cup & Handle ----------
def _cup(H, L, C, last):
    n = len(H)
    if n < 50:
        return []
    s0 = n - 60 if n >= 60 else 0
    a = s0 + max(range(0, 25), key=lambda k: H[s0 + k])
    if a + 8 >= n - 8:
        return []
    bot = _argmin(L, a + 8, n - 9)
    if bot + 5 >= n:
        return []
    r = bot + 5 + max(range(0, n - bot - 5), key=lambda k: H[bot + 5 + k])
    if r >= last or r - a < 20:
        return []
    if abs(H[r] - H[a]) / H[a] > 0.06:
        return []
    depth = (H[a] - L[bot]) / H[a]
    frac = (bot - a) / (r - a)
    if depth < 0.12 or not 0.3 <= frac <= 0.7 or last - r > 12:
        return []
    pb = (H[r] - min(L[r:])) / (H[r] - L[bot])
    if pb > 0.5:
        return []
    rim = max(H[a], H[r])
    return [_pat("Cup & Handle", "BULL", C[-1] > rim,
                 [(a, H[a], bot, L[bot], 0), (bot, L[bot], r, H[r], 0),
                  (a, rim, last + 2, rim, 1)],
                 rim, rim + (rim - L[bot]))]


# ---------- candlestick patterns (last closed candle) ----------
def _candles(O, H, L, C):
    n = len(C)
    if n < 8:
        return []
    last = n - 1
    o, h, l, c = O[-1], H[-1], L[-1], C[-1]
    body, rng = abs(c - o), h - l
    if rng <= 0:
        return []
    upper, lower = h - max(o, c), min(o, c) - l
    down, upr = C[-1] < C[-6], C[-1] > C[-6]
    out = []

    def add(name, bias):
        out.append(dict(name=name, bias=bias, broke=False, lines=[], neck=None,
                        target=None, kind="candle", x=last, y=h))
    if body > 0 and lower >= 2 * body and upper <= 0.5 * body and down:
        add("Hammer", "BULL")
    if body > 0 and upper >= 2 * body and lower <= 0.5 * body and upr:
        add("Shooting Star", "BEAR")
    if body <= 0.1 * rng:
        add("Doji", "NEUTRAL")
    o1, c1 = O[-2], C[-2]
    if c1 < o1 and c > o and o <= c1 and c >= o1:
        add("Bullish Engulfing", "BULL")
    if c1 > o1 and c < o and o >= c1 and c <= o1:
        add("Bearish Engulfing", "BEAR")
    o2, c2 = O[-3], C[-3]
    b2, b1 = abs(c2 - o2), abs(c1 - o1)
    if b2 > 0 and c2 < o2 and b1 <= 0.3 * b2 and c > o and c > (o2 + c2) / 2:
        add("Morning Star", "BULL")
    if b2 > 0 and c2 > o2 and b1 <= 0.3 * b2 and c < o and c < (o2 + c2) / 2:
        add("Evening Star", "BEAR")
    if (C[-3] > O[-3] and C[-2] > O[-2] and c > o and C[-2] > C[-3] and c > C[-2]
            and O[-2] > O[-3] and o > O[-2] and body >= 0.5 * rng):
        add("Three White Soldiers", "BULL")
    if (C[-3] < O[-3] and C[-2] < O[-2] and c < o and C[-2] < C[-3] and c < C[-2]
            and O[-2] < O[-3] and o < O[-2] and body >= 0.5 * rng):
        add("Three Black Crows", "BEAR")
    return out


# ---------- main entry ----------
def detect_all(O, H, L, C, s, tol=0.04, depth=0.05, order=3):
    """s = chart window ka start index. Saare x-indices window ke hisaab se hain."""
    O, H, L, C = O[s:], H[s:], L[s:], C[s:]
    if len(C) < 30:
        return []
    last = len(C) - 1
    ph, pl = _pivots(H, L, order)
    res = _top_patterns(H, L, C, ph, last, tol, depth)
    K = max(H) + min(L)   # price ko mirror karte hain (positive rehta hai)
    iH, iL, iC = [K - x for x in L], [K - x for x in H], [K - x for x in C]
    iph = _pivots(iH, iL, order)[0]
    res += [_flip(p, K) for p in _top_patterns(iH, iL, iC, iph, last, tol, depth)]
    res += _trend_shapes(H, L, C, ph, pl, last)
    res += _cup(H, L, C, last)
    res += _candles(O, H, L, C)
    res.sort(key=lambda p: (p["kind"] == "candle", not p["broke"]))
    return res


def bear_breakout(pats):
    return any(p["kind"] == "chart" and p["bias"] == "BEAR" and p["broke"] for p in pats)


def has_bull_chart(pats):
    return any(p["kind"] == "chart" and p["bias"] == "BULL" for p in pats)


COL = {"BULL": "#26a69a", "BEAR": "#ef5350", "NEUTRAL": "#f5b942"}


def draw_pattern(ax, pats, max_chart=3, max_candle=2):
    nc = nk = 0
    for p in pats:
        col = COL[p["bias"]]
        if p["kind"] == "chart" and nc < max_chart:
            for x1, y1, x2, y2, d in p["lines"]:
                ax.plot([x1, x2], [y1, y2], color=col, linewidth=1.3,
                        linestyle="--" if d else "-", zorder=5)
            ax.text(p["lines"][0][0], p["lines"][0][1], p["name"], color=col,
                    fontsize=9, fontweight="bold", va="bottom", zorder=7)
            nc += 1
        elif p["kind"] == "candle" and nk < max_candle:
            ax.annotate(p["name"], (p["x"], p["y"]), color=col, fontsize=8,
                        fontweight="bold", ha="right", va="bottom",
                        xytext=(-4, 6 + 10 * nk), textcoords="offset points", zorder=7)
            nk += 1


def pattern_text(pats, d=6, maxn=5):
    if not pats:
        return ""
    icon = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}
    out = ["", "📐 Patterns:"]
    for p in pats[:maxn]:
        if p["kind"] == "candle":
            st = "last candle"
        else:
            st = "BREAKOUT ✅" if p["broke"] else "forming ⏳"
        line = f"{icon[p['bias']]} {p['name']} - {st}"
        if p.get("target"):
            line += f" | tgt {p['target']:.{d}g}"
        out.append(line)
    return "\n".join(out)
# ================= END PATTERNS =================


def pivots(h, l, w=3):
    ph, pl = [], []
    for i in range(w, len(h) - w):
        if h[i] == max(h[i - w:i + w + 1]):
            ph.append(h[i])
        if l[i] == min(l[i - w:i + w + 1]):
            pl.append(l[i])
    return ph, pl


def load_candles(pair, interval=None):
    raw = bot.get("/spot/candlesticks",
                  {"currency_pair": pair, "interval": interval or bp.TF, "limit": 200})[:-1]
    o = [bp.f(x[5]) for x in raw]
    h = [bp.f(x[3]) for x in raw]
    l = [bp.f(x[4]) for x in raw]
    c = [bp.f(x[2]) for x in raw]
    v = [bp.f(x[6]) for x in raw]
    return o, h, l, c, v


def ema_list(x, n):
    k = 2 / (n + 1)
    out = [x[0]]
    for p in x[1:]:
        out.append(p * k + out[-1] * (1 - k))
    return out


def rsi_last(c, n=14):
    g = ls = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        ls += max(-d, 0)
    g, ls = g / n, ls / n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        ls = (ls * (n - 1) + max(-d, 0)) / n
    return 100.0 if ls == 0 else 100 - 100 / (1 + g / ls)


def rsi_series(c, n=14):
    """Poori RSI line (chart ke RSI panel ke liye)."""
    out = [50.0] * len(c)
    if len(c) <= n:
        return out
    g = ls = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        ls += max(-d, 0)
    g, ls = g / n, ls / n
    out[n] = 100.0 if ls == 0 else 100 - 100 / (1 + g / ls)
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        ls = (ls * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if ls == 0 else 100 - 100 / (1 + g / ls)
    return out


def macd_state(c):
    e12, e26 = ema_list(c, 12), ema_list(c, 26)
    m = [a - b for a, b in zip(e12, e26)]
    sg = ema_list(m, 9)
    hist = [a - b for a, b in zip(m, sg)]
    return m[-1] > sg[-1] and hist[-1] > hist[-2]


def supertrend_up(h, l, c, n=10, mult=3.0):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]),
                              abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    atr = [tr[0]]
    for i in range(1, len(c)):
        atr.append((atr[-1] * (n - 1) + tr[i]) / n)
    fu = fl = 0.0
    up = True
    for i in range(len(c)):
        hl = (h[i] + l[i]) / 2
        bu, bl = hl + mult * atr[i], hl - mult * atr[i]
        if i == 0:
            fu, fl = bu, bl
            continue
        fu = bu if (bu < fu or c[i - 1] > fu) else fu
        fl = bl if (bl > fl or c[i - 1] < fl) else fl
        if up and c[i] < fl:
            up = False
        elif not up and c[i] > fu:
            up = True
    return up


def confirm(o, h, l, c, v):
    e20, e50 = ema_list(c, 20)[-1], ema_list(c, 50)[-1]
    r = rsi_last(c)
    vavg = sum(v[-21:-1]) / 20
    checks = {
        "EMA": c[-1] > e20 > e50,
        "Supertrend": supertrend_up(h, l, c),
        "RSI 50-70": 50 <= r <= 70,
        "MACD": macd_state(c),
        "Volume": v[-1] > 1.2 * vavg,
    }
    return sum(checks.values()), checks, r


def plan_levels(entry, atr, h, l):
    """Structure-based SL / T1 / T2. None return = setup skip."""
    ph, pl = pivots(h[-150:], l[-150:])
    swing = min(l[-20:])
    risk_struct = entry - (swing - 0.2 * atr)
    risk = min(max(risk_struct, 1.5 * atr), 3.5 * atr)
    sl = entry - risk
    res = sorted({p for p in ph if p > entry * 1.002})
    if any(entry < r < entry + 1.0 * risk for r in res):
        return None  # resistance bilkul paas, R:R kharab
    cands = [r for r in res if r >= entry + 1.2 * risk]
    t1 = cands[0] if cands and cands[0] <= entry + 3 * risk else entry + 1.5 * risk
    later = [r for r in cands if r > t1 * 1.002 and r <= entry + 6 * risk]
    t2 = later[0] if later else entry + 3 * risk
    if t2 <= t1:
        t2 = t1 + 1.5 * risk
    pl_below = sorted([p for p in pl if p < entry], reverse=True)
    return dict(sl=sl, t1=t1, t2=t2, risk=risk, res=res[:3],
                support=pl_below[0] if pl_below else None,
                rr1=(t1 - entry) / risk, rr2=(t2 - entry) / risk)


def make_chart(pair, o, h, l, c, v, entry, lv, score, path, pats=None,
               side="LONG", score_txt=None):
    n = len(c)
    s = max(0, n - CHART_BARS)
    cnt = n - s
    xs = list(range(cnt))

    # indicators (poori history par nikaal ke last CHART_BARS kaat rahe hain)
    e9, e26 = ema_list(c, 9), ema_list(c, 26)
    rs = rsi_series(c)[s:]
    ups, dns = [], []
    for k in range(max(s, 1), n):
        if e9[k] > e26[k] and e9[k - 1] <= e26[k - 1]:
            ups.append(k - s)
        elif e9[k] < e26[k] and e9[k - 1] >= e26[k - 1]:
            dns.append(k - s)
    e9s, e26s = e9[s:], e26[s:]
    off = sum(h[k] - l[k] for k in range(s, n)) / cnt * 0.6   # arrow ka gap

    bg, fg = "#0e1117", "#d0d4dc"
    green, red, blue, orange = "#26a69a", "#ef5350", "#4a90ff", "#f5b942"
    fig, (ax, axv, axr) = plt.subplots(
        3, 1, figsize=(9, 7.5), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [5, 1.2, 1.6], "hspace": 0.05})
    fig.patch.set_facecolor(bg)
    for a in (ax, axv, axr):
        a.set_facecolor(bg)
        a.tick_params(colors=fg, labelsize=8)
        a.grid(color="#1d222c", linewidth=0.5)
        for sp in a.spines.values():
            sp.set_color("#2a2f3a")

    # candles + volume
    for i, k in enumerate(range(s, n)):
        up = c[k] >= o[k]
        col = green if up else red
        ax.vlines(i, l[k], h[k], color=col, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o[k], c[k])), 0.6,
                                   max(abs(c[k] - o[k]), 1e-12), color=col))
        axv.bar(i, v[k], color=col, width=0.7, alpha=0.7)

    # EMA 9 / 26 + cross markers (blue +) + arrows
    ax.plot(xs, e9s, color=orange, linewidth=1.2, label="EMA 9")
    ax.plot(xs, e26s, color="#4caf50", linewidth=1.2, label="EMA 26")
    cross = ups + dns
    if cross:
        ax.scatter(cross, [e9s[i] for i in cross], marker="+", s=130,
                   color=blue, linewidths=2.5, zorder=6)
    if ups:
        ax.scatter(ups, [l[s + i] - off for i in ups], marker=r"$\uparrow$",
                   s=170, color=green, zorder=6)
    if dns:
        ax.scatter(dns, [h[s + i] + off for i in dns], marker=r"$\downarrow$",
                   s=170, color=red, zorder=6)

    # position boxes (Entry / SL / T1 / T2) + Liquidation line
    right = cnt + 3
    ax.fill_between([cnt - 1, right], entry, lv["t2"], color=green, alpha=0.15)
    ax.fill_between([cnt - 1, right], lv["sl"], entry, color=red, alpha=0.15)
    liq = entry * (1 - 0.99 / LEVERAGE) if side == "LONG" else entry * (1 + 0.99 / LEVERAGE)
    for y, col, name, ls in ((entry, blue, "Entry", "--"),
                             (lv["sl"], red, "SL", "-"),
                             (lv["t1"], green, "T1", "-"),
                             (lv["t2"], "#7bdcb5", "T2", "-"),
                             (liq, "#f4a6a6", f"Liq {LEVERAGE:g}x", "-")):
        ax.axhline(y, color=col, linestyle=ls, linewidth=1.1)
        ax.text(right + 0.5, y, f"{name} {y:.6g}", color=col, fontsize=8, va="center")
    for r in lv["res"]:
        ax.axhline(r, color="#8a8f9c", linestyle=":", linewidth=0.8)

    # patterns (Double Top/Bottom, H&S, Triangle, Flag, Cup, candles ...)
    if pats:
        draw_pattern(ax, pats)

    ax.set_xlim(-1, right + 16)
    lo = min(min(l[s:]), lv["sl"], lv["t2"], liq) * 0.997
    hi = max(max(h[s:]), lv["sl"], lv["t2"], liq) * 1.005
    ax.set_ylim(lo, hi)
    ax.set_title(f"{pair}  {bp.TF}  {side}   {score_txt or f'setup score {score:.0f}/8'}   "
                 f"T1 {lv['rr1']:.1f}R  T2 {lv['rr2']:.1f}R",
                 color=fg, fontsize=11, loc="left")
    ax.legend(loc="upper left", facecolor=bg, edgecolor="#2a2f3a", labelcolor=fg, fontsize=8)

    # RSI panel
    axr.plot(xs, rs, color=orange, linewidth=1.2)
    axr.axhline(70, color=red, linestyle="--", linewidth=0.8)
    axr.axhline(50, color="#888888", linestyle="--", linewidth=0.8)
    axr.axhline(30, color=green, linestyle="--", linewidth=0.8)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI 14", color=fg, fontsize=8)
    axr.text(cnt - 1, rs[-1], f" {rs[-1]:.0f}", color=orange, fontsize=8, va="center")
    axv.set_ylabel("Vol", color=fg, fontsize=8)
    axv.set_yticks([])
    ax.set_xticks([])
    fig.savefig(path, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def caption(x, lv, entry, pats=None):
    d = bp.dec(entry)
    risk_pct = (entry - lv["sl"]) / entry * 100
    lines = [
        f"{x['pair']}  LONG setup  (score {x['score']:.0f}/8, 24h {x['chg']:+.1f}%)",
        f"Entry ~{entry:.{d}f}",
        f"SL {lv['sl']:.{d}f}  (-{risk_pct:.1f}%)",
        f"T1 {lv['t1']:.{d}f}  ({lv['rr1']:.1f}R)  |  T2 {lv['t2']:.{d}f}  ({lv['rr2']:.1f}R)",
        "",
        f"Kyun: taker buying {x['buy_ratio'] * 100:.0f}%, book bids {x['imb'] * 100:.0f}%, "
        f"volume {x['spike']:.1f}x, CMF {x['cmf']:+.2f}, EMA20>EMA50",
        f"Indicators: {x.get('ind', '')}",
    ]
    if lv["res"]:
        lines.append("Upar resistance: " + ", ".join(f"{r:.{d}f}" for r in lv["res"]))
    if lv["support"]:
        lines.append(f"Neeche support: {lv['support']:.{d}f}")
    lines.append(pattern_text(pats, d) if pats else "\n📐 Patterns: koi clear pattern nahi mila")
    lines += ["", "T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."]
    return "\n".join(lines)[:1020]


def atr_calc(h, l, c, n=14):
    tr = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
          for i in range(1, len(c))]
    return sum(tr[-n:]) / n


def confirm_short(o, h, l, c, v):
    e20, e50 = ema_list(c, 20)[-1], ema_list(c, 50)[-1]
    r = rsi_last(c)
    vavg = sum(v[-21:-1]) / 20
    e12, e26 = ema_list(c, 12), ema_list(c, 26)
    m = [a - b for a, b in zip(e12, e26)]
    sg = ema_list(m, 9)
    hist = [a - b for a, b in zip(m, sg)]
    checks = {
        "EMA": c[-1] < e20 < e50,
        "Supertrend": not supertrend_up(h, l, c),
        "RSI 30-50": 30 <= r <= 50,
        "MACD": m[-1] < sg[-1] and hist[-1] < hist[-2],
        "Volume": v[-1] > 1.2 * vavg,
    }
    return sum(checks.values()), checks, r


def plan_levels_short(entry, atr, h, l):
    """SHORT ke liye: SL swing high ke upar, targets neeche ke supports par.
    lv['res'] = neeche ke support levels, lv['support'] = upar ka nearest resistance."""
    ph, pl = pivots(h[-150:], l[-150:])
    swing = max(h[-20:])
    risk_struct = (swing + 0.2 * atr) - entry
    risk = min(max(risk_struct, 1.5 * atr), 3.5 * atr)
    sl = entry + risk
    sup = sorted({p for p in pl if p < entry * 0.998}, reverse=True)
    if any(entry - 1.0 * risk < s_ < entry for s_ in sup):
        return None  # support bilkul paas, R:R kharab
    cands = [s_ for s_ in sup if s_ <= entry - 1.2 * risk]
    t1 = cands[0] if cands and cands[0] >= entry - 3 * risk else entry - 1.5 * risk
    later = [s_ for s_ in cands if s_ < t1 * 0.998 and s_ >= entry - 6 * risk]
    t2 = later[0] if later else entry - 3 * risk
    if t2 >= t1:
        t2 = t1 - 1.5 * risk
    if min(t1, t2) <= entry * 0.05:
        return None  # target zero ke paas, short ka matlab nahi
    above = sorted([p for p in ph if p > entry])
    return dict(sl=sl, t1=t1, t2=t2, risk=risk, res=sup[:3],
                support=above[0] if above else None,
                rr1=(entry - t1) / risk, rr2=(entry - t2) / risk)


_ctx = {}


def _trend(pair, interval):
    """+1 uptrend, -1 downtrend, 0 sideways, None data nahi mila."""
    key = (pair, interval)
    if key not in _ctx:
        try:
            _, _, _, c, _ = load_candles(pair, interval)
            e20, e50 = ema_list(c, 20)[-1], ema_list(c, 50)[-1]
            _ctx[key] = 1 if c[-1] > e20 > e50 else (-1 if c[-1] < e20 < e50 else 0)
            time.sleep(0.1)
        except Exception:
            _ctx[key] = None
    return _ctx[key]


def trade_filters(pair, side, h, l, c):
    """Fail hone wale trades kam karne ke filters. (ok, reason) return."""
    want = 1 if side == "LONG" else -1
    if HTF_FILTER and _trend(pair, "4h") == -want:
        return False, "4h trend ulta hai"
    if BTC_FILTER and _trend("BTC_USDT", "1h") == -want:
        return False, "BTC trend ulta hai"
    if EXT_ATR:
        a, e20 = atr_calc(h, l, c), ema_list(c, 20)[-1]
        if a > 0 and abs(c[-1] - e20) > EXT_ATR * a:
            return False, "price EMA20 se bahut door (late entry)"
    return True, ""


def caption_pat(pair, side, lv, entry, ind, pats, chg):
    d = bp.dec(entry)
    risk_pct = abs(entry - lv["sl"]) / entry * 100
    sg = "-" if side == "LONG" else "+"
    main = next((p for p in pats if p["kind"] == "chart"), None)
    lines = [
        f"{pair}  {side} setup  (pattern: {main['name'] if main else '-'}, 24h {chg:+.1f}%)",
        f"Entry ~{entry:.{d}f}",
        f"SL {lv['sl']:.{d}f}  ({sg}{risk_pct:.1f}%)",
        f"T1 {lv['t1']:.{d}f}  ({lv['rr1']:.1f}R)  |  T2 {lv['t2']:.{d}f}  ({lv['rr2']:.1f}R)",
        "",
        f"Indicators: {ind}",
    ]
    if side == "LONG":
        if lv["res"]:
            lines.append("Upar resistance: " + ", ".join(f"{r:.{d}f}" for r in lv["res"]))
        if lv["support"]:
            lines.append(f"Neeche support: {lv['support']:.{d}f}")
    else:
        if lv["res"]:
            lines.append("Neeche support: " + ", ".join(f"{r:.{d}f}" for r in lv["res"]))
        if lv["support"]:
            lines.append(f"Upar resistance: {lv['support']:.{d}f}")
    lines.append(pattern_text(pats, d))
    tail = "T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."
    if side == "SHORT":
        tail += " Short ke liye futures/perp chahiye."
    lines += ["", tail]
    return "\n".join(lines)[:1020]


def pattern_trades(uni, state, now, skip_pairs):
    """Saare coins par pattern scan: bullish breakout -> LONG, bearish breakdown -> SHORT."""
    sent = 0
    for r in uni[:PAT_SCAN_LIMIT]:
        if sent >= MAX_PAT_ALERTS:
            break
        pair = r["pair"]
        if pair in skip_pairs:
            continue
        try:
            o, h, l, c, v = load_candles(pair)
            time.sleep(0.12)
            if len(c) < 80:
                continue
            pats = detect_all(o, h, l, c, max(0, len(c) - CHART_BARS))
            charts = [p for p in pats if p["kind"] == "chart"]
            if PAT_BREAKOUT_ONLY:
                charts = [p for p in charts if p["broke"]]
            if not charts or charts[0]["bias"] == "NEUTRAL":
                continue
            side = "LONG" if charts[0]["bias"] == "BULL" else "SHORT"
            opp = "BEAR" if side == "LONG" else "BULL"
            if any(p["kind"] == "chart" and p["bias"] == opp and p["broke"] for p in pats):
                continue  # ulte pattern ka breakout bhi hai, confusion
            key = f"{pair}|{side}"
            if now - state.get(key, 0) < bp.COOLDOWN_H * 3600:
                continue
            if side == "LONG" and now - state.get(pair, 0) < bp.COOLDOWN_H * 3600:
                continue
            if side == "LONG":
                conf, checks, _ = confirm(o, h, l, c, v)
            else:
                conf, checks, _ = confirm_short(o, h, l, c, v)
            if conf < MIN_CONFIRM_PAT:
                continue
            ok_f, why = trade_filters(pair, side, h, l, c)
            if not ok_f:
                print(f"{pair}: {side} skip ({why})")
                continue
            entry, a = c[-1], atr_calc(h, l, c)
            lv = plan_levels(entry, a, h, l) if side == "LONG" else plan_levels_short(entry, a, h, l)
            if not lv:
                print(f"{pair}: {side} level paas me, skip")
                continue
            if abs(entry - lv["sl"]) / entry < MIN_RISK_PCT:
                continue
            tg = charts[0].get("target")      # pattern ka measured target T2 bana do agar sahi ho
            if tg:
                if side == "LONG" and lv["t1"] * 1.002 < tg <= entry + 8 * lv["risk"]:
                    lv["t2"], lv["rr2"] = tg, (tg - entry) / lv["risk"]
                if side == "SHORT" and entry - 8 * lv["risk"] <= tg < lv["t1"] * 0.998 and tg > 0:
                    lv["t2"], lv["rr2"] = tg, (entry - tg) / lv["risk"]
            ind = " ".join(("✅" if ok else "❌") + k for k, ok in checks.items())
            chg = (c[-1] / c[-25] - 1) * 100 if len(c) > 25 else 0.0
            path = f"/tmp/{pair}_{side}.png"
            make_chart(pair, o, h, l, c, v, entry, lv, conf, path, pats,
                       side=side, score_txt=f"confirm {conf}/5")
            cap = caption_pat(pair, side, lv, entry, ind, pats, chg)
            if not send_photo(path, cap):
                bot.send_telegram(cap)
            state[key] = now
            sent += 1
            print(f"{pair}: {side} alert ({charts[0]['name']})")
        except Exception as e:
            print(f"{pair}: pattern trade fail ({repr(e)[:150]})")
    return sent


def send_photo(path, cap):
    token, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not token or not chat:
        print("Telegram keys nahi mili, sirf print kar raha hoon.")
        print(cap)
        return False
    try:
        with open(path, "rb") as fh:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                              data={"chat_id": chat, "caption": cap},
                              files={"photo": fh}, timeout=30)
        print("Telegram photo:", r.status_code, r.text[:120])
        return r.status_code == 200
    except Exception as e:
        print("Photo fail:", repr(e)[:150])
        return False


def load_state():
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(state):
    cutoff = time.time() - 48 * 3600
    with open(STATE_FILE, "w") as fh:
        json.dump({k: v for k, v in state.items() if v > cutoff}, fh)


def main():
    _ctx.clear()
    uni, passed = bp.universe()
    print(f"Universe: {passed} coins liquidity paas, {len(uni)} scan honge")
    stage = []
    for r in uni:
        try:
            x = bp.stage_a(r)
        except Exception as e:
            print(f"{r['pair']}: skip A ({e})")
            continue
        time.sleep(0.12)
        if x and x["ext"] <= 3.0 and x["chg"] <= 40:
            stage.append(x)
    stage.sort(key=lambda x: (x["score_a"], x["spike"]), reverse=True)
    deep = []
    for x in stage[:bp.TOP_DEEP]:
        if x["score_a"] < 3:
            break
        try:
            y = bp.stage_b(x)
        except Exception as e:
            print(f"{x['pair']}: skip B ({e})")
            continue
        time.sleep(0.15)
        if y:
            deep.append(y)
            print(f"{y['pair']}: A={y['score_a']:.0f} buy={y['buy_ratio']:.2f} "
                  f"book={y['imb']:.2f} -> {y['score']:.0f}")
    cands = [y for y in deep if y["score"] >= bp.MIN_SCORE and y["buy_ratio"] >= bp.MIN_BUY_RATIO]
    cands.sort(key=lambda y: (y["score"], y["buy_ratio"]), reverse=True)

    state, now, sent = load_state(), time.time(), 0
    sent_pairs = set()
    for y in cands:
        if sent >= MAX_ALERTS:
            break
        if now - state.get(y["pair"], 0) < bp.COOLDOWN_H * 3600:
            continue
        try:
            o, h, l, c, v = load_candles(y["pair"])
            conf, checks, rsi_v = confirm(o, h, l, c, v)
            if conf < MIN_CONFIRM:
                continue
            y["ind"] = " ".join(("✅" if ok else "❌") + k for k, ok in checks.items())
            ask = y["ask"]
            entry = ask if ask and abs(ask / c[-1] - 1) < 0.03 else c[-1]
            lv = plan_levels(entry, y["atr"], h, l)
            if not lv:
                print(f"{y['pair']}: resistance paas, skip")
                continue
            if (entry - lv["sl"]) / entry < MIN_RISK_PCT:
                print(f"{y['pair']}: SL bahut chhota (fees ka asar), skip")
                continue
            pats = detect_all(o, h, l, c, max(0, len(c) - CHART_BARS))
            print(f"{y['pair']}: patterns = {[p['name'] for p in pats]}")
            if bear_breakout(pats):
                print(f"{y['pair']}: bearish pattern breakdown, long skip")
                continue
            ok_f, why = trade_filters(y["pair"], "LONG", h, l, c)
            if not ok_f:
                print(f"{y['pair']}: LONG skip ({why})")
                continue
            if PATTERN_ONLY and not has_bull_chart(pats):
                print(f"{y['pair']}: bullish pattern nahi, skip (PATTERN_ONLY)")
                continue
            path = f"/tmp/{y['pair']}.png"
            make_chart(y["pair"], o, h, l, c, v, entry, lv, y["score"], path, pats)
            cap = caption(y, lv, entry, pats)
            ok = send_photo(path, cap)
            if not ok:
                bot.send_telegram(cap)       # photo na jaye to text alert
            state[y["pair"]] = now
            sent_pairs.add(y["pair"])
            sent += 1
        except Exception as e:
            print(f"{y['pair']}: alert fail ({repr(e)[:150]})")
        time.sleep(0.3)
    # ---- pattern based LONG + SHORT alerts (saare coins)
    try:
        sent += pattern_trades(uni, state, now, sent_pairs)
    except Exception as e:
        print("pattern_trades fail:", repr(e)[:150])
    save_state(state)
    print(f"Alerts bheje: {sent}")
    return sent


if __name__ == "__main__":
    main()
