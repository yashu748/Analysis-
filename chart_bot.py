#!/usr/bin/env python3
"""
Gate.io crypto chart-alert bot  (15m alerts, 1h/4h/1d trend filter, continuous loop)
Telegram secrets: TG_TOKEN, TG_CHAT_ID

v2 - galat signal rokne ke filters:
  1. Ulta pattern block   : LONG par koi bhi bearish pattern  -> alert nahi
                            SHORT par koi bhi bullish pattern -> alert nahi
  2. Resistance/Support   : entry ke 1R ke andar resistance  / support  ho -> alert nahi (15m + 1h levels)
  3. Pump candle block    : badi candle / EMA se door / tez run /  -> alert 
  4. RSI limit            : LONG RSI <= 62, SHORT RSI >= 38, 1h RSI extreme par bhi block
  5. Structure SL         : SL swing low/high ke bahar, max risk 4% 
  6. Illiquid coin skip   : movers ke liye bhi minimum volume badhaya
Har scan ke baad log me dikhta hai ki kaun sa filter kitne signal roka.
"""
import os, io, re, time, threading, traceback
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= CONFIG =================
TG_TOKEN = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN", "")
TG_CHAT = os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")

ALERT_TF = "15m"                                   # alert + chart timeframe
HTF = {"1h": 900, "4h": 1800, "1d": 7200}          # trend-filter TFs : cache seconds
SCAN_EVERY = 120                                   # seconds between scans
RUN_MINUTES = float(os.getenv("RUN_MINUTES", "350"))  # GitHub job limit is 6h
MIN_VOL_USDT = 1_000_000                             # 24h volume filter
MOVER_PCT, MOVER_MIN_VOL = 10.0, 600_000           # fast movers (bade volume wale)
MAX_COINS = 250
MIN_CONFIRM = 5                                    # out of 5 indicators (5 = sabhi)
VOL_MULT = 2.0                                     # volume spike kitna bada ho
VOL_MAX = 10.0                                     # isse bada volume = blow-off, skip
REQUIRE_PATTERN = False                            # True = pattern bhi zaroori (bahut kam alert)
MAX_ALERTS_PER_SCAN = 2                            # ek scan mein sirf top alerts
MAX_ALERTS_PER_DAY = 6                             # 24 ghante me maximum itne alert (best quality wale)
MIN_QUALITY = 75                                   # quality score (0-100) isse kam ho to alert nahi
BTC_FILTER = True                                  # BTC 1h trend against ho to altcoin alert nahi
TRACK_RESULTS = True                               # har alert ka SL/T1/T2 result Telegram par bhejo
COOLDOWN_MIN = 360                                 # same coin+side repeat gap
ATR_MULT, MAX_RISK_PCT = 1.5, 4.0                  # SL settings (risk isse zyada = skip)
T1_R, T2_R = 1.5, 3.0
SEND_LONG, SEND_SHORT = True, True

# --- v2 filter settings ---
LONG_RSI_MAX, SHORT_RSI_MIN = 62, 38               # 15m RSI limit
HTF_RSI_MAX, HTF_RSI_MIN = 70, 30                  # 1h RSI extreme
RES_MIN_R = 1.0                                    # resistance/support kam se kam itne R door ho (1R = SL distance)

# ---------- MODE: yahi ek line badalke strictness chuno ----------
#   "safe"     : sab filters ON (sabse kam alert, sabse achhi quality)
#   "balanced" : resistance(0.5R) + RSI 66/34 + pump filter (dheela). Ulta pattern block NAHI (sirf caption me warning) <- default
#   "loose"    : sirf trend/SL filter, pattern/resistance/pump block nahi (sabse zyada alert, purane jaisa)
MODE = os.getenv("BOT_MODE", "balanced").lower()
if MODE == "safe":
    USE_PUMP, USE_PATTERN, RES_MIN_R, LONG_RSI_MAX, SHORT_RSI_MIN = True, True, 1.0, 62, 38
elif MODE == "loose":
    USE_PUMP, USE_PATTERN, RES_MIN_R, LONG_RSI_MAX, SHORT_RSI_MIN = False, False, 0.0, 72, 28
else:
    MODE = "balanced"
    USE_PUMP, USE_PATTERN, RES_MIN_R, LONG_RSI_MAX, SHORT_RSI_MIN = True, False, 0.5, 66, 34
BIG_CANDLE_ATR, BIG_BODY_ATR = 3.5, 3.0            # pump candle
EXTENSION_ATR = 3.0                                # price EMA26 se itna ATR door = chasing
RUN_ATR = 6.0                                      # pichhli 5 candles me itna move = late

BASE = "https://api.gateio.ws/api/v4"
STABLE = {"USDC", "TUSD", "FDUSD", "DAI", "USDD", "USDP", "PYUSD", "EURT"}

# ================= HTTP =================
_lock, _next = threading.Lock(), [0.0]
def get(url, params=None, tries=3):
    for k in range(tries):
        with _lock:                                  # global rate limit ~16 req/s
            w = _next[0] - time.time()
            if w > 0: time.sleep(w)
            _next[0] = time.time() + 0.06
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429: time.sleep(2 + k * 2); continue
            if r.status_code == 200: return r.json()
        except Exception:
            time.sleep(1)
    return None

def fp(x):
    x = float(x)
    a = abs(x)
    if a >= 100: return f"{x:.2f}"
    if a >= 1: return f"{x:.4f}"
    if a >= 0.01: return f"{x:.5f}"
    return f"{x:.8f}"

def fetch(pair, tf, limit):
    r = get(f"{BASE}/spot/candlesticks", {"currency_pair": pair, "interval": tf, "limit": limit})
    if not r or len(r) < 30: return None
    df = pd.DataFrame([[float(x[0]), float(x[5]), float(x[4]), float(x[3]), float(x[2]), float(x[1])]
                       for x in r], columns=["t", "o", "l", "h", "c", "v"])
    df = df[["t", "o", "h", "l", "c", "v"]]
    return df.iloc[:-1].reset_index(drop=True)       # drop forming candle

# ================= INDICATORS =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n=14):
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def supertrend(df, n=10, m=3.0):
    a = atr(df, n).values; hl2 = ((df.h + df.l) / 2).values; c = df.c.values
    ub, lb = hl2 + m * a, hl2 - m * a
    fub, flb, d = ub.copy(), lb.copy(), np.ones(len(df))
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
        if d[i-1] == 1 and c[i] < flb[i]: d[i] = -1
        elif d[i-1] == -1 and c[i] > fub[i]: d[i] = 1
        else: d[i] = d[i-1]
    return d
def htf_trend(df):
    if df is None or len(df) < 55: return 0
    c = df.c; e21, e50 = ema(c, 21).iloc[-1], ema(c, 50).iloc[-1]
    if c.iloc[-1] > e21 > e50: return 1
    if c.iloc[-1] < e21 < e50: return -1
    return 0

# ================= PATTERNS =================
def pivots(df, k=3):
    h, l, hi, lo = df.h.values, df.l.values, [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i-k:i+k+1].max(): hi.append((i, h[i]))
        if l[i] == l[i-k:i+k+1].min(): lo.append((i, l[i]))
    def dedupe(p, hi_):                               # merge pivots closer than k candles
        out = []
        for q in p:
            if out and q[0] - out[-1][0] <= k:
                if (q[1] > out[-1][1]) == hi_: out[-1] = q
            else: out.append(q)
        return out
    return dedupe(hi, True), dedupe(lo, False)

def detect_patterns(df, tol=0.008):
    out, n = [], len(df)
    hi, lo = pivots(df); close = df.c.iloc[-1]
    hi, lo = [p for p in hi if p[0] > n - 120], [p for p in lo if p[0] > n - 120]
    near = lambda a, b: abs(a - b) / max(a, b) <= tol
    def between_min(i, j): return df.l.values[i:j+1].min()
    def between_max(i, j): return df.h.values[i:j+1].max()
    def add(name, side, neck, height, lvl):
        # lvl = pattern invalidation level (SHORT: pattern ka top, LONG: pattern ka bottom)
        # price uske bahar nikal chuka ho to pattern fail ho gaya - ignore
        if side == "SHORT" and close > lvl * 1.003: return
        if side == "LONG" and close < lvl * 0.997: return
        brk = close < neck if side == "SHORT" else close > neck
        tgt = neck - height if side == "SHORT" else neck + height
        out.append(dict(name=name, side=side, breakout=bool(brk), target=tgt))
    # double / triple top
    if len(hi) >= 2 and near(hi[-1][1], hi[-2][1]) and hi[-1][0] - hi[-2][0] >= 5 and n - hi[-1][0] <= 40:
        neck = between_min(hi[-2][0], hi[-1][0]); top = max(hi[-1][1], hi[-2][1])
        if (top - neck) / top > 0.012: add("Double Top", "SHORT", neck, top - neck, top)
    if len(hi) >= 3 and near(hi[-1][1], hi[-2][1]) and near(hi[-2][1], hi[-3][1]) and n - hi[-1][0] <= 40:
        neck = between_min(hi[-3][0], hi[-1][0]); top = max(p[1] for p in hi[-3:])
        if (top - neck) / top > 0.012: add("Triple Top", "SHORT", neck, top - neck, top)
    # double / triple bottom
    if len(lo) >= 2 and near(lo[-1][1], lo[-2][1]) and lo[-1][0] - lo[-2][0] >= 5 and n - lo[-1][0] <= 40:
        neck = between_max(lo[-2][0], lo[-1][0]); bot = min(lo[-1][1], lo[-2][1])
        if (neck - bot) / neck > 0.012: add("Double Bottom", "LONG", neck, neck - bot, bot)
    if len(lo) >= 3 and near(lo[-1][1], lo[-2][1]) and near(lo[-2][1], lo[-3][1]) and n - lo[-1][0] <= 40:
        neck = between_max(lo[-3][0], lo[-1][0]); bot = min(p[1] for p in lo[-3:])
        if (neck - bot) / neck > 0.012: add("Triple Bottom", "LONG", neck, neck - bot, bot)
    # head & shoulders / inverse
    if len(hi) >= 3 and n - hi[-1][0] <= 40:
        a, b, c3 = hi[-3][1], hi[-2][1], hi[-1][1]
        if b > a * 1.01 and b > c3 * 1.01 and near(a, c3):
            neck = (between_min(hi[-3][0], hi[-2][0]) + between_min(hi[-2][0], hi[-1][0])) / 2
            add("Head & Shoulders", "SHORT", neck, b - neck, b)
    if len(lo) >= 3 and n - lo[-1][0] <= 40:
        a, b, c3 = lo[-3][1], lo[-2][1], lo[-1][1]
        if b < a * 0.99 and b < c3 * 0.99 and near(a, c3):
            neck = (between_max(lo[-3][0], lo[-2][0]) + between_max(lo[-2][0], lo[-1][0])) / 2
            add("Inverse H&S", "LONG", neck, neck - b, b)
    # ascending / descending triangle
    if len(hi) >= 3 and len(lo) >= 3 and n - hi[-1][0] <= 30:
        h3, l3 = [p[1] for p in hi[-3:]], [p[1] for p in lo[-3:]]
        if near(max(h3), min(h3)) and l3[0] < l3[1] < l3[2]:
            add("Ascending Triangle", "LONG", max(h3), max(h3) - l3[0], l3[2])
        if near(max(l3), min(l3)) and h3[0] > h3[1] > h3[2]:
            add("Descending Triangle", "SHORT", min(l3), h3[0] - min(l3), h3[2])
    # rising / falling wedge + symmetrical triangle (last 3 pivots par trendline fit)
    if len(hi) >= 3 and len(lo) >= 3:
        H, W = hi[-3:], lo[-3:]
        last_piv = max(H[-1][0], W[-1][0])
        if last_piv - min(H[0][0], W[0][0]) >= 20 and n - last_piv <= 25:
            try:
                sh, ih = np.polyfit([q[0] for q in H], [q[1] for q in H], 1)
                sw_, iw = np.polyfit([q[0] for q in W], [q[1] for q in W], 1)
            except Exception:
                sh = None
            if sh is not None:
                xn = n - 1
                up_line, lo_line = sh * xn + ih, sw_ * xn + iw
                x0 = min(H[0][0], W[0][0])
                width0 = (sh * x0 + ih) - (sw_ * x0 + iw)
                if up_line > lo_line and width0 > (up_line - lo_line) * 1.15:      # lines paas aa rahi hain
                    if sh > 0 and sw_ > 0 and sw_ > sh:
                        add("Rising Wedge", "SHORT", lo_line, width0, up_line)
                    elif sh < 0 and sw_ < 0 and sh < sw_:
                        add("Falling Wedge", "LONG", up_line, width0, lo_line)
                    elif sh < 0 and sw_ > 0:
                        if close > up_line: add("Symmetrical Triangle", "LONG", up_line, width0, lo_line)
                        elif close < lo_line: add("Symmetrical Triangle", "SHORT", lo_line, width0, up_line)
    return out

# ================= SIGNAL =================
hcache, last_alert, seen, stats = {}, {}, {}, {}

def rej(why):
    """Filter ne signal roka - count karo (log me dikhega), None return."""
    stats[why] = stats.get(why, 0) + 1
    return None

def htf_df(pair, tf):
    k, now = (pair, tf), time.time()
    if k not in hcache or now - hcache[k][0] > HTF[tf]:
        hcache[k] = (now, fetch(pair, tf, 200))
    return hcache[k][1]

def get_trends(pair):
    return {tf: htf_trend(htf_df(pair, tf)) for tf in HTF}

def analyze(pair):
    df = fetch(pair, ALERT_TF, 250)
    if df is None or len(df) < 120: return None
    ts = df.t.iloc[-1]
    if seen.get(pair) == ts: return None
    seen[pair] = ts
    c = df.c; e9, e26 = ema(c, 9), ema(c, 26)
    st = supertrend(df); r = rsi(c); a = atr(df).iloc[-1]
    ml = ema(c, 12) - ema(c, 26); hist = ml - ema(ml, 9)
    vavg = df.v.iloc[-21:-1].mean()
    vr = df.v.iloc[-1] / vavg if vavg > 0 else 0
    vol_ok = vr > VOL_MULT
    R, price = r.iloc[-1], c.iloc[-1]
    if np.isnan(R) or a <= 0: return None
    L = dict(EMA=e9.iloc[-1] > e26.iloc[-1] and price > e26.iloc[-1], Supertrend=st[-1] == 1,
             RSI=50 <= R <= 70, MACD=hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2], Volume=vol_ok)
    S = dict(EMA=e9.iloc[-1] < e26.iloc[-1] and price < e26.iloc[-1], Supertrend=st[-1] == -1,
             RSI=30 <= R <= 50, MACD=hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2], Volume=vol_ok)
    sl_, ss_ = sum(L.values()), sum(S.values())
    side = None
    if SEND_LONG and sl_ >= MIN_CONFIRM and sl_ > ss_:
        if R > LONG_RSI_MAX: return rej("RSI bahut high (LONG)")
        side, ind = "LONG", L
    elif SEND_SHORT and ss_ >= MIN_CONFIRM and ss_ > sl_:
        if R < SHORT_RSI_MIN: return rej("RSI bahut low (SHORT)")
        side, ind = "SHORT", S
    if not side: return None
    stats["[candidate] 5/5 setup mila"] = stats.get("[candidate] 5/5 setup mila", 0) + 1
    if time.time() - last_alert.get((pair, side), 0) < COOLDOWN_MIN * 60: return None
    sgn = 1 if side == "LONG" else -1

    # ---- FILTER 3: pump candle / chasing / blow-off ----
    last = df.iloc[-1]
    rng, body = last.h - last.l, abs(last.c - last.o)
    if USE_PUMP:
        if vr > VOL_MAX: return rej("volume blow-off")
        if rng > BIG_CANDLE_ATR * a or body > BIG_BODY_ATR * a: return rej("pump candle")
        if abs(price - e26.iloc[-1]) > EXTENSION_ATR * a: return rej("price EMA se bahut door")
        if (price - c.iloc[-6]) * sgn > RUN_ATR * a: return rej("tez run, late entry")

    # ---- trend filter (1h / 4h / 1d) ----
    tr = get_trends(pair); want = sgn
    # 1h aur 4h against na hon, aur inme se kam se kam ek saath ho. 1d sirf caption me dikhta hai (block nahi karta)
    if tr["1h"] == -want or tr["4h"] == -want or (tr["1h"] != want and tr["4h"] != want):
        return rej("1h/4h trend saath nahi")

    # ---- BTC filter ----
    btc_tr = 0
    if BTC_FILTER and pair != "BTC_USDT":
        btc_tr = htf_trend(htf_df("BTC_USDT", "1h"))
        if btc_tr == -want: return rej("BTC trend against")

    # ---- FILTER 1: ulta pattern (forming ho ya breakout, dono block) ----
    pats = detect_patterns(df)
    opp = [p for p in pats if p["side"] != side]
    if USE_PATTERN and opp: return rej(f"ulta pattern: {opp[0]['name']}")
    if REQUIRE_PATTERN and not any(p["side"] == side for p in pats): return rej("pattern nahi mila")

    # ---- FILTER 4: 1h RSI extreme ----
    h1 = htf_df(pair, "1h")
    if h1 is not None and len(h1) > 20:
        r1 = rsi(h1.c).iloc[-1]
        if not np.isnan(r1):
            if side == "LONG" and r1 > HTF_RSI_MAX: return rej("1h RSI overbought")
            if side == "SHORT" and r1 < HTF_RSI_MIN: return rej("1h RSI oversold")

    # ---- FILTER 5: structure SL (chhota nahi karte, zyada door ho to skip) ----
    sw = df.tail(10)
    if side == "LONG": sl = min(price - ATR_MULT * a, sw.l.min() - 0.25 * a)
    else: sl = max(price + ATR_MULT * a, sw.h.max() + 0.25 * a)
    risk = abs(price - sl)
    if risk > price * MAX_RISK_PCT / 100: return rej("SL bahut door (risk zyada)")
    if risk < price * 0.002: return rej("SL bahut paas")
    t1, t2 = price + sgn * T1_R * risk, price + sgn * T2_R * risk

    # ---- FILTER 2: resistance / support T1 se pehle (15m + 1h levels) ----
    hi, lo = pivots(df)
    if h1 is not None and len(h1) > 20:
        hi1, lo1 = pivots(h1.tail(150).reset_index(drop=True), 3)
        hi, lo = hi + hi1, lo + lo1
    res = [p[1] for p in hi if p[1] > price * 1.0005]
    sup = [p[1] for p in lo if p[1] < price * 0.9995]
    nres, nsup = (min(res) if res else None), (max(sup) if sup else None)
    if side == "LONG" and nres is not None and (nres - price) < RES_MIN_R * risk: return rej("resistance bahut paas")
    if side == "SHORT" and nsup is not None and (price - nsup) < RES_MIN_R * risk: return rej("support bahut paas")

    # ---- QUALITY SCORE (0-100) ----
    q = 40 + 6 * sum(1 for v in tr.values() if v == want)
    same = [p_ for p_ in pats if p_["side"] == side]
    if same: q += 8 + (6 if any(p_["breakout"] for p_ in same) else 0)
    lvl = nres if side == "LONG" else nsup
    dist = None if lvl is None else abs(lvl - price)
    if dist is None or dist >= 3 * risk: q += 12
    elif dist >= 2 * risk: q += 8
    else: q += 3
    if 2 <= vr <= 5: q += 6
    if (side == "LONG" and 52 <= R <= 62) or (side == "SHORT" and 38 <= R <= 48): q += 6
    if btc_tr == want: q += 6
    q = min(q, 100)
    if q < MIN_QUALITY: return rej("quality score kam")

    return dict(pair=pair, side=side, price=price, sl=sl, t1=t1, t2=t2, risk_pct=risk / price * 100,
                ind=ind, score=max(sl_, ss_), q=q, rsi=R, trends=tr, pats=pats, df=df,
                e9=e9, e26=e26, rsi_s=r, res=nres, sup=nsup)

# ================= CHART + TELEGRAM =================
def make_chart(s):
    d = s["df"].tail(100).reset_index(drop=True); n = len(d); off = len(s["df"]) - n
    bg = "#0d1117"
    fig, (ax, av, ar) = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor=bg,
                                     gridspec_kw={"height_ratios": [5, 1.2, 1.5], "hspace": 0.05})
    for a_ in (ax, av, ar):
        a_.set_facecolor(bg); a_.tick_params(colors="#9ca3af", labelsize=8)
        for sp in a_.spines.values(): sp.set_color("#374151")
    for i, row in d.iterrows():
        col = "#26a69a" if row.c >= row.o else "#ef5350"
        ax.vlines(i, row.l, row.h, color=col, lw=1)
        ax.bar(i, max(abs(row.c - row.o), row.c * 1e-5), bottom=min(row.o, row.c), color=col, width=0.65)
        av.bar(i, row.v, color=col, width=0.65)
    ax.plot(range(n), s["e9"].values[off:], color="#f59e0b", lw=1, label="EMA 9")
    ax.plot(range(n), s["e26"].values[off:], color="#22c55e", lw=1, label="EMA 26")
    lv = [("Entry", s["price"], "#60a5fa"), ("SL", s["sl"], "#ef5350"),
          ("T1", s["t1"], "#2dd4bf"), ("T2", s["t2"], "#2dd4bf")]
    if s["side"] == "LONG" and s["res"]: lv.append(("Res", s["res"], "#ffb74d"))
    if s["side"] == "SHORT" and s["sup"]: lv.append(("Sup", s["sup"], "#ffb74d"))
    for name, v, col in lv:
        ax.axhline(v, color=col, ls="--", lw=1)
        ax.text(n + 0.5, v, f"{name} {fp(v)}", color=col, fontsize=8, va="center")
    lo_, hi_ = min(d.l.min(), s["sl"], s["t2"]), max(d.h.max(), s["sl"], s["t2"])
    pad = (hi_ - lo_) * 0.05; ax.set_ylim(lo_ - pad, hi_ + pad); ax.set_xlim(-1, n + 14)
    if s["pats"]:
        ax.text(1, hi_, " | ".join(f"{p['name']}{' ✔' if p['breakout'] else ''}" for p in s["pats"]),
                color="#fbbf24", fontsize=9, va="top")
    ax.legend(loc="upper left", fontsize=7, facecolor=bg, labelcolor="#9ca3af", edgecolor="#374151",
              bbox_to_anchor=(0, 0.93))
    ax.set_title(f"{s['pair']}  {ALERT_TF}  {s['side']}  confirm {s['score']}/5   T1 {T1_R}R  T2 {T2_R}R",
                 color="#e5e7eb", fontsize=11, loc="left")
    av.set_ylabel("Vol", color="#9ca3af", fontsize=8)
    ar.plot(range(n), s["rsi_s"].values[off:], color="#fbbf24", lw=1)
    for y in (30, 70): ar.axhline(y, color="#6b7280", ls="--", lw=0.8)
    ar.set_ylim(0, 100); ar.set_ylabel("RSI 14", color="#9ca3af", fontsize=8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf

def caption(s):
    ok = lambda k: "✅" if s["ind"][k] else "❌"
    arrow = {1: "⬆️", -1: "⬇️", 0: "➖"}
    t = [f"{s['pair']}  {s['side']} setup ({ALERT_TF})",
         f"Entry ~{fp(s['price'])}",
         f"SL {fp(s['sl'])}  ({'-' if s['side']=='LONG' else '+'}{s['risk_pct']:.1f}%)",
         f"T1 {fp(s['t1'])} ({T1_R}R)  |  T2 {fp(s['t2'])} ({T2_R}R)", "",
         "Indicators: " + " ".join(f"{ok(k)}{k}" for k in s["ind"]) + f"  (RSI {s['rsi']:.0f})",
         "Trend: " + "  ".join(f"{tf} {arrow[v]}" for tf, v in s["trends"].items()),
         f"Quality: {s['q']}/100"]
    if s["side"] == "LONG" and s["res"]: t.append(f"Upar resistance: {fp(s['res'])}")
    if s["side"] == "SHORT" and s["sup"]: t.append(f"Neeche support: {fp(s['sup'])}")
    if s["pats"]:
        t += ["", "📐 Patterns:"]
        for p in s["pats"]:
            t.append(f"{'🟢' if p['side']=='LONG' else '🔴'} {p['name']} - "
                     f"{'BREAKOUT ✅' if p['breakout'] else 'forming ⏳'} | tgt {fp(p['target'])}")
    opp = [p for p in s["pats"] if p["side"] != s["side"]]
    if opp: t += ["", f"\u26a0\ufe0f Dhyan: ulta pattern bhi hai ({opp[0]['name']}), size chhota rakho"]
    t += ["", "T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."]
    if s["side"] == "SHORT": t[-1] += " Short ke liye futures/perp chahiye."
    return "\n".join(t)

def send(s):
    if not TG_TOKEN or not TG_CHAT: print("Telegram secrets missing"); return
    cap, api = caption(s), f"https://api.telegram.org/bot{TG_TOKEN}"
    try:
        img = make_chart(s)
        r = requests.post(f"{api}/sendPhoto", data={"chat_id": TG_CHAT, "caption": cap[:1024]},
                          files={"photo": ("chart.png", img)}, timeout=30)
        if r.status_code != 200: raise RuntimeError(r.text)
        if len(cap) > 1024: requests.post(f"{api}/sendMessage", data={"chat_id": TG_CHAT, "text": cap}, timeout=20)
    except Exception as e:
        print("photo failed:", e)
        requests.post(f"{api}/sendMessage", data={"chat_id": TG_CHAT, "text": cap}, timeout=20)

# ================= RESULT TRACKER =================
open_trades, tally, day_alerts = [], {"W": 0, "L": 0, "R": 0.0}, []

def tg_text(text):
    if not TG_TOKEN or not TG_CHAT: print(text); return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text}, timeout=20)
    except Exception as e:
        print("tg_text failed:", e)

def evaluate(tr_, candles):
    """Alert ke baad ki candles chalao. Same candle me SL aur target dono touch ho to SL pehle maana jata hai (conservative).
    Return (result, r_value) ya None agar abhi trade open hai. result: SL / T1+BE / T2"""
    sgn = 1 if tr_["side"] == "LONG" else -1
    for row in candles.itertuples():
        adverse = (row.l <= tr_["stop"]) if sgn == 1 else (row.h >= tr_["stop"])
        hit1 = (row.h >= tr_["t1"]) if sgn == 1 else (row.l <= tr_["t1"])
        hit2 = (row.h >= tr_["t2"]) if sgn == 1 else (row.l <= tr_["t2"])
        if not tr_["t1_hit"]:
            if adverse: return "SL", -1.0
            if hit1:
                tr_["t1_hit"], tr_["stop"] = True, tr_["entry"]
                tg_text(f"T1 hit: {tr_['pair']} {tr_['side']} - 50% book karo, SL entry par kar do")
                if hit2: return "T2", T1_R * 0.5 + T2_R * 0.5
        else:
            if adverse: return "T1+BE", T1_R * 0.5
            if hit2: return "T2", T1_R * 0.5 + T2_R * 0.5
    return None

def track_results():
    now = time.time()
    for tr_ in list(open_trades):
        df = fetch(tr_["pair"], ALERT_TF, 120)
        if df is None: continue
        res = evaluate(tr_, df[df.t >= tr_["since"]])
        if res is None and now - tr_["since"] > 24 * 3600:
            res = ("expire", 0.0)
        if res is None: continue
        open_trades.remove(tr_)
        name, rv = res
        if name == "expire":
            tg_text(f"Expire: {tr_['pair']} {tr_['side']} - 24 ghante me na SL na T1 hit hua")
            continue
        tally["R"] += rv
        if rv > 0: tally["W"] += 1
        else: tally["L"] += 1
        tot = tally["W"] + tally["L"]
        icon = "\u274c" if name == "SL" else "\u2705"
        tg_text(f"{icon} Result: {tr_['pair']} {tr_['side']} - {name} ({rv:+.2f}R)\n"
                f"Total: {tally['W']} win / {tally['L']} loss ({tally['W'] / tot * 100:.0f}%), net {tally['R']:+.2f}R")

# ================= SCAN LOOP =================
def coin_list():
    tk = get(f"{BASE}/spot/tickers") or []
    out = []
    for x in tk:
        p = x.get("currency_pair", "")
        if not p.endswith("_USDT") or p.split("_")[0] in STABLE: continue
        if re.search(r"(3L|3S|5L|5S|UP|DOWN|BULL|BEAR)_USDT$", p): continue
        try: v, ch = float(x.get("quote_volume") or 0), abs(float(x.get("change_percentage") or 0))
        except ValueError: continue
        if v >= MIN_VOL_USDT or (ch >= MOVER_PCT and v >= MOVER_MIN_VOL): out.append((v, p))
    out.sort(reverse=True)
    return [p for _, p in out[:MAX_COINS]]

def scan():
    pairs = coin_list(); sent = 0
    stats.clear()
    def work(p):
        try: return analyze(p)
        except Exception: traceback.print_exc(); return None
    with ThreadPoolExecutor(6) as ex:
        res = [s for s in ex.map(work, pairs) if s]
    # best quality pehle: pattern breakout wale, phir bade volume wale
    res.sort(key=lambda s: s["q"], reverse=True)
    now = time.time()
    day_alerts[:] = [t for t in day_alerts if now - t < 24 * 3600]
    room = max(0, MAX_ALERTS_PER_DAY - len(day_alerts))
    if res and room == 0: stats["daily limit poori (6/24h)"] = len(res)
    for s in res[:min(MAX_ALERTS_PER_SCAN, room)]:
        last_alert[(s["pair"], s["side"])] = now; day_alerts.append(now); send(s); sent += 1
        if TRACK_RESULTS:
            open_trades.append(dict(pair=s["pair"], side=s["side"], entry=s["price"], stop=s["sl"],
                                    t1=s["t1"], t2=s["t2"], since=float(s["df"].t.iloc[-1]) + 900, t1_hit=False))
    if TRACK_RESULTS and open_trades:
        try: track_results()
        except Exception: traceback.print_exc()
    blocked = ", ".join(f"{k}: {v}" for k, v in sorted(stats.items(), key=lambda x: -x[1]))
    print(time.strftime("%H:%M:%S"), f"scanned {len(pairs)} coins, alerts {sent} (24h: {len(day_alerts)}/{MAX_ALERTS_PER_DAY}), open {len(open_trades)}"
          + (f" | roke gaye -> {blocked}" if blocked else ""), flush=True)

def main():
    end = time.time() + RUN_MINUTES * 60
    print(f"Bot start (v2 filters, mode={MODE}): alerts on {ALERT_TF}, trend filter {list(HTF)}, runs {RUN_MINUTES:.0f} min", flush=True)
    while time.time() < end:
        t0 = time.time()
        try: scan()
        except Exception: traceback.print_exc()
        time.sleep(max(5, SCAN_EVERY - (time.time() - t0)))

if __name__ == "__main__":
    main()
