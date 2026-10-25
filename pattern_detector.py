"""pattern_detector.py - chart + candlestick patterns ek jagah.
Chart patterns: Double/Triple Top & Bottom, Head&Shoulders (+Inverse), Triangles
(Ascending/Descending/Symmetrical), Rising/Falling Wedge, Channels, Rectangle,
Broadening, Bull/Bear Flag, Cup&Handle.
Candles: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star,
Three White Soldiers / Three Black Crows.
Sab rule-based hai, guarantee nahi."""


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
