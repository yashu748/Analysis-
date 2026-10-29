def main():
    """chart_loop.py isko call karta hai: ek baar poora scan"""
    scan()
      """
chart_scanner.py  (single file)
- NORMAL alert: 1D + 4h + 1h trend must agree, entry trigger on 1h. Chart = 1h.
- PUMP alert: 5m volume spike + breakout (catches fast moves like Q/USDT). Chart = 1h.
- Patterns: breakout/breakdown, engulfing, double top/bottom (LONG and SHORT).
Run:  python chart_scanner.py          -> loops every 5 minutes
      RUN_ONCE=1 python chart_scanner.py -> one scan (GitHub Actions)
Secrets (env): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (change names below if yours differ)
Needs: pip install requests pandas matplotlib
"""
import os
import io
import time
import requests
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE = "https://api.gateio.ws/api/v4"

TOP_N = 150               # top coins by 24h volume
MIN_24H_VOL = 300000      # min 24h USDT volume
LOOP_SECONDS = 300        # 5 minutes
PUMP_VOL_MULT = 3.0       # 5m volume must be 3x of last 20 candles
PUMP_MOVE_PCT = 3.0       # min move in last 3 candles (5m)
PUMP_LATE_PCT = 15.0      # skip if already up this much in last 1h (late entry)
COOLDOWN = {"NORMAL": 4 * 3600, "PUMP": 3600}
RUN_ONCE = os.getenv("RUN_ONCE") == "1"
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME", "0"))  # seconds, 0 = run forever

last_sent = {}


# ---------------- DATA ----------------
def get_pairs():
    r = requests.get(f"{BASE}/spot/tickers", timeout=15).json()
    rows = []
    for t in r:
        p = t["currency_pair"]
        if not p.endswith("_USDT"):
            continue
        if any(x in p for x in ("3L_", "3S_", "5L_", "5S_")):
            continue
        qv = float(t.get("quote_volume") or 0)
        if qv >= MIN_24H_VOL:
            rows.append((p, qv))
    rows.sort(key=lambda x: -x[1])
    return [p for p, _ in rows[:TOP_N]]


def get_df(pair, interval, limit=200):
    r = requests.get(
        f"{BASE}/spot/candlesticks",
        params={"currency_pair": pair, "interval": interval, "limit": limit},
        timeout=15,
    ).json()
    if not isinstance(r, list) or len(r) < 60:
        return None
    df = pd.DataFrame(r, columns=["t", "qv", "c", "h", "l", "o", "v", "w"])
    for col in ["c", "h", "l", "o", "v"]:
        df[col] = df[col].astype(float)
    df["t"] = pd.to_datetime(df["t"].astype(int), unit="s")
    if str(df["w"].iloc[-1]).lower() == "false":  # drop running candle
        df = df.iloc[:-1]
    df = df.reset_index(drop=True)
    return add_ind(df)


def add_ind(df):
    df["ema20"] = df.c.ewm(span=20, adjust=False).mean()
    df["ema50"] = df.c.ewm(span=50, adjust=False).mean()
    d = df.c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, 1e-9))
    tr = pd.concat(
        [df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ---------------- ANALYSIS ----------------
def trend(df, big=False):
    x = df.iloc[-1]
    if big:  # 1D: simple bigger-picture trend
        if x.c > x.ema50 and x.ema20 > x.ema50:
            return 1
        if x.c < x.ema50 and x.ema20 < x.ema50:
            return -1
        return 0
    if x.c > x.ema20 > x.ema50 and x.rsi > 50:
        return 1
    if x.c < x.ema20 < x.ema50 and x.rsi < 50:
        return -1
    return 0


def patterns(df):
    """returns list of (name, direction) direction: 1 bullish, -1 bearish"""
    out = []
    a, b = df.iloc[-1], df.iloc[-2]
    hi20 = df.h.iloc[-21:-1].max()
    lo20 = df.l.iloc[-21:-1].min()
    if a.c > hi20:
        out.append(("Breakout (20-candle high)", 1))
    if a.c < lo20:
        out.append(("Breakdown (20-candle low)", -1))
    if a.c > a.o and b.c < b.o and a.c > b.o and a.o < b.c:
        out.append(("Bullish Engulfing", 1))
    if a.c < a.o and b.c > b.o and a.c < b.o and a.o > b.c:
        out.append(("Bearish Engulfing", -1))

    w = df.iloc[-40:].reset_index(drop=True)
    h, l = w.h.values, w.l.values
    n = len(w)
    ph = [i for i in range(2, n - 2) if h[i] == max(h[i - 2:i + 3])]
    pl = [i for i in range(2, n - 2) if l[i] == min(l[i - 2:i + 3])]
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        neck = min(l[i1:i2 + 1])
        if i2 - i1 >= 5 and abs(h[i1] - h[i2]) / h[i1] < 0.006 and a.c < neck:
            out.append(("Double Top breakdown (M)", -1))
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        neck = max(h[i1:i2 + 1])
        if i2 - i1 >= 5 and abs(l[i1] - l[i2]) / l[i1] < 0.006 and a.c > neck:
            out.append(("Double Bottom breakout (W)", 1))
    return out


def levels(df, direction, entry):
    atr = df.atr.iloc[-1]
    if direction == 1:
        sl = df.l.iloc[-5:].min() - 0.1 * atr
        if entry - sl > 2.5 * atr:
            sl = entry - 2.5 * atr
        risk = entry - sl
        return sl, entry + 1.5 * risk, entry + 3 * risk
    sl = df.h.iloc[-5:].max() + 0.1 * atr
    if sl - entry > 2.5 * atr:
        sl = entry + 2.5 * atr
    risk = sl - entry
    return sl, entry - 1.5 * risk, entry - 3 * risk


def normal_signal(pair):
    h1 = get_df(pair, "1h", 200)
    if h1 is None:
        return None
    t1 = trend(h1)
    if t1 == 0:
        return None
    x, prev = h1.iloc[-1], h1.iloc[-2]
    if (t1 == 1 and x.rsi > 75) or (t1 == -1 and x.rsi < 25):
        return None
    pats = [p for p in patterns(h1) if p[1] == t1]
    volavg = h1.v.iloc[-21:-1].mean()
    if t1 == 1:
        cross = x.c > x.ema20 and prev.c <= h1.ema20.iloc[-2]
    else:
        cross = x.c < x.ema20 and prev.c >= h1.ema20.iloc[-2]
    if cross and x.v > 1.2 * volavg:
        pats.append(("EMA20 reclaim + volume", t1))
    if not pats:
        return None
    h4 = get_df(pair, "4h", 200)
    if h4 is None or trend(h4) != t1:
        return None
    d1 = get_df(pair, "1d", 250)
    if d1 is None or trend(d1, big=True) != t1:
        return None
    entry = x.c
    sl, tp1, tp2 = levels(h1, t1, entry)
    return dict(kind="NORMAL", pair=pair, dir=t1, entry=entry, sl=sl, tp1=tp1,
                tp2=tp2, notes=", ".join(p[0] for p in pats),
                tf="1D: OK | 4h: OK | 1h: Entry", df=h1)


def pump_signal(pair):
    m5 = get_df(pair, "5m", 120)
    if m5 is None:
        return None
    x = m5.iloc[-1]
    avgv = m5.v.iloc[-21:-1].mean()
    if avgv <= 0 or x.v < PUMP_VOL_MULT * avgv:
        return None
    move = (x.c / m5.o.iloc[-3] - 1) * 100
    if move < PUMP_MOVE_PCT:
        return None
    if x.c <= m5.h.iloc[-21:-1].max():
        return None
    late = (x.c / m5.c.iloc[-13] - 1) * 100
    if late > PUMP_LATE_PCT:
        return None
    h1 = get_df(pair, "1h", 200)
    if h1 is None:
        return None
    xh = h1.iloc[-1]
    if xh.c < xh.ema20 < xh.ema50:  # 1h clearly bearish -> skip
        return None
    entry = x.c
    sl = max(m5.l.iloc[-4:].min(), entry - 2 * x.atr)  # tight SL
    risk = entry - sl
    return dict(kind="PUMP", pair=pair, dir=1, entry=entry, sl=sl,
                tp1=entry + 1.5 * risk, tp2=entry + 3 * risk,
                notes=f"5m volume {x.v / avgv:.1f}x, +{move:.1f}% in 15 min, "
                      f"+{late:.1f}% in last 1h",
                tf="5m: PUMP trigger | 1h: not bearish", df=h1)


# ---------------- ALERT ----------------
def fmt(v):
    return f"{v:.6g}"


def make_chart(sig):
    d = sig["df"].iloc[-60:].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in d.iterrows():
        col = "#26a69a" if r.c >= r.o else "#ef5350"
        ax.vlines(i, r.l, r.h, color=col, lw=1)
        ax.bar(i, abs(r.c - r.o) or r.c * 1e-5, bottom=min(r.o, r.c),
               color=col, width=0.7)
    ax.plot(d.ema20, lw=1, label="EMA20")
    ax.plot(d.ema50, lw=1, label="EMA50")
    for name, val, col in [("Entry", sig["entry"], "gray"), ("SL", sig["sl"], "red"),
                           ("TP1", sig["tp1"], "green"), ("TP2", sig["tp2"], "green")]:
        ax.axhline(val, color=col, ls="--", lw=1)
        ax.text(len(d) - 1, val, f" {name} {fmt(val)}", color=col,
                va="center", fontsize=8)
    side = "LONG" if sig["dir"] == 1 else "SHORT"
    ax.set_title(f"{sig['pair']} 1H - {side} ({sig['kind']})")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-1, len(d) + 8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def send(sig):
    side = "LONG 🟢" if sig["dir"] == 1 else "SHORT 🔴"
    head = "🚀 PUMP ALERT (HIGH RISK)" if sig["kind"] == "PUMP" else "📊 TREND ALERT"
    cap = (
        f"{head}\n{sig['pair']} - {side}\n\n"
        f"Timeframes: {sig['tf']}\n"
        f"Reason: {sig['notes']}\n\n"
        f"Entry: {fmt(sig['entry'])}\n"
        f"SL: {fmt(sig['sl'])}\n"
        f"TP1: {fmt(sig['tp1'])}\n"
        f"TP2: {fmt(sig['tp2'])}\n\n"
        "Not financial advice. Use small size."
    )
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": cap[:1000]},
        files={"photo": ("chart.png", make_chart(sig))},
        timeout=30,
    )


# ---------------- MAIN ----------------
def scan():
    pairs = get_pairs()
    print(f"Scanning {len(pairs)} coins...")
    for p in pairs:
        for kind, fn in (("PUMP", pump_signal), ("NORMAL", normal_signal)):
            try:
                if time.time() - last_sent.get((p, kind), 0) < COOLDOWN[kind]:
                    continue
                sig = fn(p)
                if sig:
                    send(sig)
                    last_sent[(p, kind)] = time.time()
                    print("ALERT", p, kind)
            except Exception as e:
                print("error", p, kind, e)
        time.sleep(0.15)


if __name__ == "__main__":
    begin = time.time()
    while True:
        start = time.time()
        try:
            scan()
        except Exception as e:
            print("scan error", e)
        if RUN_ONCE:
            break
        if MAX_RUNTIME and time.time() - begin > MAX_RUNTIME:
            break
        time.sleep(max(10, LOOP_SECONDS - (time.time() - start)))
          
