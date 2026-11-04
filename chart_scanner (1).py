#!/usr/bin/env python3
"""
chart_scanner.py  -  Gate.io multi-timeframe scanner (single file, v2 filters)

ANALYSIS : 1D + 4H + 1H  -> trend filter (direction must agree)
ALERT    : 15m           -> entry trigger + 15m chart sent to Telegram

NEW in v2 (galat signal rokne ke filters):
  1. Ulta pattern block   : LONG par Double Top, SHORT par Double Bottom -> alert nahi
  2. Resistance/Support   : T1 se pehle resistance (LONG) / support (SHORT) aaye -> alert nahi
  3. Pump candle block    : badi candle / EMA21 se bahut door / tez run -> alert nahi (chasing band)
  4. RSI limit            : LONG RSI <= 62, SHORT RSI >= 38 ; 1h RSI extreme par bhi block
  5. Structure SL         : SL swing low/high ke bahar (ATR ke saath), max risk 4%
  6. Illiquid coin skip   : min 24h volume 1,000,000 USDT

Run:      python chart_scanner.py          (sends Telegram alerts)
Test run: python chart_scanner.py --dry    (prints only, no Telegram, skip reasons dikhata hai)
Needs:    pip install requests pandas numpy matplotlib
Secrets:  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (env vars / GitHub secrets)
"""
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ============================ CONFIG ============================
API = "https://api.gateio.ws/api/v4"

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
            or os.getenv("BOT_TOKEN") or "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or ""

ENTRY_TF = "15m"          # alert / chart timeframe
MAX_PAIRS = 150           # scan top N USDT pairs by volume
MIN_QUOTE_VOL = 1_000_000 # min 24h volume in USDT (illiquid coins skip)
VOL_MULT = 1.3            # 15m volume must be >= 1.3x average of last 20
VOL_MAX = 4.0             # volume itna zyada = blow-off/pump, skip
SL_ATR = 1.5              # stop loss = 1.5 x ATR(14) on 15m (minimum distance)
MAX_RISK_PCT = 4.0        # SL distance isse zyada ho to skip
TP1_R, TP2_R = 1.5, 3.0   # targets in multiples of risk
MIN_SCORE = 70            # send alert only if score >= this
MAX_SIGNAL_AGE_MIN = 14   # signal candle must have closed within this many minutes
COOLDOWN_MIN = 120        # same coin + same side: no repeat within this time
COOLDOWN_FILE = "sent_alerts.json"
CALL_SLEEP = 0.08         # pause between API calls (rate limit)

# --- v2 filter settings ---
LONG_RSI_MAX = 62         # LONG ke liye 15m RSI isse upar nahi
SHORT_RSI_MIN = 38        # SHORT ke liye 15m RSI isse neeche nahi
HTF_RSI_MAX = 70          # 1h RSI isse upar LONG block
HTF_RSI_MIN = 30          # 1h RSI isse neeche SHORT block
BIG_CANDLE_ATR = 2.5      # last candle range > 2.5 ATR = pump candle
BIG_BODY_ATR = 2.0        # last candle body > 2 ATR = pump candle
EXTENSION_ATR = 2.0       # close EMA21 se 2 ATR se zyada door = chasing
RUN_ATR = 4.0             # pichhli 5 candles me 4 ATR se zyada move = late entry

STABLES = {"USDC", "TUSD", "FDUSD", "DAI", "USDD", "USDE", "PYUSD", "USD1", "BUSD"}
SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
DRY = "--dry" in sys.argv


# ============================ DATA ============================
def get_candles(pair, interval, limit=200):
    """Return DataFrame of CLOSED candles only (unclosed last candle removed)."""
    for _ in range(3):
        try:
            r = requests.get(
                f"{API}/spot/candlesticks",
                params={"currency_pair": pair, "interval": interval, "limit": limit},
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            rows = r.json()
            if not rows:
                return None
            cols = ["ts", "quote_vol", "close", "high", "low", "open", "base_vol", "closed"]
            df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
            for c in ["ts", "quote_vol", "close", "high", "low", "open"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["ts", "quote_vol", "close", "high", "low", "open"])
            df = df.sort_values("ts").reset_index(drop=True)
            now = time.time()
            df = df[df["ts"] + SECONDS[interval] <= now].reset_index(drop=True)
            return df if len(df) else None
        except Exception:
            time.sleep(1)
    return None


def get_universe():
    r = requests.get(f"{API}/spot/tickers", timeout=20)
    r.raise_for_status()
    out = []
    for t in r.json():
        pair = t.get("currency_pair", "")
        if not pair.endswith("_USDT"):
            continue
        base = pair[:-5]
        if base in STABLES or base.endswith(("3L", "3S", "5L", "5S")):
            continue
        try:
            qv = float(t.get("quote_volume") or 0)
        except ValueError:
            continue
        if qv >= MIN_QUOTE_VOL:
            out.append((pair, qv))
    out.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in out[:MAX_PAIRS]]


# ============================ INDICATORS ============================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd_hist(c):
    m = ema(c, 12) - ema(c, 26)
    return m - ema(m, 9)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ============================ STRUCTURE (pivots / patterns) ============================
def pivot_points(series, k, kind):
    """Return list of (index, price) swing highs ('high') or swing lows ('low').
    A pivot needs k candles on each side, so the last k candles can't be pivots."""
    vals = series.values
    out = []
    n = len(vals)
    for i in range(k, n - k):
        window = vals[i - k:i + k + 1]
        if kind == "high" and vals[i] == window.max():
            out.append((i, float(vals[i])))
        elif kind == "low" and vals[i] == window.min():
            out.append((i, float(vals[i])))
    return out


def opposite_pattern(df, side):
    """LONG -> Double Top forming ho to naam return karo. SHORT -> Double Bottom."""
    d = df.tail(90).reset_index(drop=True)
    n = len(d)
    last_close = float(d["close"].iloc[-1])

    if side == "LONG":
        pv = pivot_points(d["high"], 2, "high")
        if len(pv) < 2:
            return None
        (i1, h1), (i2, h2) = pv[-2], pv[-1]
        top = max(h1, h2)
        if i2 < n - 45 or (i2 - i1) < 5:
            return None
        if abs(h1 - h2) / top > 0.012:          # dono top lagbhag barabar
            return None
        trough = float(d["low"].iloc[i1:i2 + 1].min())
        if (min(h1, h2) - trough) / min(h1, h2) < 0.015:   # beech me dip kam se kam 1.5%
            return None
        if last_close > top * 1.002:            # breakout ho gaya = pattern fail, block nahi
            return None
        return "Double Top"

    pv = pivot_points(d["low"], 2, "low")
    if len(pv) < 2:
        return None
    (i1, l1), (i2, l2) = pv[-2], pv[-1]
    bot = min(l1, l2)
    if i2 < n - 45 or (i2 - i1) < 5:
        return None
    if abs(l1 - l2) / max(l1, l2) > 0.012:
        return None
    peak = float(d["high"].iloc[i1:i2 + 1].max())
    if (peak - max(l1, l2)) / peak < 0.015:
        return None
    if last_close < bot * 0.998:
        return None
    return "Double Bottom"


def nearest_level(df15, df1h, side, entry):
    """LONG -> entry ke upar sabse paas resistance. SHORT -> neeche sabse paas support.
    15m (last 120) aur 1h (last 150) ke swing points se."""
    levels = []
    kind = "high" if side == "LONG" else "low"
    col = "high" if side == "LONG" else "low"
    sources = [(df15.tail(120).reset_index(drop=True), 2)]
    if df1h is not None and len(df1h) > 20:
        sources.append((df1h.tail(150).reset_index(drop=True), 3))
    for d, k in sources:
        levels += [p for _, p in pivot_points(d[col], k, kind)]
    if side == "LONG":
        above = [x for x in levels if x > entry * 1.0005]
        return min(above) if above else None
    below = [x for x in levels if x < entry * 0.9995]
    return max(below) if below else None


# ============================ ANALYSIS ============================
def tf_bias(df):
    """Return (bias, score). bias: +1 up, -1 down, 0 flat/unknown."""
    if df is None or len(df) < 55:
        return 0, 0
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    score = 0
    score += 1 if c.iloc[-1] > e20.iloc[-1] else -1
    score += 1 if e20.iloc[-1] > e50.iloc[-1] else -1
    score += 1 if macd_hist(c).iloc[-1] > 0 else -1
    score += 1 if rsi(c).iloc[-1] > 50 else -1
    bias = 1 if score >= 2 else (-1 if score <= -2 else 0)
    return bias, score


def trigger_15m(df, skips=None):
    """Entry trigger on the LAST CLOSED 15m candle. Returns dict or None.
    skips: optional list, filter ka reason yahan append hota hai (debug ke liye)."""
    def skip(msg):
        if skips is not None:
            skips.append(msg)
        return None

    if df is None or len(df) < 60:
        return None
    c = df["close"]
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    rs = rsi(c)
    a = atr(df)
    vol_avg = df["quote_vol"].rolling(20).mean().shift(1)
    if not vol_avg.iloc[-1] or np.isnan(vol_avg.iloc[-1]):
        return None
    vr = df["quote_vol"].iloc[-1] / vol_avg.iloc[-1]

    hh = df["high"].iloc[-21:-1].max()
    ll = df["low"].iloc[-21:-1].min()
    cross_up = e9.iloc[-2] <= e21.iloc[-2] and e9.iloc[-1] > e21.iloc[-1]
    cross_dn = e9.iloc[-2] >= e21.iloc[-2] and e9.iloc[-1] < e21.iloc[-1]
    brk_up = c.iloc[-1] > hh
    brk_dn = c.iloc[-1] < ll
    r_now = float(rs.iloc[-1])

    side, why = None, ""
    if (cross_up or brk_up) and 45 <= r_now <= LONG_RSI_MAX and VOL_MULT <= vr <= VOL_MAX \
            and c.iloc[-1] > e50.iloc[-1]:
        side = "LONG"
        why = "EMA9/21 cross up" if cross_up else "20-candle breakout up"
    elif (cross_dn or brk_dn) and SHORT_RSI_MIN <= r_now <= 55 and VOL_MULT <= vr <= VOL_MAX \
            and c.iloc[-1] < e50.iloc[-1]:
        side = "SHORT"
        why = "EMA9/21 cross down" if cross_dn else "20-candle breakdown"
    if not side:
        return None

    atr_now = float(a.iloc[-1])
    if atr_now <= 0:
        return None
    entry = float(c.iloc[-1])
    sign = 1 if side == "LONG" else -1

    # ---- FILTER 3: pump candle / chasing ----
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    if rng > BIG_CANDLE_ATR * atr_now or body > BIG_BODY_ATR * atr_now:
        return skip("pump candle (badi candle) - retest ka wait")
    if abs(entry - float(e21.iloc[-1])) > EXTENSION_ATR * atr_now:
        return skip("price EMA21 se bahut door (chasing)")
    run = (entry - float(c.iloc[-6])) * sign
    if run > RUN_ATR * atr_now:
        return skip("pichhli 5 candles me tez run (late entry)")

    # ---- FILTER 5: structure based SL ----
    atr_risk = SL_ATR * atr_now
    if side == "LONG":
        struct_sl = float(df["low"].iloc[-10:].min()) - 0.25 * atr_now
        sl = min(entry - atr_risk, struct_sl)
    else:
        struct_sl = float(df["high"].iloc[-10:].max()) + 0.25 * atr_now
        sl = max(entry + atr_risk, struct_sl)
    risk = abs(entry - sl)
    if risk <= 0 or risk / entry * 100 > MAX_RISK_PCT:
        return skip("SL bahut door (risk > %.1f%%)" % MAX_RISK_PCT)

    return {
        "side": side, "why": why, "entry": entry,
        "sl": sl,
        "tp1": entry + sign * TP1_R * risk,
        "tp2": entry + sign * TP2_R * risk,
        "rsi": r_now, "vr": float(vr), "level": None,
    }


def structure_filters(sig, df15, df1h):
    """Return reason (string) agar alert block hona chahiye, warna None."""
    side = sig["side"]

    # ---- FILTER 1: ulta pattern ----
    pat = opposite_pattern(df15, side)
    if pat:
        return f"{pat} forming - {side} ke against"
    if df1h is not None and len(df1h) > 40:
        pat = opposite_pattern(df1h, side)
        if pat:
            return f"1h par {pat} forming - {side} ke against"

    # ---- FILTER 4: higher timeframe RSI extreme ----
    if df1h is not None and len(df1h) > 20:
        r1h = float(rsi(df1h["close"]).iloc[-1])
        if side == "LONG" and r1h > HTF_RSI_MAX:
            return f"1h RSI {r1h:.0f} overbought"
        if side == "SHORT" and r1h < HTF_RSI_MIN:
            return f"1h RSI {r1h:.0f} oversold"

    # ---- FILTER 2: resistance / support T1 se pehle ----
    lvl = nearest_level(df15, df1h, side, sig["entry"])
    if lvl is not None:
        blocked = (lvl < sig["tp1"]) if side == "LONG" else (lvl > sig["tp1"])
        if blocked:
            name = "resistance" if side == "LONG" else "support"
            return f"{name} {fmt(lvl)} T1 {fmt(sig['tp1'])} se pehle hai"
    sig["level"] = lvl
    return None


def fmt(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}".rstrip("0")


def label(b):
    return "UP" if b > 0 else ("DOWN" if b < 0 else "FLAT")


# ============================ CHART ============================
def make_chart(df, pair, sig):
    d = df.tail(80).reset_index(drop=True)
    c = df["close"]
    e9, e21, e50 = ema(c, 9).tail(80).values, ema(c, 21).tail(80).values, ema(c, 50).tail(80).values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for i, r in d.iterrows():
        col = "#26a69a" if r["close"] >= r["open"] else "#ef5350"
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1)
        h = max(abs(r["close"] - r["open"]), 1e-12)
        ax.add_patch(plt.Rectangle((i - 0.35, min(r["open"], r["close"])), 0.7, h, color=col))
    x = np.arange(len(d))
    ax.plot(x, e9, color="#ffd54f", lw=1, label="EMA9")
    ax.plot(x, e21, color="#42a5f5", lw=1, label="EMA21")
    ax.plot(x, e50, color="#ab47bc", lw=1, label="EMA50")

    n = len(d)
    green, red = "#26a69a", "#ef5350"
    good, bad = (green, red)
    ax.axhspan(sig["entry"], sig["tp2"], xmin=0.78, xmax=1, color=good, alpha=0.18)
    ax.axhspan(sig["entry"], sig["sl"], xmin=0.78, xmax=1, color=bad, alpha=0.18)
    lines = [("ENTRY", sig["entry"], "#ffffff"), ("SL", sig["sl"], bad),
             ("TP1", sig["tp1"], good), ("TP2", sig["tp2"], good)]
    if sig.get("level"):
        lines.append(("RES" if sig["side"] == "LONG" else "SUP", sig["level"], "#ffb74d"))
    for name, val, col in lines:
        ax.axhline(val, color=col, lw=0.9, ls="--")
        ax.text(n + 0.5, val, f" {name} {fmt(val)}", color=col, va="center", fontsize=8)

    lo = min(d["low"].min(), sig["sl"])
    hi = max(d["high"].max(), sig["tp2"])
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-1, n + 12)
    ax.set_title(f"{pair.replace('_', '/')}  {ENTRY_TF}  {sig['side']}", color="white", fontsize=12)
    ax.tick_params(colors="#9e9e9e", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333333")
    ax.legend(loc="upper left", facecolor="#0e1117", labelcolor="white", fontsize=8, frameon=False)
    ax.grid(color="#1e232d", lw=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================ TELEGRAM ============================
def send_photo(img, caption):
    if DRY or not TG_TOKEN or not TG_CHAT:
        print("[DRY/NO TOKEN]\n" + caption + "\n")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    try:
        r = requests.post(url, data={"chat_id": TG_CHAT, "caption": caption[:1000]},
                          files={"photo": ("chart.png", img)}, timeout=40)
        if r.status_code != 200:
            print("Telegram error:", r.text[:200])
    except Exception as e:
        print("Telegram exception:", e)


def load_sent():
    try:
        with open(COOLDOWN_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_sent(d):
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


# ============================ MAIN ============================
def main():
    pairs = get_universe()
    print(f"Scanning {len(pairs)} pairs | analysis 1D/4H/1H | alert {ENTRY_TF}")
    sent = load_sent()
    now = time.time()
    found = 0

    for pair in pairs:
        try:
            df15 = get_candles(pair, ENTRY_TF, 150)
            time.sleep(CALL_SLEEP)
            if df15 is None:
                continue
            closed_at = df15["ts"].iloc[-1] + SECONDS[ENTRY_TF]
            if now - closed_at > MAX_SIGNAL_AGE_MIN * 60:
                continue

            skips = []
            sig = trigger_15m(df15, skips)
            if not sig:
                if DRY and skips:
                    print(f"SKIP {pair}: {skips[0]}")
                continue

            # higher timeframes only for coins that already have a 15m trigger
            df1h = get_candles(pair, "1h", 200); time.sleep(CALL_SLEEP)
            b1h, _ = tf_bias(df1h)
            b4h, _ = tf_bias(get_candles(pair, "4h", 200)); time.sleep(CALL_SLEEP)
            b1d, _ = tf_bias(get_candles(pair, "1d", 200)); time.sleep(CALL_SLEEP)

            want = 1 if sig["side"] == "LONG" else -1
            # 4H and 1H must agree, 1D must not be against
            if b4h != want or b1h != want or b1d == -want:
                continue

            # v2 structure filters (pattern, resistance/support, HTF RSI)
            block = structure_filters(sig, df15, df1h)
            if block:
                if DRY:
                    print(f"SKIP {pair} {sig['side']}: {block}")
                continue

            score = 50
            score += 10 if b1h == want else 0
            score += 10 if b4h == want else 0
            score += 10 if b1d == want else 0
            score += int(min(sig["vr"], 3) * 5)
            score = min(score, 95)
            if score < MIN_SCORE:
                continue

            key = f"{pair}_{sig['side']}"
            if now - sent.get(key, 0) < COOLDOWN_MIN * 60:
                continue

            e = sig["entry"]
            pct = lambda v: f"{(v - e) / e * 100:+.2f}%"
            icon = "🟢" if sig["side"] == "LONG" else "🔴"
            lvl_line = ""
            if sig.get("level"):
                nm = "Resistance" if sig["side"] == "LONG" else "Support"
                lvl_line = f"{nm}: {fmt(sig['level'])} ({pct(sig['level'])})\n"
            caption = (
                f"{icon} {sig['side']} | {pair.replace('_', '/')} | {ENTRY_TF}\n"
                f"Entry: {fmt(e)}\n"
                f"SL: {fmt(sig['sl'])} ({pct(sig['sl'])})\n"
                f"TP1: {fmt(sig['tp1'])} ({pct(sig['tp1'])})\n"
                f"TP2: {fmt(sig['tp2'])} ({pct(sig['tp2'])})\n"
                f"{lvl_line}"
                f"Trend: 1D {label(b1d)} | 4H {label(b4h)} | 1H {label(b1h)}\n"
                f"Trigger: {sig['why']} | Vol x{sig['vr']:.1f} | RSI {sig['rsi']:.0f}\n"
                f"Score: {score}/100\n"
                f"T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."
            )
            send_photo(make_chart(df15, pair, sig), caption)
            sent[key] = now
            found += 1
        except Exception as ex:
            print(f"{pair} error: {ex}")

    save_sent(sent)
    print(f"Done. Alerts sent: {found}")


if __name__ == "__main__":
    main()
            
