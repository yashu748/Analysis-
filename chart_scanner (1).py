#!/usr/bin/env python3
"""
chart_scanner.py  -  Gate.io multi-timeframe scanner (single file)

ANALYSIS : 1D + 4H + 1H  -> trend filter (direction must agree)
ALERT    : 15m           -> entry trigger + 15m chart sent to Telegram

Run:      python chart_scanner.py          (sends Telegram alerts)
Test run: python chart_scanner.py --dry    (prints only, no Telegram)
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
MIN_QUOTE_VOL = 300_000   # min 24h volume in USDT
VOL_MULT = 1.3            # 15m volume must be >= 1.3x average of last 20
SL_ATR = 1.5              # stop loss = 1.5 x ATR(14) on 15m
TP1_R, TP2_R = 1.5, 3.0   # targets in multiples of risk
MIN_SCORE = 70            # send alert only if score >= this
MAX_SIGNAL_AGE_MIN = 14   # signal candle must have closed within this many minutes
COOLDOWN_MIN = 120        # same coin + same side: no repeat within this time
COOLDOWN_FILE = "sent_alerts.json"
CALL_SLEEP = 0.08         # pause between API calls (rate limit)

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
            df = df.dropna().sort_values("ts").reset_index(drop=True)
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


def trigger_15m(df):
    """Entry trigger on the LAST CLOSED 15m candle. Returns dict or None."""
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
    if (cross_up or brk_up) and 45 <= r_now <= 72 and vr >= VOL_MULT and c.iloc[-1] > e50.iloc[-1]:
        side = "LONG"
        why = "EMA9/21 cross up" if cross_up else "20-candle breakout up"
    elif (cross_dn or brk_dn) and 28 <= r_now <= 55 and vr >= VOL_MULT and c.iloc[-1] < e50.iloc[-1]:
        side = "SHORT"
        why = "EMA9/21 cross down" if cross_dn else "20-candle breakdown"
    if not side:
        return None

    entry = float(c.iloc[-1])
    risk = SL_ATR * float(a.iloc[-1])
    if risk <= 0:
        return None
    sign = 1 if side == "LONG" else -1
    return {
        "side": side, "why": why, "entry": entry,
        "sl": entry - sign * risk,
        "tp1": entry + sign * TP1_R * risk,
        "tp2": entry + sign * TP2_R * risk,
        "rsi": r_now, "vr": float(vr),
    }


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
    for name, val, col in [("ENTRY", sig["entry"], "#ffffff"), ("SL", sig["sl"], bad),
                           ("TP1", sig["tp1"], good), ("TP2", sig["tp2"], good)]:
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

            sig = trigger_15m(df15)
            if not sig:
                continue

            # higher timeframes only for coins that already have a 15m trigger
            b1h, _ = tf_bias(get_candles(pair, "1h", 200)); time.sleep(CALL_SLEEP)
            b4h, _ = tf_bias(get_candles(pair, "4h", 200)); time.sleep(CALL_SLEEP)
            b1d, _ = tf_bias(get_candles(pair, "1d", 200)); time.sleep(CALL_SLEEP)

            want = 1 if sig["side"] == "LONG" else -1
            # 4H and 1H must agree, 1D must not be against
            if b4h != want or b1h != want or b1d == -want:
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
            caption = (
                f"{icon} {sig['side']} | {pair.replace('_', '/')} | {ENTRY_TF}\n"
                f"Entry: {fmt(e)}\n"
                f"SL: {fmt(sig['sl'])} ({pct(sig['sl'])})\n"
                f"TP1: {fmt(sig['tp1'])} ({pct(sig['tp1'])})\n"
                f"TP2: {fmt(sig['tp2'])} ({pct(sig['tp2'])})\n"
                f"Trend: 1D {label(b1d)} | 4H {label(b4h)} | 1H {label(b1h)}\n"
                f"Trigger: {sig['why']} | Vol x{sig['vr']:.1f} | RSI {sig['rsi']:.0f}\n"
                f"Score: {score}/100\n"
                f"Alert only, not advice. Always use SL."
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
