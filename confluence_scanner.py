"""
Gate.io Confluence Scanner
- Saare USDT coins scan karta hai (volume ke hisaab se top N)
- 9 checks: EMA trend, EMA200, RSI, MACD, Supertrend, VWAP, Volume spike,
  Candle pattern/Breakout, Bollinger mid
- Score >= MIN_SCORE hone par signal, phir us coin ki apni history pe backtest
- Backtest win-rate achha ho tabhi Telegram alert (entry, SL, TP + chart)

Env vars: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (optional: TF, TOP_N, MIN_SCORE, MIN_WIN)
Install: pip install requests pandas numpy matplotlib
"""
import os, io, json, time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "https://api.gateio.ws/api/v4"
TF = os.getenv("TF", "1h")
TOP_N = int(os.getenv("TOP_N", "80"))
MIN_VOL = float(os.getenv("MIN_VOL", "1000000"))   # 24h USDT volume
MIN_SCORE = int(os.getenv("MIN_SCORE", "6"))       # 9 me se
MIN_WIN = float(os.getenv("MIN_WIN", "0.5"))       # backtest win-rate
MIN_TRADES = int(os.getenv("MIN_TRADES", "5"))
HOLD = 24                                          # candles tak trade hold
SL_ATR, TP_ATR = 1.5, 3.0                          # risk:reward = 1:2
COOLDOWN_H = 6
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
STATE = "sent.json"


def api(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_candles(pair):
    d = api("/spot/candlesticks",
            {"currency_pair": pair, "interval": TF, "limit": 500})
    df = pd.DataFrame(d, columns=["t", "qv", "c", "h", "l", "o", "v", "closed"])
    for col in ["t", "qv", "c", "h", "l", "o", "v"]:
        df[col] = df[col].astype(float)
    df = df[df["closed"].astype(str).str.lower() == "true"].reset_index(drop=True)
    return df


# ---------------- Indicators ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def atr(df, n=14):
    pc = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend_up(df, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df.h + df.l) / 2).values
    c = df.c.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    up = np.ones(len(df), dtype=bool)
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if up[i - 1]:
            up[i] = c[i] >= flb[i]
        else:
            up[i] = c[i] > fub[i]
    return pd.Series(up, index=df.index)


def add_indicators(df):
    df["ema20"], df["ema50"], df["ema200"] = ema(df.c, 20), ema(df.c, 50), ema(df.c, 200)
    df["rsi"] = rsi(df.c)
    df["atr"] = atr(df)
    m = ema(df.c, 12) - ema(df.c, 26)
    df["macd"], df["macd_sig"] = m, ema(m, 9)
    df["hist"] = df.macd - df.macd_sig
    df["bb_mid"] = df.c.rolling(20).mean()
    typ = (df.h + df.l + df.c) / 3
    df["vwap"] = (typ * df.v).rolling(24).sum() / df.v.rolling(24).sum()
    df["vol_spike"] = df.v > 1.5 * df.v.rolling(20).mean()
    df["st_up"] = supertrend_up(df)

    body = (df.c - df.o).abs()
    upper = df.h - df[["c", "o"]].max(axis=1)
    lower = df[["c", "o"]].min(axis=1) - df.l
    engulf = (df.c.shift(1) < df.o.shift(1)) & (df.c > df.o) & \
             (df.c >= df.o.shift(1)) & (df.o <= df.c.shift(1))
    hammer = (lower >= 2 * body) & (upper <= body) & (body > 0)
    breakout = df.c > df.h.rolling(20).max().shift(1)
    df["pattern"] = engulf | hammer | breakout

    checks = pd.DataFrame({
        "EMA20>50": (df.ema20 > df.ema50) & (df.c > df.ema50),
        "Price>EMA200": df.c > df.ema200,
        "RSI 50-70": (df.rsi > 50) & (df.rsi < 70),
        "MACD bullish": (df.macd > df.macd_sig) & (df.hist > df.hist.shift(1)),
        "Supertrend up": df.st_up,
        "Price>VWAP": df.c > df.vwap,
        "Volume spike": df.vol_spike,
        "Pattern/Breakout": df.pattern,
        "Price>BB mid": df.c > df.bb_mid,
    }).fillna(False)
    df["score"] = checks.sum(axis=1)
    df["signal"] = (df.score >= MIN_SCORE) & (df.score.shift(1) < MIN_SCORE)
    return df, checks


# ---------------- Backtest ----------------
def backtest(df):
    wins = losses = 0
    sig = df.signal.values
    for i in np.where(sig)[0][:-1] if sig[-1] else np.where(sig)[0]:
        a, e = df.atr.iloc[i], df.c.iloc[i]
        if np.isnan(a):
            continue
        sl, tp = e - SL_ATR * a, e + TP_ATR * a
        for j in range(i + 1, min(i + 1 + HOLD, len(df))):
            if df.l.iloc[j] <= sl:      # same candle me dono hit = loss (safe side)
                losses += 1
                break
            if df.h.iloc[j] >= tp:
                wins += 1
                break
    n = wins + losses
    return n, (wins / n if n else 0.0)


# ---------------- Telegram ----------------
def chart_png(df, pair, entry, sl, tp):
    d = df.tail(80).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in d.iterrows():
        col = "#26a69a" if r.c >= r.o else "#ef5350"
        ax.vlines(i, r.l, r.h, color=col, lw=1)
        ax.bar(i, abs(r.c - r.o), bottom=min(r.o, r.c), color=col, width=0.7)
    ax.plot(d.index, d.ema20, lw=1, label="EMA20")
    ax.plot(d.index, d.ema50, lw=1, label="EMA50")
    ax.axhline(entry, color="blue", ls="--", lw=1)
    ax.axhline(sl, color="red", ls="--", lw=1)
    ax.axhline(tp, color="green", ls="--", lw=1)
    ax.set_title(f"{pair} {TF}  Entry/SL/TP")
    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def send(text, photo=None):
    if not TOKEN or not CHAT:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TOKEN}/"
    if photo:
        requests.post(url + "sendPhoto", data={"chat_id": CHAT, "caption": text},
                      files={"photo": photo}, timeout=30)
    else:
        requests.post(url + "sendMessage", data={"chat_id": CHAT, "text": text}, timeout=30)


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


# ---------------- Main ----------------
def main():
    state = load_state()
    tick = api("/spot/tickers")
    coins = []
    for t in tick:
        p = t["currency_pair"]
        if not p.endswith("_USDT") or any(x in p for x in ("3L_", "3S_", "5L_", "5S_")):
            continue
        qv = float(t.get("quote_volume") or 0)
        if qv >= MIN_VOL:
            coins.append((qv, p))
    coins = [p for _, p in sorted(coins, reverse=True)[:TOP_N]]
    print(f"Scanning {len(coins)} coins on {TF}")

    for pair in coins:
        try:
            df = get_candles(pair)
            if len(df) < 210:
                continue
            df, checks = add_indicators(df)
            if not df.signal.iloc[-1]:
                continue
            if time.time() - state.get(pair, 0) < COOLDOWN_H * 3600:
                continue
            n, win = backtest(df)
            if n < MIN_TRADES or win < MIN_WIN:
                print(f"{pair}: signal but backtest weak ({n} trades, {win:.0%})")
                continue
            r = df.iloc[-1]
            entry = r.c
            sl, tp = entry - SL_ATR * r.atr, entry + TP_ATR * r.atr
            passed = [k for k, v in checks.iloc[-1].items() if v]
            txt = (f"🟢 BUY SIGNAL {pair} ({TF})\n"
                   f"Entry: {entry:.6g}\nSL: {sl:.6g} ({(sl/entry-1)*100:.1f}%)\n"
                   f"TP: {tp:.6g} (+{(tp/entry-1)*100:.1f}%)\n"
                   f"Score: {int(r.score)}/9  RSI: {r.rsi:.0f}\n"
                   f"Backtest (is coin pe): {n} trades, win {win:.0%}\n"
                   f"Confirm: {', '.join(passed)}\n"
                   f"⚠️ Sirf analysis hai, guarantee nahi. SL zaroor lagao.")
            send(txt, chart_png(df, pair, entry, sl, tp))
            state[pair] = time.time()
        except Exception as e:
            print(f"{pair} error: {e}")
        time.sleep(0.2)   # rate limit se bachne ke liye

    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
