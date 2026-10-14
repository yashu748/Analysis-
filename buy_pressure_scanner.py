"""
Buying-pressure scanner: Gate.io ke SAARE USDT spot coins (liquid wale) scan karta hai
aur jin coins me khareedari (buying) zyada hai unka alert SL + targets ke saath bhejta hai.
sol_bot.py aur team_scanner.py ke saath ek hi folder me rakho.

Stage A (sasta, har coin): TF candles (default 15m) se volume spike, money flow (CMF), OBV, trend (EMA20/50)
Stage B (top coins): asli taker buy ratio (recent trades) + order book bid/ask imbalance
Alert tabhi jab score kaafi ho AND taker buying >= MIN_BUY_RATIO.
Late/pump ho chuke coins (extended) aur illiquid coins skip hote hain.

ENV (optional): MIN_QUOTE_VOL (1000000), MAX_COINS (150), TOP_DEEP (25), MIN_SCORE (6),
                MIN_BUY_RATIO (0.58), COOLDOWN_H (6), MAX_ALERTS (6)
Sirf alert hai, advice nahi.
"""
import json
import os
import time

import sol_bot as bot
import team_scanner as ts

MIN_QUOTE_VOL = float(os.getenv("MIN_QUOTE_VOL", "1000000"))   # 24h turnover (USDT)
MAX_COINS = int(os.getenv("MAX_COINS", "150"))
TOP_DEEP = int(os.getenv("TOP_DEEP", "25"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "6"))
MIN_BUY_RATIO = float(os.getenv("MIN_BUY_RATIO", "0.58"))
COOLDOWN_H = float(os.getenv("COOLDOWN_H", "6"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "6"))
STATE_FILE = "alert_state.json"
TF = os.getenv("TF", "1h")   # candle timeframe: 5m / 15m / 1h
STABLES = {"USDC", "FDUSD", "USDE", "DAI", "TUSD", "USD1", "PYUSD", "USDD", "USDT",
           "EUR", "EURI", "AEUR", "BUSD", "USDP", "GUSD", "XUSD", "RLUSD", "USDG"}
SL_ATR, T1_R, T2_R = 1.5, 1.5, 3.0


def f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ stage 0: universe
def universe():
    rows = []
    for t in bot.get("/spot/tickers"):
        p = t.get("currency_pair", "")
        if not p.endswith("_USDT") or any(s in p for s in ts.SKIP):
            continue
        if p.split("_")[0] in STABLES:
            continue
        qv = f(t.get("quote_volume"))
        bid, ask = f(t.get("highest_bid")), f(t.get("lowest_ask"))
        if qv < MIN_QUOTE_VOL or bid <= 0 or ask <= 0:
            continue
        if (ask - bid) / bid > 0.01:          # spread 1% se zyada = illiquid
            continue
        rows.append(dict(pair=p, price=f(t.get("last")), qv=qv,
                         chg=f(t.get("change_percentage"))))
    rows.sort(key=lambda r: r["qv"], reverse=True)
    return rows[:MAX_COINS], len(rows)


# ------------------------------------------------------------ stage A
def stage_a(r):
    raw = bot.get("/spot/candlesticks",
                  {"currency_pair": r["pair"], "interval": TF, "limit": 100})[:-1]
    if len(raw) < 60:
        return None
    qv = [f(x[1]) for x in raw]
    c = [f(x[2]) for x in raw]
    h = [f(x[3]) for x in raw]
    l = [f(x[4]) for x in raw]
    bv = [f(x[6]) for x in raw]
    n = len(c)
    atr = bot.atr(h, l, c, 14)[-1]
    if atr <= 0 or c[-1] <= 0:
        return None
    e20, e50 = bot.ema(c, 20), bot.ema(c, 50)
    prev = sum(qv[-27:-3]) / 24
    spike = (sum(qv[-3:]) / 3) / prev if prev > 0 else 0
    # Chaikin money flow (20)
    num = den = 0.0
    for i in range(n - 20, n):
        rng = h[i] - l[i]
        mfm = ((c[i] - l[i]) - (h[i] - c[i])) / rng if rng > 0 else 0
        num += mfm * bv[i]
        den += bv[i]
    cmf = num / den if den else 0
    # OBV vs its EMA(20)
    obv = [0.0]
    for i in range(1, n):
        obv.append(obv[-1] + bv[i] * ((c[i] > c[i - 1]) - (c[i] < c[i - 1])))
    obv_up = obv[-1] > bot.ema(obv, 20)[-1]
    ext = (c[-1] - e20[-1]) / atr
    score = 0.0
    score += 1 if spike >= 1.5 else 0
    score += 1 if cmf > 0.10 else 0
    score += 1 if obv_up else 0
    score += 1 if c[-1] > e20[-1] else 0
    score += 1 if e20[-1] > e50[-1] else 0
    return dict(r, close=c[-1], atr=atr, spike=spike, cmf=cmf, obv_up=obv_up,
                ext=ext, score_a=score, swing_low=min(l[-10:]))


# ------------------------------------------------------------ stage B
def stage_b(x):
    pair = x["pair"]
    trades = bot.get("/spot/trades", {"currency_pair": pair, "limit": 1000})
    buy = sell = 0.0
    for t in trades:
        v = f(t.get("amount")) * f(t.get("price"))
        if t.get("side") == "buy":
            buy += v
        else:
            sell += v
    ratio = buy / (buy + sell) if (buy + sell) > 0 else 0.5
    span_min = 0
    if len(trades) > 1:
        ts_ms = [f(t.get("create_time_ms")) for t in trades]
        span_min = (max(ts_ms) - min(ts_ms)) / 60000
    book = bot.get("/spot/order_book", {"currency_pair": pair, "limit": 50})
    bids, asks = book.get("bids", []), book.get("asks", [])
    if not bids or not asks:
        return None
    mid = (f(bids[0][0]) + f(asks[0][0])) / 2
    bv = sum(f(p) * f(a) for p, a in bids if f(p) >= mid * 0.98)
    av = sum(f(p) * f(a) for p, a in asks if f(p) <= mid * 1.02)
    imb = bv / (bv + av) if (bv + av) > 0 else 0.5
    score = x["score_a"]
    score += 2 if ratio >= MIN_BUY_RATIO else 1 if ratio >= 0.54 else 0
    score += 1 if imb >= 0.58 else 0
    return dict(x, buy_ratio=ratio, span_min=span_min, imb=imb, score=score,
                ask=f(asks[0][0]))


# ------------------------------------------------------------ alerts
def load_state():
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(state):
    cutoff = time.time() - 48 * 3600
    state = {k: v for k, v in state.items() if v > cutoff}
    with open(STATE_FILE, "w") as fh:
        json.dump(state, fh)


def dec(px):
    return 6 if px < 0.01 else 4 if px < 10 else 2


def fmt(x):
    entry = x["ask"] or x["close"]
    risk = SL_ATR * x["atr"]
    sl = entry - risk
    d = dec(entry)
    return (f"{x['pair']}  score {x['score']:.0f}/8  (24h {x['chg']:+.1f}%)\n"
            f"Entry ~{entry:.{d}f} | SL {sl:.{d}f} ({risk / entry * 100:.1f}% niche)\n"
            f"T1 {entry + T1_R * risk:.{d}f} | T2 {entry + T2_R * risk:.{d}f}\n"
            f"Taker buying {x['buy_ratio'] * 100:.0f}% (last ~{x['span_min']:.0f} min) | "
            f"Book bids {x['imb'] * 100:.0f}%\n"
            f"Volume spike {x['spike']:.1f}x | CMF {x['cmf']:+.2f} | 24h vol ${x['qv'] / 1e6:.1f}M")


def main():
    uni, passed = universe()
    print(f"Universe: {passed} coins liquidity filter paas, {len(uni)} scan honge")
    stage = []
    for r in uni:
        try:
            x = stage_a(r)
        except Exception as e:
            print(f"{r['pair']}: skip A ({e})")
            continue
        time.sleep(0.12)
        if not x:
            continue
        # pump ho chuke / bahut extended coins me late entry nahi
        if x["ext"] > 3.0 or x["chg"] > 40:
            continue
        stage.append(x)
    stage.sort(key=lambda x: (x["score_a"], x["spike"]), reverse=True)
    deep = []
    for x in stage[:TOP_DEEP]:
        if x["score_a"] < 3:
            break
        try:
            y = stage_b(x)
        except Exception as e:
            print(f"{x['pair']}: skip B ({e})")
            continue
        time.sleep(0.15)
        if y:
            deep.append(y)
            print(f"{y['pair']}: A={y['score_a']:.0f} buy={y['buy_ratio']:.2f} "
                  f"book={y['imb']:.2f} spike={y['spike']:.1f} -> {y['score']:.0f}")

    cands = [y for y in deep if y["score"] >= MIN_SCORE and y["buy_ratio"] >= MIN_BUY_RATIO]
    cands.sort(key=lambda y: (y["score"], y["buy_ratio"]), reverse=True)
    state = load_state()
    now = time.time()
    fresh = [y for y in cands if now - state.get(y["pair"], 0) > COOLDOWN_H * 3600]
    if not fresh:
        print("Koi naya high-buying setup nahi.")
        save_state(state)
        return
    chosen = fresh[:MAX_ALERTS]
    msg = (f"BUYING PRESSURE alerts ({len(chosen)} coin, Gate.io)\n\n" +
           "\n\n".join(fmt(y) for y in chosen) +
           "\n\nSL = 1.5xATR(1h), T1 = 1.5R, T2 = 3R. Khareedari zyada hone ka matlab "
           "price badhna pakka nahi. Chhote coins me pump-dump hota hai: SL zaroor lagao, "
           "total risk 1-2%.")
    print(msg)
    bot.send_telegram(msg)
    for y in chosen:
        state[y["pair"]] = now
    save_state(state)


if __name__ == "__main__":
    main()
  
