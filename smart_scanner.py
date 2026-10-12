"""
Smart scanner: backtest-filtered alerts (Gate.io spot, 1h candles).
sol_bot.py aur team_scanner.py ke saath ek hi folder me rakho.

Kaise kaam karta hai:
 1) Har coin ke ~1000 candle (~41 din) par 700+ analysts ka "net score" (-1..+1) har bar ke liye nikalta hai
 2) Isi history par naye signals ka backtest (SL 1.5xATR, T1 1.5R, fees ke saath)
 3) Global gate: poore market ka backtest + out-of-sample half positive na ho to KOI alert nahi
 4) Coin gate: us coin ka khud ka backtest bhi positive ho
 5) Abhi ka signal sirf tab bhejta hai jab: taaza cross ho, price EMA20 se zyada door (extended) na ho,
    volume theek ho, aur BTC regime ulta na ho
Alert me backtest ke asli numbers dikhte hain. Accuracy guarantee nahi, sirf filter hai.

ENV (optional): TH (0.45), HOLD (24 bars), FEE (0.002), TOP_N (40),
                MIN_COIN_TRADES (8), MIN_WR (0.40), MIN_EXP (0.05)
"""
import os
import time

import sol_bot as bot
import team_scanner as ts

TH = float(os.getenv("TH", "0.45"))
HOLD = int(os.getenv("HOLD", "24"))
FEE = float(os.getenv("FEE", "0.002"))          # round-trip fee (0.2%)
TOP_N = int(os.getenv("TOP_N", "40"))
MIN_COIN_TRADES = int(os.getenv("MIN_COIN_TRADES", "8"))
MIN_WR = float(os.getenv("MIN_WR", "0.40"))
MIN_EXP = float(os.getenv("MIN_EXP", "0.05"))
W = 250            # warmup bars
MAX_ALERTS = 5
SL_ATR, T1_R, T2_R = 1.5, 1.5, 3.0


def sgn(x):
    return (x > 0) - (x < 0)


# ----------------------------------------------------------------- data
def fetch(pair, interval="1h", limit=1000):
    raw = bot.get("/spot/candlesticks",
                  {"currency_pair": pair, "interval": interval, "limit": limit})
    raw = raw[:-1]  # abhi ban rahi candle hata do
    o = [float(x[5]) for x in raw]
    h = [float(x[3]) for x in raw]
    l = [float(x[4]) for x in raw]
    c = [float(x[2]) for x in raw]
    v = [float(x[6]) for x in raw]
    return o, h, l, c, v


# ----------------------------------------------------------------- analysts
def rsi_series(c, p):
    n = len(c)
    out = [50.0] * n
    if n <= p:
        return out
    g = sum(max(c[i] - c[i - 1], 0) for i in range(1, p + 1)) / p
    s = sum(max(c[i - 1] - c[i], 0) for i in range(1, p + 1)) / p
    out[p] = 100.0 if s == 0 else 100 - 100 / (1 + g / s)
    for i in range(p + 1, n):
        d = c[i] - c[i - 1]
        g = (g * (p - 1) + max(d, 0)) / p
        s = (s * (p - 1) + max(-d, 0)) / p
        out[i] = 100.0 if s == 0 else 100 - 100 / (1 + g / s)
    return out


def st_trend_series(h, l, c, period, mult):
    a = bot.atr(h, l, c, period)
    n = len(c)
    fub, flb, trend = [0.0] * n, [0.0] * n, [1] * n
    for i in range(n):
        hl2 = (h[i] + l[i]) / 2
        ub, lb = hl2 + mult * a[i], hl2 - mult * a[i]
        if i == 0:
            fub[i], flb[i] = ub, lb
            continue
        fub[i] = ub if (ub < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb if (lb > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < flb[i] else 1
        else:
            trend[i] = 1 if c[i] > fub[i] else -1
    return trend


def prefix(vals):
    out = [0.0]
    for x in vals:
        out.append(out[-1] + x)
    return out


def team_net(o, h, l, c, v):
    """Har bar ke liye -1..+1 net score. (net list, analysts ki sankhya)"""
    n = len(c)
    net = [0] * n
    cnt = 0
    cache = {}

    def E(p):
        if p not in cache:
            cache[p] = bot.ema(c, p)
        return cache[p]

    def acc(votes):
        nonlocal cnt
        for i in range(W, n):
            net[i] += votes[i]
        cnt += 1

    # 1) EMA crossovers (377)
    for f in range(3, 21):
        for s in range(20, 125, 5):
            if f < s:
                acc([sgn(a - b) for a, b in zip(E(f), E(s))])
    # 2) price vs EMA (39)
    for p in range(10, 205, 5):
        acc([sgn(x - e) for x, e in zip(c, E(p))])
    # 3) RSI (26)
    for p in range(5, 31):
        acc([1 if r > 55 else -1 if r < 45 else 0 for r in rsi_series(c, p)])
    # 4) SuperTrend grid (96)
    for per in range(5, 21):
        for m in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            acc(st_trend_series(h, l, c, per, m))
    # 5) MACD grid (32)
    for f in (5, 8, 12, 15):
        for s in (17, 21, 26, 34):
            for sg in (5, 9):
                macd = [a - b for a, b in zip(E(f), E(s))]
                sig = bot.ema(macd, sg)
                acc([sgn(a - b) for a, b in zip(macd, sig)])
    # 6) Donchian breakout (12)
    for p in range(5, 65, 5):
        votes = [0] * n
        for i in range(p, n):
            if c[i] > max(h[i - p:i]):
                votes[i] = 1
            elif c[i] < min(l[i - p:i]):
                votes[i] = -1
        acc(votes)
    # 7) momentum / ROC (38)
    for p in range(3, 41):
        acc([0] * p + [sgn(c[i] - c[i - p]) for i in range(p, n)])
    # 8) OBV vs its EMA (12) - volume
    obv = [0.0]
    for i in range(1, n):
        obv.append(obv[-1] + v[i] * sgn(c[i] - c[i - 1]))
    for p in range(5, 65, 5):
        acc([sgn(a - b) for a, b in zip(obv, bot.ema(obv, p))])
    # 9) Chaikin money flow (36) - volume
    mfv = [(((c[i] - l[i]) - (h[i] - c[i])) / (h[i] - l[i]) * v[i]) if h[i] > l[i] else 0.0
           for i in range(n)]
    pm, pv = prefix(mfv), prefix(v)
    for p in range(5, 41):
        votes = [0] * n
        for i in range(p, n):
            vol = pv[i + 1] - pv[i + 1 - p]
            cmf = (pm[i + 1] - pm[i + 1 - p]) / vol if vol else 0
            votes[i] = 1 if cmf > 0.05 else -1 if cmf < -0.05 else 0
        acc(votes)
    # 10) rolling VWAP (20) - volume
    ptv = prefix([(h[i] + l[i] + c[i]) / 3 * v[i] for i in range(n)])
    for p in range(10, 210, 10):
        votes = [0] * n
        for i in range(p, n):
            vol = pv[i + 1] - pv[i + 1 - p]
            if vol:
                votes[i] = sgn(c[i] - (ptv[i + 1] - ptv[i + 1 - p]) / vol)
        acc(votes)
    # 11) Aroon (11)
    for p in range(10, 65, 5):
        votes = [0] * n
        for i in range(p, n):
            wh, wl = h[i - p:i + 1], l[i - p:i + 1]
            up = wh.index(max(wh))
            dn = wl.index(min(wl))
            votes[i] = sgn(up - dn)
        acc(votes)
    # 12) +DI vs -DI (22)
    pdm, mdm = [0.0], [0.0]
    for i in range(1, n):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
    for p in range(7, 29):
        sp, sm = sum(pdm[1:p + 1]), sum(mdm[1:p + 1])
        votes = [0] * n
        votes[p] = sgn(sp - sm)
        for i in range(p + 1, n):
            sp = sp - sp / p + pdm[i]
            sm = sm - sm / p + mdm[i]
            votes[i] = sgn(sp - sm)
        acc(votes)
    return [x / cnt for x in net], cnt


# ----------------------------------------------------------------- backtest
def backtest(o, h, l, c, v, net, atr, ema20):
    """Fresh-cross signals ka backtest. Return trades: (bar, R_net, hit_target)."""
    n = len(c)
    trades = []
    i = W + 1
    while i < n - HOLD - 1:
        d = 0
        if net[i] >= TH and net[i - 1] < TH:
            d = 1
        elif net[i] <= -TH and net[i - 1] > -TH:
            d = -1
        if d == 0 or atr[i] <= 0:
            i += 1
            continue
        avg_v = sum(v[i - 20:i]) / 20
        if abs(c[i] - ema20[i]) > 2.5 * atr[i] or v[i] < 0.8 * avg_v:
            i += 1
            continue
        entry = o[i + 1]
        risk = SL_ATR * atr[i]
        sl, t1 = entry - d * risk, entry + d * T1_R * risk
        hit, r, exit_j = False, None, i + HOLD
        for j in range(i + 1, i + HOLD + 1):
            hit_sl = l[j] <= sl if d > 0 else h[j] >= sl
            hit_t = h[j] >= t1 if d > 0 else l[j] <= t1
            if hit_sl:                 # dono ek hi candle me ho to SL maano (conservative)
                r, exit_j = -1.0, j
                break
            if hit_t:
                r, hit, exit_j = T1_R, True, j
                break
        if r is None:
            r = d * (c[i + HOLD] - entry) / risk
        r -= FEE * entry / risk
        trades.append((i, r, hit))
        i = exit_j + 1
    return trades


def stats(trades):
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=0.0, exp=0.0)
    return dict(n=n, wr=sum(1 for t in trades if t[2]) / n,
                exp=sum(t[1] for t in trades) / n)


# ----------------------------------------------------------------- main
def analyse_pair(pair):
    o, h, l, c, v = fetch(pair)
    if len(c) < W + HOLD + 100:
        raise ValueError("history kam hai")
    net, cnt = team_net(o, h, l, c, v)
    atr = bot.atr(h, l, c, 14)
    ema20 = bot.ema(c, 20)
    trades = backtest(o, h, l, c, v, net, atr, ema20)
    split = W + int((len(c) - W) * 0.7)
    n = len(c) - 1
    # taaza signal (last closed bar ya usse pichli bar par cross)
    cur = 0
    for k in (n, n - 1):
        if net[k] >= TH and net[k - 1] < TH and net[n] >= TH:
            cur = 1
        elif net[k] <= -TH and net[k - 1] > -TH and net[n] <= -TH:
            cur = -1
        if cur:
            break
    avg_v = sum(v[n - 20:n]) / 20
    ok_filters = (abs(c[n] - ema20[n]) <= 2.5 * atr[n]) and v[n] >= 0.8 * avg_v
    return dict(pair=pair, cnt=cnt, net=net[n], price=c[n], atr=atr[n],
                trades=trades, train=[t for t in trades if t[0] < split],
                test=[t for t in trades if t[0] >= split],
                cur=cur, ok_filters=ok_filters)


def fmt_pct(x):
    return f"{x * 100:.0f}%"


def main():
    pairs = ts.top_pairs(TOP_N)
    if "BTC_USDT" not in pairs:
        pairs = ["BTC_USDT"] + pairs
    res = {}
    for p in pairs:
        try:
            res[p] = analyse_pair(p)
            r = res[p]
            s = stats(r["trades"])
            print(f"{p}: net {r['net']:+.2f} cur {r['cur']:+d} | backtest n={s['n']} "
                  f"wr={fmt_pct(s['wr'])} exp={s['exp']:+.2f}R")
        except Exception as e:
            print(f"{p}: skip ({e})")
        time.sleep(0.3)

    all_tr = [t for r in res.values() for t in r["trades"]]
    all_test = [t for r in res.values() for t in r["test"]]
    g, gt = stats(all_tr), stats(all_test)
    print(f"GLOBAL: n={g['n']} wr={fmt_pct(g['wr'])} exp={g['exp']:+.2f}R | "
          f"out-of-sample n={gt['n']} wr={fmt_pct(gt['wr'])} exp={gt['exp']:+.2f}R")

    if g["n"] < 30 or g["exp"] <= 0 or gt["exp"] <= 0:
        print("Global gate band: is period me strategy ka edge nahi dikha, koi alert nahi.")
        return

    btc = res.get("BTC_USDT")
    alerts = []
    for p, r in res.items():
        if not r["cur"] or not r["ok_filters"]:
            continue
        s = stats(r["trades"])
        st = stats(r["test"])
        if (s["n"] < MIN_COIN_TRADES or s["exp"] < MIN_EXP or s["wr"] < MIN_WR
                or (st["n"] >= 3 and st["exp"] <= 0)):
            continue
        if btc and p != "BTC_USDT" and sgn(btc["net"]) == -r["cur"] and abs(btc["net"]) >= 0.3:
            continue   # BTC regime ulta hai
        alerts.append((s["exp"], p, r, s, st))
    alerts.sort(key=lambda x: x[0], reverse=True)

    if not alerts:
        print("Filters ke baad koi setup nahi.")
        return
    blocks = []
    for _, p, r, s, st in alerts[:MAX_ALERTS]:
        d = r["cur"]
        px, risk = r["price"], SL_ATR * r["atr"]
        dec = 4 if px < 10 else 2
        blocks.append(
            f"{p} {'LONG' if d > 0 else 'SHORT'} (team score {r['net']:+.2f}, taaza signal)\n"
            f"Entry ~{px:.{dec}f} | SL {px - d * risk:.{dec}f}\n"
            f"T1 {px + d * T1_R * risk:.{dec}f} | T2 {px + d * T2_R * risk:.{dec}f}\n"
            f"Backtest ({r['cnt']} analysts, ~41 din): {s['n']} trades, "
            f"T1-hit {fmt_pct(s['wr'])}, avg {s['exp']:+.2f}R"
            + (f" | out-of-sample {st['n']} trades {st['exp']:+.2f}R" if st["n"] else ""))
    msg = ("HIGH-QUALITY setups (backtest-filtered)\n\n" + "\n\n".join(blocks) +
           f"\n\nPure market backtest: {g['n']} trades, T1-hit {fmt_pct(g['wr'])}, "
           f"avg {g['exp']:+.2f}R (fees ke baad).\n"
           "Ye guarantee nahi hai, purana record hai. SL ke bina trade mat lo, "
           "total risk 1-2% se zyada nahi.")
    print(msg)
    bot.send_telegram(msg)


if __name__ == "__main__":
    main()
