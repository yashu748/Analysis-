#!/usr/bin/env python3
"""
backtest3.py - alag-alag STRATEGY families ka research backtest (sirf LONG, spot, fees ke saath).

Pichhla result bata chuka hai ki 15m/1h indicator-signals me edge nahi hai. Yahan ye naye tareeke test hote hain:
  donch   : Donchian breakout (20/55 candle high toot-ta hai) + trailing stop       [4h, 1d]
  pull    : uptrend me RSI dip khareedna, RSI 55 par nikalna                          [1h, 4h]
  squeeze : Bollinger squeeze ke baad volume ke saath breakout + trailing stop        [1h, 4h]
  mom     : daily top-momentum coins (20 din ka return) 5 din hold                    [1d]

Har strategy: fees minus, train (pehle 2/3) aur validation (aakhri 1/3) alag, aur random-entry baseline (same exit).
Fee default 0.4% round trip (Gate spot 0.2% har taraf). Badalna ho: env FEE=0.003

Env (optional): BT_PAIRS=40, BT_BUDGET_MIN=50, FEE=0.004
"""
import os
import time

import numpy as np
import pandas as pd
import requests

import chart_bot as b
import backtest2 as b2

PAIRS_N = int(os.getenv("BT_PAIRS", "40"))
BUDGET = float(os.getenv("BT_BUDGET_MIN", "50")) * 60
FEE = float(os.getenv("FEE", "0.004"))
b2.LOAD_DAYS.update({"1h": 120, "4h": 300, "1d": 700})
TFS = ("1h", "4h", "1d")
WARM = 210


# ----------------------------------------------------------------- features
def feat(df):
    c = df.c
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    width = 4 * sd / mid
    F = dict(o=df.o.values, h=df.h.values, l=df.l.values, c=c.values, t=df.t.values,
             atr=b.atr(df).values, rsi=b.rsi(c).values,
             ema50=b.ema(c, 50).values, ema100=b.ema(c, 100).values, ema200=b.ema(c, 200).values,
             bbup=(mid + 2 * sd).values,
             bbrank=width.rolling(100).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True).values,
             dh20=df.h.rolling(20).max().shift(1).values, dh55=df.h.rolling(55).max().shift(1).values,
             vr=(df.v / df.v.rolling(20).mean().shift(1)).values)
    return F


def clean(m):
    m = np.nan_to_num(m.astype(float), nan=0.0) > 0
    m[:WARM] = False
    return m


# ----------------------------------------------------------------- simulation (long only)
def sim(F, i, sl_atr, exit_kind, param, maxhold):
    n = len(F["c"])
    if i + 2 >= n:
        return None
    entry, a = F["c"][i], F["atr"][i]
    if not a > 0:
        return None
    stop = entry - sl_atr * a
    risk = entry - stop
    if risk / entry < 0.004 or risk / entry > 0.20:
        return None
    best = entry
    last = min(i + maxhold, n - 1)
    ex = None
    for j in range(i + 1, last + 1):
        if F["l"][j] <= stop:
            ex = min(stop, F["o"][j])
            break
        cj = F["c"][j]
        best = max(best, cj)
        if exit_kind == "trail":
            stop = max(stop, best - param * F["atr"][j])
        elif exit_kind == "rsi" and F["rsi"][j] > param:
            ex = cj
            break
        elif exit_kind == "time" and j - i >= param:
            ex = cj
            break
    if ex is None:
        ex = F["c"][last]
    return (ex - entry) / risk - FEE * entry / risk


def pick(mask, cd):
    out, last = [], -10 ** 9
    for i in np.where(mask)[0]:
        if i - last >= cd:
            out.append(i)
            last = i
    return out


def st(vals):
    if not vals:
        return 0, 0.0, 0.0, 0.0
    v = np.array(vals)
    w = (v > 0).sum()
    return len(v), float(v.mean()), float(v.sum()), float(w / len(v) * 100)


# ----------------------------------------------------------------- main
def main():
    t0 = time.time()
    pairs = [p for p in b.coin_list() if p != "BTC_USDT"][:PAIRS_N]
    print(f"Backtest3: {len(pairs)} coins, fee {FEE * 100:.2f}% round trip", flush=True)
    DATA = {}
    for p in pairs:
        if time.time() - t0 > BUDGET * 0.4:
            break
        d = {tf: b2.load(p, tf) for tf in TFS}
        if all(v is not None and len(v) > WARM + 60 for v in d.values()):
            DATA[p] = d
    print(f"Data mila: {len(DATA)} coins ({time.time() - t0:.0f}s)", flush=True)
    PRE = {(p, tf): feat(DATA[p][tf]) for p in DATA for tf in TFS}

    # daily momentum signals (cross-sectional)
    closes = pd.DataFrame({p: DATA[p]["1d"].set_index("t").c for p in DATA})
    ret = closes.pct_change(20)
    rank = ret.rank(axis=1, ascending=False)
    mom_rank = {p: rank[p].reindex(DATA[p]["1d"].t.values).values for p in DATA}

    configs = []   # (name, tf, mask_fn, cd, sl, exit_kind, param, maxhold)
    for N in (20, 55):
        for tf, mh in (("4h", 60), ("1d", 40)):
            configs.append((f"donch{N}", tf, (lambda F, p, N=N: (F["c"] > F[f"dh{N}"]) & (F["c"] > F["ema100"])),
                            5, 3.0, "trail", 3.0, mh))
    for th in (30, 35):
        for tf, mh in (("1h", 48), ("4h", 30)):
            configs.append((f"pull_rsi{th}", tf,
                            (lambda F, p, th=th: (F["ema50"] > F["ema200"]) & (F["c"] > F["ema200"]) & (F["rsi"] < th)),
                            3, 2.5, "rsi", 55, mh))
    for tf, mh in (("1h", 48), ("4h", 30)):
        configs.append(("squeeze", tf,
                        (lambda F, p: (F["bbrank"] <= 0.2) & (F["c"] > F["bbup"]) & (F["vr"] > 1.5) & (F["c"] > F["ema50"])),
                        3, 2.5, "trail", 2.5, mh))
    for k in (3, 5):
        configs.append((f"mom_top{k}", "1d",
                        (lambda F, p, k=k: (mom_rank[p] <= k) & (F["c"] > F["ema50"])), 5, 3.0, "time", 5, 6))

    rng = np.random.default_rng(11)
    results = []
    for name, tf, fn, cd, sl, ek, par, mh in configs:
        if time.time() - t0 > BUDGET:
            print("time budget poora", flush=True)
            break
        tr_v, va_v, all_v = [], [], []
        for p in DATA:
            F = PRE[(p, tf)]
            n = len(F["c"])
            split_i = WARM + int((n - WARM) * 2 / 3)
            mask = clean(fn(F, p))
            for i in pick(mask, cd):
                rv = sim(F, i, sl, ek, par, mh)
                if rv is None:
                    continue
                all_v.append(rv)
                (tr_v if i < split_i else va_v).append(rv)
        # random baseline, same exit
        base = []
        plist = list(DATA)
        for _ in range(500):
            p = plist[rng.integers(len(plist))]
            F = PRE[(p, tf)]
            i = int(rng.integers(WARM, len(F["c"]) - 2))
            rv = sim(F, i, sl, ek, par, mh)
            if rv is not None:
                base.append(rv)
        n, avg, net, win = st(all_v)
        results.append(dict(name=name, tf=tf, n=n, avg=avg, net=net, win=win,
                            tn=len(tr_v), tavg=st(tr_v)[1], vn=len(va_v), vavg=st(va_v)[1],
                            bavg=st(base)[1], bwin=st(base)[3]))
        print(f"  done {name} {tf}: {n} trades", flush=True)

    results.sort(key=lambda r: r["avg"], reverse=True)
    lines = [f"BACKTEST3 (sirf long): {len(DATA)} coins, fees {FEE * 100:.1f}% round trip.",
             "Format: strategy|tf  n win% avg R  (TRAIN avg,n | VALID avg,n)  [random baseline avg]", ""]
    for r in results:
        lines.append(f"{r['name']}|{r['tf']}  n={r['n']} win{r['win']:.0f}% {r['avg']:+.2f}R "
                     f"(T {r['tavg']:+.2f},{r['tn']} | V {r['vavg']:+.2f},{r['vn']}) [rand {r['bavg']:+.2f}]")
    good = [r for r in results if r["tn"] >= 20 and r["vn"] >= 10 and r["tavg"] > 0.05 and r["vavg"] > 0.05
            and r["avg"] > r["bavg"] + 0.05]
    lines += ["", f"DONO (train+valid) me positive aur random se behtar: {len(good)}"]
    for r in good:
        lines.append(f"  -> {r['name']}|{r['tf']}  {r['avg']:+.2f}R/trade, n={r['n']}")
    lines += ["", "Samajh: sirf wahi bharose ke layak hai jo train AUR valid dono me positive ho aur random se saaf behtar ho. "
                  "Trades kam ho to (n<50) natija ittefaq bhi ho sakta hai."]
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    if b.TG_TOKEN and b.TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{b.TG_TOKEN}/sendMessage",
                          data={"chat_id": b.TG_CHAT, "text": text[:4000]}, timeout=20)
        except Exception as e:
            print("telegram failed:", e)


if __name__ == "__main__":
    main()
