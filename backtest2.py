#!/usr/bin/env python3
"""
backtest2.py - lamba research backtest (15m aur 1h, alag-alag tareeke), fees ke saath.

Kya karta hai:
  * Top coins ka asli Gate.io data: 15m ke 30 din, 1h ke 90 din (HTF trend ke liye 4h, 1d bhi).
  * Bot jaise signal (5 me se 4 indicators) ko alag-alag settings par chalata hai:
      entry timeframe (15m / 1h), higher-timeframe trend filter (none / loose / strict),
      volume (1.5x / 2x), SL (1.5 ATR / 2.5 ATR), exit (bot jaisa 1.5R+3R / sirf 2R / sirf 1R)
  * Har combo ka result fees (0.2%) minus karke dikhata hai.
  * Data ko time ke hisaab se 2 hisson me baantta hai: TRAIN (pehle 2/3) aur VALIDATION (aakhri 1/3).
    Sirf wahi combo bharose ke layak hai jo DONO me positive ho.
  * Random entries ka baseline bhi deta hai (agar strategy random se behtar nahi, to koi edge nahi).

Env (optional): BT15_DAYS=30, BT1H_DAYS=90, BT_PAIRS=40, BT_BUDGET_MIN=50
"""
import os
import time
import itertools

import numpy as np
import pandas as pd
import requests

import chart_bot as b

D15 = int(os.getenv("BT15_DAYS", "30"))
D1H = int(os.getenv("BT1H_DAYS", "90"))
PAIRS_N = int(os.getenv("BT_PAIRS", "40"))
BUDGET = float(os.getenv("BT_BUDGET_MIN", "50")) * 60
FEE = 0.002
SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
HTF = {"15m": ("1h", "4h"), "1h": ("4h", "1d")}
ENTRY_DAYS = {"15m": D15, "1h": D1H}
LOAD_DAYS = {"15m": D15 + 2, "1h": max(D1H, D15) + 4, "4h": 160, "1d": 420}
HORIZON = {"15m": 96, "1h": 48}          # trade kitni candles tak dekhna (24h / 48h)
COOLDOWN = {"15m": 24, "1h": 6}          # same side repeat gap (6 ghante)
WARMUP = 100

GRID = dict(
    tf=["15m", "1h"],
    trend=["none", "loose", "strict"],
    vol=[1.5, 2.0],
    sl=[1.5, 2.5],
    exit=["bot", "tp2", "tp1"],
)


# ----------------------------------------------------------------- data
def load(pair, tf):
    sec = SEC[tf]
    end = int(time.time())
    start = end - LOAD_DAYS[tf] * 86400
    step = 900 * sec
    rows, t = [], start
    while t < end:
        r = b.get(f"{b.BASE}/spot/candlesticks",
                  {"currency_pair": pair, "interval": tf, "from": t, "to": min(t + step, end)})
        if r:
            rows += r
        t += step
    if len(rows) < 120:
        return None
    df = pd.DataFrame([[float(x[0]), float(x[5]), float(x[4]), float(x[3]), float(x[2]), float(x[1])]
                       for x in rows], columns=["t", "o", "l", "h", "c", "v"])
    df = df[["t", "o", "h", "l", "c", "v"]].drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df = df[df.t + sec <= time.time()].reset_index(drop=True)      # sirf closed candles
    return df if len(df) >= 120 else None


# ----------------------------------------------------------------- features / signals
def feats(df):
    c = df.c
    e9, e26 = b.ema(c, 9), b.ema(c, 26)
    ml = b.ema(c, 12) - b.ema(c, 26)
    hist = ml - b.ema(ml, 9)
    vavg = df.v.rolling(20).mean().shift(1)
    return dict(
        c=c.values, h=df.h.values, l=df.l.values, t=df.t.values,
        e9=e9.values, e26=e26.values, st=b.supertrend(df), rsi=b.rsi(c).values,
        hist=hist.values, hist_prev=hist.shift(1).values, vr=(df.v / vavg).values,
        atr=b.atr(df).values,
    )


def base_signals(F, vol_mult):
    """Return (long_mask, short_mask) bot ke 4/5 rule se."""
    c, r = F["c"], F["rsi"]
    L = np.vstack([(F["e9"] > F["e26"]) & (c > F["e26"]), F["st"] == 1, (r >= 50) & (r <= 70),
                   (F["hist"] > 0) & (F["hist"] > F["hist_prev"]), F["vr"] > vol_mult]).astype(int)
    S = np.vstack([(F["e9"] < F["e26"]) & (c < F["e26"]), F["st"] == -1, (r >= 30) & (r <= 50),
                   (F["hist"] < 0) & (F["hist"] < F["hist_prev"]), F["vr"] > vol_mult]).astype(int)
    sl_, ss_ = L.sum(0), S.sum(0)
    ok = ~np.isnan(r) & ~np.isnan(F["atr"]) & ~np.isnan(F["vr"])
    longm = ok & (sl_ >= 4) & (sl_ > ss_) & (r <= 66)
    shortm = ok & (ss_ >= 4) & (ss_ > sl_) & (r >= 34)
    longm[:WARMUP] = False
    shortm[:WARMUP] = False
    return longm, shortm


def htf_trend_series(df):
    c = df.c
    e21, e50 = b.ema(c, 21), b.ema(c, 50)
    tr = np.where((c > e21) & (e21 > e50), 1, np.where((c < e21) & (e21 < e50), -1, 0))
    tr[:55] = 0
    return tr, (df.t.values + 0.0)


def trend_at(htf_tr, htf_t, htf_sec, close_times):
    idx = np.searchsorted(htf_t + htf_sec, close_times, side="right") - 1
    out = np.zeros(len(close_times), dtype=int)
    ok = idx >= 0
    out[ok] = htf_tr[idx[ok]]
    return out


def pick(mask_long, mask_short, trends, mode, cd):
    """trend filter + cooldown lagao -> list of (i, side) side=+1 long, -1 short."""
    out = []
    for side, mask in ((1, mask_long), (-1, mask_short)):
        last = -10 ** 9
        for i in np.where(mask)[0]:
            t1, t2 = trends[0][i], trends[1][i]
            if mode == "loose" and (t1 == -side or t2 == -side or (t1 != side and t2 != side)):
                continue
            if mode == "strict" and not (t1 == side and t2 == side):
                continue
            if i - last < cd:
                continue
            last = i
            out.append((i, side))
    return out


# ----------------------------------------------------------------- exit simulation
def first(cond):
    return int(np.argmax(cond)) if cond.any() else -1


def sim(F, i, side, sl_atr, exit_model, horizon):
    """Return R (fees ke baad) ya None agar signal skip/unresolved."""
    n = len(F["c"])
    if i + 2 >= n:
        return None
    entry, a = F["c"][i], F["atr"][i]
    lo10, hi10 = F["l"][max(0, i - 9):i + 1].min(), F["h"][max(0, i - 9):i + 1].max()
    if side == 1:
        sl = min(entry - sl_atr * a, lo10 - 0.25 * a)
    else:
        sl = max(entry + sl_atr * a, hi10 + 0.25 * a)
    risk = abs(entry - sl)
    if risk <= 0 or risk / entry > 0.06 or risk / entry < 0.002:
        return None
    # short ko long me badal do (price ko negative karke)
    hi = F["h"][i + 1:i + 1 + horizon] * side if side == 1 else -F["l"][i + 1:i + 1 + horizon]
    lo = F["l"][i + 1:i + 1 + horizon] * side if side == 1 else -F["h"][i + 1:i + 1 + horizon]
    if len(hi) < 6:
        return None
    e, s_ = entry * side, sl * side
    r_ = risk
    i_sl = first(lo <= s_)
    if exit_model in ("tp2", "tp1"):
        mult = 2.0 if exit_model == "tp2" else 1.0
        i_tp = first(hi >= e + mult * r_)
        if i_sl >= 0 and (i_tp < 0 or i_sl <= i_tp):
            rv = -1.0
        elif i_tp >= 0:
            rv = mult
        else:
            rv = 0.0
    else:   # bot: T1 1.5R me 50% book, SL entry par, T2 3R
        t1, t2 = e + 1.5 * r_, e + 3.0 * r_
        i_t1 = first(hi >= t1)
        if i_sl >= 0 and (i_t1 < 0 or i_sl <= i_t1):
            rv = -1.0
        elif i_t1 >= 0:
            i_t2 = i_t1 + first(hi[i_t1:] >= t2) if (hi[i_t1:] >= t2).any() else -1
            i_be = (i_t1 + 1 + first(lo[i_t1 + 1:] <= e)) if len(lo) > i_t1 + 1 and (lo[i_t1 + 1:] <= e).any() else -1
            if i_t2 == i_t1:
                rv = 2.25
            elif i_be >= 0 and (i_t2 < 0 or i_be <= i_t2):
                rv = 0.75
            elif i_t2 >= 0:
                rv = 2.25
            else:
                rv = 0.75
        else:
            rv = 0.0
    return rv - FEE * entry / risk


# ----------------------------------------------------------------- main
def stats(vals):
    if not vals:
        return 0, 0.0, 0.0, 0.0
    v = np.array(vals)
    w, l = (v > 0).sum(), (v <= 0).sum()
    return len(v), float(v.mean()), float(v.sum()), float(w / (w + l) * 100)


def main():
    t0 = time.time()
    pairs = [p for p in b.coin_list() if p != "BTC_USDT"][:PAIRS_N]
    print(f"Backtest2: {len(pairs)} coins, 15m {D15} din, 1h {D1H} din", flush=True)
    DATA = {}
    for p in pairs:
        if time.time() - t0 > BUDGET * 0.4:
            break
        d = {tf: load(p, tf) for tf in SEC}
        if all(v is not None for v in d.values()):
            DATA[p] = d
    print(f"Data mila: {len(DATA)} coins ({time.time() - t0:.0f}s)", flush=True)

    # har coin/tf ke features + HTF trends
    PRE = {}
    for p, d in DATA.items():
        for tf in ("15m", "1h"):
            F = feats(d[tf])
            ct = F["t"] + SEC[tf]
            trs = []
            for h in HTF[tf]:
                tr, ht = htf_trend_series(d[h])
                trs.append(trend_at(tr, ht, SEC[h], ct))
            PRE[(p, tf)] = (F, trs)

    # entry window / train-valid split time (har tf ke liye)
    now = time.time()
    split_t, start_t = {}, {}
    for tf in ("15m", "1h"):
        st_ = now - ENTRY_DAYS[tf] * 86400
        start_t[tf] = st_
        split_t[tf] = st_ + (now - st_) * 2 / 3

    results, rng = [], np.random.default_rng(7)
    keys = list(itertools.product(GRID["tf"], GRID["trend"], GRID["vol"]))
    for tf, trend, vol in keys:
        if time.time() - t0 > BUDGET:
            print("time budget poora", flush=True)
            break
        sigs = []   # (pair, i, side)
        for p in DATA:
            F, trs = PRE[(p, tf)]
            lm, sm = base_signals(F, vol)
            lm = lm & (F["t"] >= start_t[tf])
            sm = sm & (F["t"] >= start_t[tf])
            for i, side in pick(lm, sm, trs, trend, COOLDOWN[tf]):
                sigs.append((p, i, side))
        for sl_atr, ex in itertools.product(GRID["sl"], GRID["exit"]):
            tr_v, va_v, all_v = [], [], []
            for p, i, side in sigs:
                F = PRE[(p, tf)][0]
                rv = sim(F, i, side, sl_atr, ex, HORIZON[tf])
                if rv is None:
                    continue
                all_v.append(rv)
                (tr_v if F["t"][i] < split_t[tf] else va_v).append(rv)
            n, avg, net, win = stats(all_v)
            results.append(dict(tf=tf, trend=trend, vol=vol, sl=sl_atr, ex=ex, n=n, avg=avg, net=net, win=win,
                                tn=len(tr_v), tavg=stats(tr_v)[1], vn=len(va_v), vavg=stats(va_v)[1]))
        print(f"  done {tf} trend={trend} vol={vol}: {len(sigs)} raw signals", flush=True)

    # random baseline
    base_lines = []
    for tf in ("15m", "1h"):
        for sl_atr, ex in itertools.product(GRID["sl"], GRID["exit"]):
            vals = []
            plist = list(DATA)
            for _ in range(600):
                p = plist[rng.integers(len(plist))]
                F = PRE[(p, tf)][0]
                lo_i = int(np.searchsorted(F["t"], start_t[tf]))
                if len(F["c"]) - lo_i < 20:
                    continue
                i = int(rng.integers(max(lo_i, WARMUP), len(F["c"]) - 2))
                rv = sim(F, i, int(rng.choice([-1, 1])), sl_atr, ex, HORIZON[tf])
                if rv is not None:
                    vals.append(rv)
            n, avg, _, win = stats(vals)
            base_lines.append(f"  random {tf} SL{sl_atr} {ex}: n={n} win{win:.0f}% avg{avg:+.2f}R")

    # report
    good = [r for r in results if r["tn"] >= 25 and r["vn"] >= 10]
    good.sort(key=lambda r: r["tavg"], reverse=True)
    lines = [f"BACKTEST2: {len(DATA)} coins, 15m {D15} din, 1h {D1H} din, fees 0.2% minus.",
             "Format: tf|trend|vol|SL(ATR)|exit  n win% avg R/trade  (TRAIN avg,n | VALID avg,n)", ""]
    lines.append("TOP 8 (train avg R ke hisaab se):")
    for r in good[:8]:
        lines.append(f"{r['tf']}|{r['trend']}|v{r['vol']}|SL{r['sl']}|{r['ex']}  n={r['n']} win{r['win']:.0f}% "
                     f"{r['avg']:+.2f}R  (T {r['tavg']:+.2f},{r['tn']} | V {r['vavg']:+.2f},{r['vn']})")
    consistent = [r for r in good if r["tavg"] > 0.05 and r["vavg"] > 0.05]
    lines += ["", f"DONO (train aur valid) me positive (>+0.05R): {len(consistent)} combo"]
    for r in sorted(consistent, key=lambda r: r["vavg"], reverse=True)[:5]:
        lines.append(f"  {r['tf']}|{r['trend']}|v{r['vol']}|SL{r['sl']}|{r['ex']}  n={r['n']} win{r['win']:.0f}% "
                     f"{r['avg']:+.2f}R (T {r['tavg']:+.2f} | V {r['vavg']:+.2f})")
    lines += ["", "Baseline (random entry, same exit):"] + base_lines
    lines += ["", "Samajh: agar top combos ka avg R random baseline jitna hi hai ya valid me negative hai, "
                  "to strategy me asli edge nahi."]
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
