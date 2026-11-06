#!/usr/bin/env python3
"""
backtest.py - chart_bot.py ke signals ko pichhle kuch din ke ASLI Gate.io data par parakhta hai.

Har 15m candle par bot ka analyze() waise hi chalta hai jaise us waqt chalta (sirf us waqt tak ka data dikhta hai).
Jo signal banta hai uska SL / T1 / T2 result agli 24 ghante ki candles se nikalta hai
(same candle me SL aur target dono touch ho to SL pehle maana jata hai = conservative).
Fees (0.2% round trip) minus hoti hai. 3 modes (safe / balanced / loose) alag-alag naapta hai.

Chalao: GitHub Actions se "backtest" workflow (Run workflow). Result log me aur Telegram par aayega.
Env (optional): BT_DAYS (default 6, max 6), BT_PAIRS (default 30), BT_BUDGET_MIN (default 45)
"""
import os
import time
import traceback

import numpy as np
import pandas as pd

import chart_bot as b

DAYS = min(int(os.getenv("BT_DAYS", "6")), 6)
PAIRS_N = int(os.getenv("BT_PAIRS", "30"))
BUDGET = float(os.getenv("BT_BUDGET_MIN", "45")) * 60
HORIZON = 96          # 24 ghante ki 15m candles
COOLDOWN_C = 24       # same coin + same side: 6 ghante gap (bot ke COOLDOWN_MIN jaisa)
FEE = 0.002           # round trip fee (0.1% + 0.1%)
SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
HIST_DAYS = {"15m": DAYS + 4, "1h": 40, "4h": 150, "1d": 400}

MODES = {
    "safe": dict(USE_PUMP=True, USE_PATTERN=True, RES_MIN_R=1.0, LONG_RSI_MAX=62, SHORT_RSI_MIN=38,
                 MIN_CONFIRM=5, VOL_MULT=2.0, BTC_FILTER=True, TREND_STRICT=True, MAX_RISK_PCT=4.0,
                 HTF_RSI_MAX=70, HTF_RSI_MIN=30, MIN_Q=75),
    "balanced": dict(USE_PUMP=True, USE_PATTERN=False, RES_MIN_R=0.5, LONG_RSI_MAX=66, SHORT_RSI_MIN=34,
                     MIN_CONFIRM=4, VOL_MULT=1.5, BTC_FILTER=False, TREND_STRICT=False, MAX_RISK_PCT=6.0,
                     HTF_RSI_MAX=70, HTF_RSI_MIN=30, MIN_Q=65),
    "loose": dict(USE_PUMP=False, USE_PATTERN=True, RES_MIN_R=0.0, LONG_RSI_MAX=72, SHORT_RSI_MIN=28,
                  MIN_CONFIRM=4, VOL_MULT=1.5, BTC_FILTER=False, TREND_STRICT=False, MAX_RISK_PCT=8.0,
                  HTF_RSI_MAX=101, HTF_RSI_MIN=-1, MIN_Q=0),
}

HIST = {}
ASOF = [0.0]


def load(pair, tf):
    end = int(time.time())
    start = end - HIST_DAYS[tf] * 86400
    r = b.get(f"{b.BASE}/spot/candlesticks",
              {"currency_pair": pair, "interval": tf, "from": start, "to": end})
    if not r or len(r) < 30:
        return None
    df = pd.DataFrame([[float(x[0]), float(x[5]), float(x[4]), float(x[3]), float(x[2]), float(x[1])]
                       for x in r], columns=["t", "o", "l", "h", "c", "v"])
    df = df[["t", "o", "h", "l", "c", "v"]].sort_values("t").reset_index(drop=True)
    return df


def mock_fetch(pair, tf, limit):
    """Bot ko sirf ASOF tak ki CLOSED candles dikhao."""
    df = HIST.get((pair, tf))
    if df is None:
        return None
    d = df[df.t + SEC[tf] <= ASOF[0]]
    if len(d) < 30:
        return None
    return d.tail(limit).reset_index(drop=True)


def run_mode(mode, cfg, pairs, t_start):
    for k, v in cfg.items():
        if k != "MIN_Q":
            setattr(b, k, v)
    b.MIN_QUALITY = 0                      # quality hum baad me bucket karke dekhenge
    sigs = []
    for pair in pairs:
        if time.time() - t_start > BUDGET:
            print("time budget poora, baaki pairs skip", flush=True)
            break
        df15 = HIST[(pair, "15m")]
        n = len(df15)
        test_from = df15.t.iloc[-1] - DAYS * 86400
        last_sig = {}
        for i in range(250, n - 1):
            if df15.t.iloc[i] < test_from:
                continue
            ASOF[0] = df15.t.iloc[i] + 900
            b.seen.clear(); b.hcache.clear(); b.last_alert.clear(); b.stats.clear()
            try:
                s = b.analyze(pair)
            except Exception:
                traceback.print_exc()
                continue
            if not s:
                continue
            if i - last_sig.get(s["side"], -999) < COOLDOWN_C:
                continue
            last_sig[s["side"]] = i
            fut = df15.iloc[i + 1:i + 1 + HORIZON]
            if len(fut) < 8:
                continue                   # result abhi pata nahi
            tr_ = dict(pair=pair, side=s["side"], entry=s["price"], stop=s["sl"], t1=s["t1"], t2=s["t2"],
                       t1_hit=False)
            res = b.evaluate(tr_, fut)
            if res is None:
                name, rv = ("open", b.T1_R * 0.5 if tr_["t1_hit"] else 0.0)
            else:
                name, rv = res
            risk_abs = abs(s["price"] - s["sl"])
            rv_net = rv - FEE * s["price"] / risk_abs if risk_abs > 0 else rv
            sigs.append(dict(pair=pair, side=s["side"], q=s["q"], rv=rv_net, name=name))
        print(f"[{mode}] {pair}: total signals so far {len(sigs)}", flush=True)
    return sigs


def summarize(mode, cfg, sigs):
    lines = [f"MODE {mode}  (bot ka default quality minimum: {cfg['MIN_Q']})"]
    if not sigs:
        lines.append("  koi signal nahi bana")
        return lines
    df = pd.DataFrame(sigs)
    for thr in sorted({cfg["MIN_Q"], 70, 75, 80, 85, 90}):
        d = df[df.q >= thr]
        if len(d) == 0:
            lines.append(f"  q>={thr}: 0 signal")
            continue
        w, l = int((d.rv > 0).sum()), int((d.rv <= 0).sum())
        lines.append(f"  q>={thr}: {len(d)} signal (~{len(d) / DAYS:.1f}/din), win {w / (w + l) * 100:.0f}%, "
                     f"net {d.rv.sum():+.1f}R, avg {d.rv.mean():+.2f}R/trade")
    return lines


def main():
    t0 = time.time()
    pairs = [p for p in b.coin_list() if p != "BTC_USDT"][:PAIRS_N]
    print(f"Backtest: {len(pairs)} coins, {DAYS} din, 3 modes", flush=True)
    need = pairs + ["BTC_USDT"]
    for p in need:
        for tf in SEC:
            HIST[(p, tf)] = load(p, tf)
    pairs = [p for p in pairs if all(HIST.get((p, tf)) is not None for tf in SEC) and len(HIST[(p, "15m")]) > 300]
    print(f"Data mila: {len(pairs)} coins", flush=True)
    b.fetch = mock_fetch
    b.tg_text = lambda t: None             # evaluate() ke andar ke Telegram msg band
    out = [f"BACKTEST ({DAYS} din, {len(pairs)} coins, fees 0.2% minus). Win% me T1+BE bhi win gina gaya.", ""]
    for mode, cfg in MODES.items():
        sigs = run_mode(mode, cfg, pairs, t0)
        out += summarize(mode, cfg, sigs) + [""]
    out.append("Dhyan: ye sirf pichhle kuch din ka natija hai. Market badalta hai, isliye isse guarantee mat samjho.")
    text = "\n".join(out)
    print("\n" + text, flush=True)
    import requests
    if b.TG_TOKEN and b.TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{b.TG_TOKEN}/sendMessage",
                          data={"chat_id": b.TG_CHAT, "text": text[:4000]}, timeout=20)
        except Exception as e:
            print("telegram failed:", e)


if __name__ == "__main__":
    main()
