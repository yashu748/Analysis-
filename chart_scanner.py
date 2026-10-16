"""
Chart scanner: Gate.io coins jinme khareedari + trend upar ki taraf hai, unka analysis karke
chart ki photo ke saath Telegram alert bhejta hai.
buy_pressure_scanner.py, sol_bot.py, team_scanner.py ke saath ek hi folder me rakho.

Pipeline:
 1) buy_pressure_scanner ka screening (liquid coins, volume spike, CMF, OBV, trend,
    taker buying %, order book)
 2) Chart structure se levels: SL swing low ke neeche (ATR ke andar clamp), targets agle
    resistance levels par (nahi mile to 1.5R / 3R)
 3) Resistance bilkul upar ho to setup skip (R:R kharab)
 4) Chart image (candles, EMA20/50, Entry/SL/T1/T2, volume) + caption Telegram par

ENV (optional): MIN_SCORE (6), MIN_BUY_RATIO (0.58), COOLDOWN_H (6), MAX_ALERTS (5)
Ye rule-based analysis hai, live AI model nahi. Advice nahi.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402

import sol_bot as bot  # noqa: E402
import buy_pressure_scanner as bp  # noqa: E402

MAX_ALERTS = int(os.getenv("MAX_ALERTS", "5"))
STATE_FILE = "alert_state_chart.json"
CHART_BARS = 80
MIN_RISK_PCT = float(os.getenv("MIN_RISK_PCT", "0.6")) / 100   # SL isse chhota ho to fees kha jati hain
MIN_CONFIRM = int(os.getenv("MIN_CONFIRM", "4"))

def pivots(h, l, w=3):
    ph, pl = [], []
    for i in range(w, len(h) - w):
        if h[i] == max(h[i - w:i + w + 1]):
            ph.append(h[i])
        if l[i] == min(l[i - w:i + w + 1]):
            pl.append(l[i])
    return ph, pl


def load_candles(pair):
    raw = bot.get("/spot/candlesticks",
                  {"currency_pair": pair, "interval": bp.TF, "limit": 200})[:-1]
    o = [bp.f(x[5]) for x in raw]
    h = [bp.f(x[3]) for x in raw]
    l = [bp.f(x[4]) for x in raw]
    c = [bp.f(x[2]) for x in raw]
    v = [bp.f(x[6]) for x in raw]
    return o, h, l, c, v
def ema_list(x, n):
    k = 2 / (n + 1)
    out = [x[0]]
    for p in x[1:]:
        out.append(p * k + out[-1] * (1 - k))
    return out


def rsi_last(c, n=14):
    g = ls = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        ls += max(-d, 0)
    g, ls = g / n, ls / n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        ls = (ls * (n - 1) + max(-d, 0)) / n
    return 100.0 if ls == 0 else 100 - 100 / (1 + g / ls)


def macd_state(c):
    e12, e26 = ema_list(c, 12), ema_list(c, 26)
    m = [a - b for a, b in zip(e12, e26)]
    sg = ema_list(m, 9)
    hist = [a - b for a, b in zip(m, sg)]
    return m[-1] > sg[-1] and hist[-1] > hist[-2]


def supertrend_up(h, l, c, n=10, mult=3.0):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]),
                              abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    atr = [tr[0]]
    for i in range(1, len(c)):
        atr.append((atr[-1] * (n - 1) + tr[i]) / n)
    fu = fl = 0.0
    up = True
    for i in range(len(c)):
        hl = (h[i] + l[i]) / 2
        bu, bl = hl + mult * atr[i], hl - mult * atr[i]
        if i == 0:
            fu, fl = bu, bl
            continue
        fu = bu if (bu < fu or c[i - 1] > fu) else fu
        fl = bl if (bl > fl or c[i - 1] < fl) else fl
        if up and c[i] < fl:
            up = False
        elif not up and c[i] > fu:
            up = True
    return up


def confirm(o, h, l, c, v):
    e20, e50 = ema_list(c, 20)[-1], ema_list(c, 50)[-1]
    r = rsi_last(c)
    vavg = sum(v[-21:-1]) / 20
    checks = {
        "EMA": c[-1] > e20 > e50,
        "Supertrend": supertrend_up(h, l, c),
        "RSI 50-70": 50 <= r <= 70,
        "MACD": macd_state(c),
        "Volume": v[-1] > 1.2 * vavg,
    }
    return sum(checks.values()), checks, r

def plan_levels(entry, atr, h, l):
    """Structure-based SL / T1 / T2. None return = setup skip."""
    ph, pl = pivots(h[-150:], l[-150:])
    swing = min(l[-20:])
    risk_struct = entry - (swing - 0.2 * atr)
    risk = min(max(risk_struct, 1.5 * atr), 3.5 * atr)
    sl = entry - risk
    res = sorted({p for p in ph if p > entry * 1.002})
    if any(entry < r < entry + 1.0 * risk for r in res):
        return None  # resistance bilkul paas, R:R kharab
    cands = [r for r in res if r >= entry + 1.2 * risk]
    t1 = cands[0] if cands and cands[0] <= entry + 3 * risk else entry + 1.5 * risk
    later = [r for r in cands if r > t1 * 1.002 and r <= entry + 6 * risk]
    t2 = later[0] if later else entry + 3 * risk
    if t2 <= t1:
        t2 = t1 + 1.5 * risk
    pl_below = sorted([p for p in pl if p < entry], reverse=True)
    return dict(sl=sl, t1=t1, t2=t2, risk=risk, res=res[:3],
                support=pl_below[0] if pl_below else None,
                rr1=(t1 - entry) / risk, rr2=(t2 - entry) / risk)


def make_chart(pair, o, h, l, c, v, entry, lv, score, path):
    n = len(c)
    s = max(0, n - CHART_BARS)
    xs = list(range(n - s))
    e20, e50 = bot.ema(c, 20)[s:], bot.ema(c, 50)[s:]
    bg, fg = "#0e1117", "#d0d4dc"
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(9, 6.2), dpi=110, sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
    fig.patch.set_facecolor(bg)
    for a in (ax, axv):
        a.set_facecolor(bg)
        a.tick_params(colors=fg, labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a2f3a")
    for i, k in enumerate(range(s, n)):
        up = c[k] >= o[k]
        col = "#26a69a" if up else "#ef5350"
        ax.vlines(i, l[k], h[k], color=col, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o[k], c[k])), 0.6,
                                   max(abs(c[k] - o[k]), 1e-12), color=col))
        axv.bar(i, v[k], color=col, width=0.7, alpha=0.7)
    ax.plot(xs, e20, color="#f5b942", linewidth=1.1, label="EMA20")
    ax.plot(xs, e50, color="#5aa9ff", linewidth=1.1, label="EMA50")
    right = len(xs) + 3
    ax.axhspan(lv["sl"], entry, xmin=0.0, xmax=1.0, color="#ef5350", alpha=0.07)
    ax.axhspan(entry, lv["t2"], xmin=0.0, xmax=1.0, color="#26a69a", alpha=0.06)
    for y, col, name, ls in ((entry, "#ffffff", "Entry", "--"),
                             (lv["sl"], "#ef5350", "SL", "-"),
                             (lv["t1"], "#26a69a", "T1", "-"),
                             (lv["t2"], "#7bdcb5", "T2", "-")):
        ax.axhline(y, color=col, linestyle=ls, linewidth=1.1)
        ax.text(right, y, f"{name} {y:.6g}", color=col, fontsize=8, va="center")
    for r in lv["res"]:
        ax.axhline(r, color="#8a8f9c", linestyle=":", linewidth=0.8)
    ax.set_xlim(-1, right + 14)
    lo = min(min(l[s:]), lv["sl"]) * 0.995
    hi = max(max(h[s:]), lv["t2"]) * 1.005
    ax.set_ylim(lo, hi)
    ax.set_title(f"{pair}  {bp.TF}   setup score {score:.0f}/8", color=fg, fontsize=11, loc="left")
    ax.legend(loc="upper left", facecolor=bg, edgecolor="#2a2f3a", labelcolor=fg, fontsize=8)
    ax.grid(color="#1d222c", linewidth=0.5)
    axv.grid(color="#1d222c", linewidth=0.5)
    axv.set_yticks([])
    fig.savefig(path, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def caption(x, lv, entry):
    d = bp.dec(entry)
    risk_pct = (entry - lv["sl"]) / entry * 100
    lines = [
        f"{x['pair']}  LONG setup  (score {x['score']:.0f}/8, 24h {x['chg']:+.1f}%)",
        f"Entry ~{entry:.{d}f}",
        f"SL {lv['sl']:.{d}f}  (-{risk_pct:.1f}%)",
        f"T1 {lv['t1']:.{d}f}  ({lv['rr1']:.1f}R)  |  T2 {lv['t2']:.{d}f}  ({lv['rr2']:.1f}R)",
        "",
        f"Kyun: taker buying {x['buy_ratio'] * 100:.0f}%, book bids {x['imb'] * 100:.0f}%, "
                f"volume {x['spike']:.1f}x, CMF {x['cmf']:+.2f}, EMA20>EMA50"
        f"Indicators: {x.get('ind', '')}",
    
    if lv["res"]:
        lines.append("Upar resistance: " + ", ".join(f"{r:.{d}f}" for r in lv["res"]))
    if lv["support"]:
        lines.append(f"Neeche support: {lv['support']:.{d}f}")
    lines += ["", "T1 par 50% book, SL entry par. Alert hai, advice nahi. SL zaroor lagao."]
    return "\n".join(lines)[:1000]


def send_photo(path, cap):
    token, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not token or not chat:
        print("Telegram keys nahi mili, sirf print kar raha hoon.")
        print(cap)
        return False
    try:
        with open(path, "rb") as fh:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                              data={"chat_id": chat, "caption": cap},
                              files={"photo": fh}, timeout=30)
        print("Telegram photo:", r.status_code, r.text[:120])
        return r.status_code == 200
    except Exception as e:
        print("Photo fail:", repr(e)[:150])
        return False


def load_state():
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(state):
    cutoff = time.time() - 48 * 3600
    with open(STATE_FILE, "w") as fh:
        json.dump({k: v for k, v in state.items() if v > cutoff}, fh)


def main():
    uni, passed = bp.universe()
    print(f"Universe: {passed} coins liquidity paas, {len(uni)} scan honge")
    stage = []
    for r in uni:
        try:
            x = bp.stage_a(r)
        except Exception as e:
            print(f"{r['pair']}: skip A ({e})")
            continue
        time.sleep(0.12)
        if x and x["ext"] <= 3.0 and x["chg"] <= 40:
            stage.append(x)
    stage.sort(key=lambda x: (x["score_a"], x["spike"]), reverse=True)
    deep = []
    for x in stage[:bp.TOP_DEEP]:
        if x["score_a"] < 3:
            break
        try:
            y = bp.stage_b(x)
        except Exception as e:
            print(f"{x['pair']}: skip B ({e})")
            continue
        time.sleep(0.15)
        if y:
            deep.append(y)
            print(f"{y['pair']}: A={y['score_a']:.0f} buy={y['buy_ratio']:.2f} "
                  f"book={y['imb']:.2f} -> {y['score']:.0f}")
    cands = [y for y in deep if y["score"] >= bp.MIN_SCORE and y["buy_ratio"] >= bp.MIN_BUY_RATIO]
    cands.sort(key=lambda y: (y["score"], y["buy_ratio"]), reverse=True)

    state, now, sent = load_state(), time.time(), 0
    for y in cands:
        if sent >= MAX_ALERTS:
            break
        if now - state.get(y["pair"], 0) < bp.COOLDOWN_H * 3600:
            continue
        try:
            o, h, l, c, v = load_candles(y["pair"])
            conf, checks, rsi_v = confirm(o, h, l, c, v)
            if conf < MIN_CONFIRM:
                continue
            y["ind"] = " ".join(("✅" if ok else "❌") + k for k, ok in checks.items())
            ask = y["ask"]
            entry = ask if ask and abs(ask / c[-1] - 1) < 0.03 else c[-1]
            lv = plan_levels(entry, y["atr"], h, l)
            if not lv:
                print(f"{y['pair']}: resistance paas, skip")
                continue
            if (entry - lv["sl"]) / entry < MIN_RISK_PCT:
                print(f"{y['pair']}: SL bahut chhota (fees ka asar), skip")
                continue
            path = f"/tmp/{y['pair']}.png"
            make_chart(y["pair"], o, h, l, c, v, entry, lv, y["score"], path)
            cap = caption(y, lv, entry)
            ok = send_photo(path, cap)
            if not ok:
                bot.send_telegram(cap)       # photo na jaye to text alert
            state[y["pair"]] = now
            sent += 1
        except Exception as e:
            print(f"{y['pair']}: alert fail ({repr(e)[:150]})")
        time.sleep(0.3)
    save_state(state)
    print(f"Alerts bheje: {sent}")
    return sent


if __name__ == "__main__":
    main()
 
