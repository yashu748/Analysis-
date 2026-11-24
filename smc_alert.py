"""
SMC trap alert bot -> Telegram (annotated chart photo ke saath).
  python smc_alert.py        # PAIRS scan karo, signal ho to chart + alert bhejo
  python smc_alert.py test   # sirf test message bhejo
Secrets (environment): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID. Optional: PAIRS, INTERVAL
"""
import io
import os
import sys

import numpy as np
import requests

from smc_trap import latest_signal, load_gate

INTERVAL = os.getenv("INTERVAL", "15m")
PAIRS = [x.strip() for x in os.getenv("PAIRS", "BTC_USDT,ETH_USDT,SOL_USDT").split(",") if x.strip()]
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"


def send(text):
    requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": text},
                  timeout=20).raise_for_status()


def send_photo(png, caption):
    requests.post(f"{API}/sendPhoto", data={"chat_id": CHAT, "caption": caption[:1000]},
                  files={"photo": ("smc.png", png, "image/png")}, timeout=40).raise_for_status()


def caption(pair, s):
    buy = s["side"] == "BUY"
    zone = "DEMAND" if buy else "SUPPLY"
    rr = abs(s["tp"] - s["entry"]) / abs(s["entry"] - s["sl"])
    return (f"{'🟢' if buy else '🔴'} {s['side']} {pair} ({INTERVAL}) - SMART MONEY TRAP\n\n"
            f"1) Trendline ke 3rd touch par retail {s['side']} karta hai, unka SL "
            f"{'neeche' if buy else 'upar'} = LIQUIDITY\n"
            f"2) SMART MONEY ne SL hunt kiya (sweep)\n"
            f"3) Structure break {'upar' if buy else 'neeche'} hua\n"
            f"4) {zone} ZONE retest -> ENTRY\n\n"
            f"Entry: {s['entry']:.6g}\nSL: {s['sl']:.6g}\nTarget: {s['tp']:.6g}  (RR 1:{rr:.1f})\n"
            f"Candle: {s['time']} UTC")


def draw_chart(df, s, pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d, sg, buy = s["detail"], s["sg"], s["side"] == "BUY"
    n = len(df)
    x0 = max(0, d["a"] - 12)
    xs = np.arange(x0, n)
    o, h, l, c = (df[k].to_numpy(float)[x0:] for k in ("open", "high", "low", "close"))
    cols = ["#16a34a" if cc >= oo else "#dc2626" for oo, cc in zip(o, c)]
    ly = lambda x: sg * (d["base"] + d["slope"] * (x - d["a"]))     # trendline (asli price me)
    right = n + 10

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.vlines(xs, l, h, colors=cols, linewidth=1)
    ax.bar(xs, np.abs(c - o), bottom=np.minimum(o, c), color=cols, width=0.7)

    span = max(h.max(), s["tp"], s["sl"]) - min(l.min(), s["tp"], s["sl"])
    pad = span * 0.08
    ax.set_ylim(min(l.min(), s["sl"], s["tp"]) - pad, max(h.max(), s["tp"], s["sl"]) + pad)
    ax.set_xlim(x0 - 1, right + 6)

    # trendline + 3 touch
    xe = d["mss_idx"]
    ax.plot([d["a"], xe], [ly(d["a"]), ly(xe)], color="#f97316", lw=2)
    for i in (d["a"], d["b"], d["t3"]):
        ax.scatter([i], [ly(i)], s=170, facecolors="none", edgecolors="#f97316", linewidths=2, zorder=5)
    ax.annotate(f"3rd touch: retail {s['side']} karta hai", (d["t3"], ly(d["t3"])),
                xytext=(-90, 70 * sg), textcoords="offset points", fontsize=9, color="#c2410c",
                arrowprops=dict(arrowstyle="->", color="#c2410c"))

    # liquidity (retail ke stop loss)
    band = span * 0.05
    xl = np.array([d["t3"], d["sweep_idx"] + 2])
    ax.fill_between(xl, ly(xl), ly(xl) - sg * band, color="#f59e0b", alpha=0.35)
    ax.text(xl.mean(), ly(xl.mean()) - sg * band * 1.9, "LIQUIDITY\n(retail ke SL yahan)",
            fontsize=9, color="#b45309", ha="center", va="top" if buy else "bottom")

    # smart money sweep
    j = d["low_idx"]
    jp = (df["low"] if buy else df["high"]).iloc[j]
    ax.scatter([j], [jp], s=120, color="#7c3aed", zorder=6)
    ax.annotate("SMART MONEY\nSL hunt (sweep)", (j, jp), xytext=(-70, -55 * sg),
                textcoords="offset points", fontsize=9, color="#6d28d9", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#6d28d9"))

    # structure break
    sh = sg * d["struct_hi"]
    ax.hlines(sh, d["t3"], d["mss_idx"] + 3, colors="#2563eb", linestyles="--", lw=1.4)
    ax.scatter([d["mss_idx"]], [sh], marker="^" if buy else "v", s=110, color="#2563eb", zorder=6)
    ax.text(d["t3"] + 1, sh, "STRUCTURE BREAK (MSS) level", fontsize=9, color="#1d4ed8",
            va="bottom" if buy else "top")

    # demand / supply zone
    zlo, zhi = sorted((sg * d["zl"], sg * d["zh"]))
    ax.add_patch(Rectangle((j, zlo), right - j, zhi - zlo, color="#3b82f6", alpha=0.25))
    ax.text((j + n) / 2, zlo if buy else zhi, "DEMAND ZONE" if buy else "SUPPLY ZONE",
            fontsize=9, color="#1e40af", ha="center", va="top" if buy else "bottom", fontweight="bold")

    # entry / SL / target
    e, sl, tp, xi = s["entry"], s["sl"], s["tp"], n - 1
    ax.add_patch(Rectangle((xi, min(e, tp)), right - xi, abs(tp - e), color="#22c55e", alpha=0.18))
    ax.add_patch(Rectangle((xi, min(e, sl)), right - xi, abs(e - sl), color="#ef4444", alpha=0.18))
    for val, name, col in ((tp, "TARGET (upar ki liquidity)", "#15803d"), (e, "ENTRY", "#1d4ed8"), (sl, "STOP LOSS", "#b91c1c")):
        ax.hlines(val, xi, right, colors=col, lw=1.6)
        ax.text(right + 0.5, val, f"{name} {val:.6g}", fontsize=9, color=col, va="center", fontweight="bold")

    ticks = np.linspace(x0, n - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([df.index[i].strftime("%d %b %H:%M") for i in ticks], fontsize=8)
    ax.set_title(f"{s['side']} {pair} | {INTERVAL} | SMC Smart Money Trap", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send("✅ SMC alert bot: Telegram connect ho gaya")
        return
    for pair in PAIRS:
        try:
            df = load_gate(pair, INTERVAL)
            for s in latest_signal(df):
                if s["kind"] != "SIGNAL":
                    continue
                try:
                    send_photo(draw_chart(df, s, pair), caption(pair, s))
                except Exception as e:      # chart fail ho to text alert phir bhi jaye
                    print(pair, "chart error:", e)
                    send(caption(pair, s))
        except Exception as e:              # ek pair fail ho to baaki chalte rahein
            print(pair, "error:", e)


if __name__ == "__main__":
    main()
