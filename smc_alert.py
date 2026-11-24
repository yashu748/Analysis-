
#!/usr/bin/env python3
"""
SMC Smart Money Trap Alert Bot
15-minute timeframe
Gate.io + Telegram + Annotated Chart

Commands:
python smc_alert.py
python smc_alert.py test

Environment secrets:
TELEGRAM_TOKEN
TELEGRAM_CHAT_ID

Optional:
PAIRS=BTC_USDT,ETH_USDT,SOL_USDT
INTERVAL=15m
"""

import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import requests

from smc_trap import latest_signal


# =====================================================
# CONFIGURATION
# =====================================================

INTERVAL = os.getenv("INTERVAL", "15m")

PAIRS = [
    x.strip()
    for x in os.getenv(
        "PAIRS",
        "BTC_USDT,ETH_USDT,SOL_USDT"
    ).split(",")
    if x.strip()
]

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

GATE_URL = (
    "https://api.gateio.ws/api/v4/spot/candlesticks"
)

TIMEOUT = 25
CANDLE_LIMIT = 1000


# =====================================================
# TELEGRAM VALIDATION
# =====================================================

def check_telegram():

    if not TOKEN:
        raise ValueError(
            "TELEGRAM_TOKEN GitHub Secrets mein missing hai."
        )

    if not CHAT:
        raise ValueError(
            "TELEGRAM_CHAT_ID GitHub Secrets mein missing hai."
        )


def telegram_request(method, data=None, files=None):

    check_telegram()

    url = (
        f"https://api.telegram.org/bot{TOKEN}/{method}"
    )

    response = requests.post(
        url,
        data=data,
        files=files,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API error: {result}"
        )

    return result


# =====================================================
# SEND TELEGRAM MESSAGE
# =====================================================

def send(text):

    return telegram_request(
        "sendMessage",
        data={
            "chat_id": CHAT,
            "text": text
        }
    )


# =====================================================
# SEND CHART PHOTO
# =====================================================

def send_photo(png, caption):

    return telegram_request(
        "sendPhoto",
        data={
            "chat_id": CHAT,
            "caption": caption[:1000]
        },
        files={
            "photo": (
                "smc_chart.png",
                png,
                "image/png"
            )
        }
    )


# =====================================================
# GATE.IO CANDLE DATA LOADER
# =====================================================

def load_gate(
    pair="BTC_USDT",
    interval="15m",
    limit=1000
):

    response = requests.get(
        GATE_URL,
        params={
            "currency_pair": pair,
            "interval": interval,
            "limit": min(int(limit), 1000)
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    rows = response.json()

    if not isinstance(rows, list) or not rows:

        raise ValueError(
            f"{pair}: Gate.io candle data nahi mila."
        )

    candles = []

    for row in rows:

        if not isinstance(row, (list, tuple)):
            continue

        if len(row) < 7:
            continue

        try:

            timestamp = int(row[0])

            # Gate.io candle:
            # timestamp, quote_volume, close,
            # high, low, open, base_volume
            #
            # Some responses may contain an
            # additional closed-status field.

            if len(row) >= 8:

                status = str(row[7]).strip().lower()

                if status in ("false", "0"):
                    continue

            candle = {
                "timestamp": timestamp,
                "open": float(row[5]),
                "high": float(row[3]),
                "low": float(row[4]),
                "close": float(row[2])
            }

            candles.append(candle)

        except (ValueError, TypeError, IndexError):
            continue

    if not candles:

        raise ValueError(
            f"{pair}: Valid candle rows nahi mile."
        )

    df = pd.DataFrame(candles)

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.set_index("datetime")

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    # Remove incomplete candle

    now = pd.Timestamp.now(tz="UTC")

    duration = pd.Timedelta(interval)

    df = df[
        df.index + duration <= now
    ]

    df = df[
        ["open", "high", "low", "close"]
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(df) < 50:

        raise ValueError(
            f"{pair}: Sirf {len(df)} completed candles mile."
        )

    return df.astype(float)


# =====================================================
# SIGNAL CAPTION
# =====================================================

def caption(pair, s):

    buy = s["side"] == "BUY"

    zone = (
        "DEMAND"
        if buy
        else "SUPPLY"
    )

    risk = abs(
        s["entry"] - s["sl"]
    )

    reward = abs(
        s["tp"] - s["entry"]
    )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )

    emoji = (
        "🟢"
        if buy
        else "🔴"
    )

    return (
        f"{emoji} {s['side']} {pair} ({INTERVAL})\n"
        f"SMART MONEY TRAP\n\n"

        f"1. Three trendline touches confirmed\n"
        f"2. Liquidity sweep detected\n"
        f"3. Market structure break confirmed\n"
        f"4. {zone} zone retest entry\n\n"

        f"Entry: {s['entry']:.8g}\n"
        f"Stop Loss: {s['sl']:.8g}\n"
        f"Target: {s['tp']:.8g}\n\n"

        f"Risk Reward: 1:{rr:.2f}\n"
        f"Candle: {s['time']}\n\n"

        f"Timeframe: {INTERVAL}\n"
        f"Strategy: SMC Trap\n"
        f"Signal only - not an automatic trade."
    )


# =====================================================
# DRAW ANNOTATED CHART
# =====================================================

def draw_chart(df, s, pair):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from matplotlib.patches import Rectangle

    d = s["detail"]

    sg = s["sg"]

    buy = s["side"] == "BUY"

    n = len(df)

    if n < 10:
        raise ValueError("Chart ke liye sufficient candles nahi hain.")

    x0 = max(
        0,
        d["a"] - 12
    )

    xs = np.arange(
        x0,
        n
    )

    o = df["open"].to_numpy(float)[x0:]
    h = df["high"].to_numpy(float)[x0:]
    l = df["low"].to_numpy(float)[x0:]
    c = df["close"].to_numpy(float)[x0:]

    cols = [
        "#16a34a" if cc >= oo else "#dc2626"
        for oo, cc in zip(o, c)
    ]

    def trendline(x):

        return sg * (
            d["base"]
            + d["slope"] * (x - d["a"])
        )

    right = n + 10

    fig, ax = plt.subplots(
        figsize=(12, 8),
        dpi=120
    )

    try:

        # -----------------------------------------
        # CANDLESTICKS
        # -----------------------------------------

        ax.vlines(
            xs,
            l,
            h,
            colors=cols,
            linewidth=1
        )

        ax.bar(
            xs,
            np.abs(c - o),
            bottom=np.minimum(o, c),
            color=cols,
            width=0.7
        )

        # -----------------------------------------
        # CHART LIMITS
        # -----------------------------------------

        price_values = [
            h.max(),
            l.min(),
            s["entry"],
            s["sl"],
            s["tp"]
        ]

        upper = max(price_values)
        lower = min(price_values)

        span = max(
            upper - lower,
            abs(s["entry"]) * 0.001,
            1e-8
        )

        pad = span * 0.12

        ax.set_ylim(
            lower - pad,
            upper + pad
        )

        ax.set_xlim(
            x0 - 1,
            right + 7
        )

        # -----------------------------------------
        # TRENDLINE
        # -----------------------------------------

        xe = d["mss_idx"]

        ax.plot(
            [d["a"], xe],
            [
                trendline(d["a"]),
                trendline(xe)
            ],
            color="#f97316",
            linewidth=2,
            label="Trendline"
        )

        for i in (
            d["a"],
            d["b"],
            d["t3"]
        ):

            ax.scatter(
                [i],
                [trendline(i)],
                s=140,
                facecolors="none",
                edgecolors="#f97316",
                linewidths=2,
                zorder=5
            )

        # -----------------------------------------
        # THIRD TOUCH
        # -----------------------------------------

        ax.annotate(
            "3rd touch",
            (
                d["t3"],
                trendline(d["t3"])
            ),
            xytext=(0, 30),
            textcoords="offset points",
            fontsize=9,
            color="#c2410c",
            arrowprops={
                "arrowstyle": "->",
                "color": "#c2410c"
            }
        )

        # -----------------------------------------
        # LIQUIDITY AREA
        # -----------------------------------------

        band = span * 0.04

        xl = np.array(
            [
                d["t3"],
                d["sweep_idx"] + 2
            ]
        )

        ax.fill_between(
            xl,
            trendline(xl),
            trendline(xl) - sg * band,
            color="#f59e0b",
            alpha=0.30
        )

        ax.text(
            xl.mean(),
            trendline(xl.mean()),
            "LIQUIDITY SWEEP AREA",
            fontsize=8,
            color="#b45309"
        )

        # -----------------------------------------
        # SWEEP POINT
        # -----------------------------------------

        j = d["low_idx"]

        jp = (
            df["low"].iloc[j]
            if buy
            else df["high"].iloc[j]
        )

        ax.scatter(
            [j],
            [jp],
            s=130,
            color="#7c3aed",
            zorder=6
        )

        ax.annotate(
            "SMART MONEY SWEEP",
            (j, jp),
            xytext=(15, -35 if buy else 35),
            textcoords="offset points",
            fontsize=8,
            color="#6d28d9",
            arrowprops={
                "arrowstyle": "->",
                "color": "#6d28d9"
            }
        )

        # -----------------------------------------
        # MARKET STRUCTURE BREAK
        # -----------------------------------------

        sh = sg * d["struct_hi"]

        ax.hlines(
            sh,
            d["t3"],
            d["mss_idx"] + 3,
            colors="#2563eb",
            linestyles="--",
            linewidth=1.5
        )

        ax.scatter(
            [d["mss_idx"]],
            [sh],
            marker="^" if buy else "v",
            s=100,
            color="#2563eb",
            zorder=6
        )

        ax.text(
            d["t3"] + 1,
            sh,
            "MSS / STRUCTURE BREAK",
            fontsize=8,
            color="#1d4ed8"
        )

        # -----------------------------------------
        # DEMAND / SUPPLY ZONE
        # -----------------------------------------

        zlo, zhi = sorted(
            (
                sg * d["zl"],
                sg * d["zh"]
            )
        )

        ax.add_patch(
            Rectangle(
                (j, zlo),
                right - j,
                max(zhi - zlo, 1e-9),
                color="#3b82f6",
                alpha=0.25
            )
        )

        ax.text(
            (j + n) / 2,
            (zlo + zhi) / 2,
            "DEMAND ZONE" if buy else "SUPPLY ZONE",
            fontsize=9,
            color="#1e40af",
            ha="center",
            va="center",
            fontweight="bold"
        )

        # -----------------------------------------
        # ENTRY / SL / TARGET
        # -----------------------------------------

        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]

        xi = n - 1

        ax.add_patch(
            Rectangle(
                (xi, min(entry, tp)),
                right - xi,
                abs(tp - entry),
                color="#22c55e",
                alpha=0.15
            )
        )

        ax.add_patch(
            Rectangle(
                (xi, min(entry, sl)),
                right - xi,
                abs(entry - sl),
                color="#ef4444",
                alpha=0.15
            )
        )

        levels = [
            (
                tp,
                "TARGET",
                "#15803d"
            ),
            (
                entry,
                "ENTRY",
                "#1d4ed8"
            ),
            (
                sl,
                "STOP LOSS",
                "#b91c1c"
            )
        ]

        for val, name, color in levels:

            ax.hlines(
                val,
                xi,
                right,
                colors=color,
                linewidth=1.7
            )

            ax.text(
                right + 0.5,
                val,
                f"{name} {val:.6g}",
                fontsize=8,
                color=color,
                va="center",
                fontweight="bold"
            )

        # -----------------------------------------
        # X AXIS
        # -----------------------------------------

        ticks = np.linspace(
            x0,
            n - 1,
            6
        ).astype(int)

        ax.set_xticks(ticks)

        ax.set_xticklabels(
            [
                df.index[i].strftime("%d %b %H:%M")
                for i in ticks
            ],
            fontsize=8
        )

        ax.set_title(
            f"{s['side']} {pair} | {INTERVAL} | SMC SMART MONEY TRAP",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)

        ax.legend(loc="upper left")

        # -----------------------------------------
        # SAVE IMAGE
        # -----------------------------------------

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight"
        )

        buf.seek(0)

        return buf.getvalue()

    finally:

        plt.close(fig)


# =====================================================
# MAIN SCANNER
# =====================================================

def main():

    # -----------------------------------------
    # TELEGRAM TEST
    # -----------------------------------------

    if (
        len(sys.argv) > 1
        and sys.argv[1].lower() == "test"
    ):

        send(
            "SMC Alert Bot: Telegram connection successful."
        )

        print("Telegram test successful.")

        return

    # -----------------------------------------
    # CONFIG CHECK
    # -----------------------------------------

    check_telegram()

    if not PAIRS:

        raise ValueError(
            "PAIRS list empty hai."
        )

    print("\n================================")
    print("SMC SMART MONEY TRAP BOT")
    print("================================")

    print("Timeframe:", INTERVAL)
    print("Pairs:", PAIRS)

    # -----------------------------------------
    # SCAN EACH PAIR
    # -----------------------------------------

    total_signals = 0

    for pair in PAIRS:

        print(
            f"\nScanning {pair}..."
        )

        try:

            # Fetch candles
            df = load_gate(
                pair=pair,
                interval=INTERVAL,
                limit=CANDLE_LIMIT
            )

            print(
                f"{pair}: {len(df)} candles loaded."
            )

            # Find BUY / SELL signals
            signals = latest_signal(df)

            if not signals:

                print(
                    f"{pair}: No signal."
                )

                continue

            for s in signals:

                if s.get("kind") != "SIGNAL":
                    continue

                total_signals += 1

                print(
                    f"{pair}: {s['side']} signal detected."
                )

                message = caption(
                    pair,
                    s
                )

                # ---------------------------------
                # SEND CHART + TELEGRAM ALERT
                # ---------------------------------

                try:

                    chart = draw_chart(
                        df,
                        s,
                        pair
                    )

                    send_photo(
                        chart,
                        message
                    )

                    print(
                        f"{pair}: Chart alert sent."
                    )

                except Exception as chart_error:

                    print(
                        f"{pair}: Chart error: {chart_error}"
                    )

                    # Text-only fallback
                    send(message)

                    print(
                        f"{pair}: Text alert sent."
                    )

        except Exception as e:

            print(
                f"{pair} ERROR: {type(e).__name__}: {e}"
            )

            traceback.print_exc()

            # Continue scanning other pairs
            continue

    print("\n================================")
    print("SCAN COMPLETED")
    print("Total signals:", total_signals)
    print("================================")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            "\nFATAL ERROR:",
            type(e).__name__,
            str(e)
        )

        raise
      
