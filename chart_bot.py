
#!/usr/bin/env python3
"""
GATE.IO STRICT SIGNAL BOT V3
15m Signals | 1H / 4H / 1D Trend Filter
Telegram Chart Alerts | Strict 5/5 Confirmation

IMPORTANT:
This bot sends alerts only. It does NOT execute trades.
"""

import os
import io
import re
import time
import threading
import traceback

from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# CONFIGURATION
# =====================================================

TG_TOKEN = (
    os.getenv("TG_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or os.getenv("BOT_TOKEN", "")
)

TG_CHAT = (
    os.getenv("TG_CHAT_ID")
    or os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID", "")
)

BASE = "https://api.gateio.ws/api/v4"

ALERT_TF = "15m"

HTF = {
    "1h": 900,
    "4h": 1800,
    "1d": 7200
}

SCAN_EVERY = 120

RUN_MINUTES = float(
    os.getenv("RUN_MINUTES", "350")
)

# Liquidity
MIN_VOL_USDT = 1_000_000
MOVER_PCT = 10.0
MOVER_MIN_VOL = 600_000

MAX_COINS = 200
MAX_ALERTS_PER_SCAN = 2

# Strict confirmation
MIN_CONFIRM = 5

# Cooldown
COOLDOWN_MIN = 360

# RSI
LONG_RSI_MIN = 52
LONG_RSI_MAX = 65

SHORT_RSI_MIN = 35
SHORT_RSI_MAX = 48

# Volume
VOL_MULT = 2.0

# ADX
MIN_ADX = 20

# Higher timeframe
MIN_HTF_ALIGN = 2
MAX_HTF_OPPOSITE = 0

# Candle quality
MIN_BODY_ATR = 0.25
CLOSE_ZONE = 0.70

# Extension filter
MAX_EXTENSION_ATR = 2.0

# Stop loss
ATR_MULT = 1.5
MAX_RISK_PCT = 4.0
MIN_RISK_PCT = 0.20

# Targets
T1_R = 1.5
T2_R = 3.0

SEND_LONG = True
SEND_SHORT = True

STABLE = {
    "USDC", "TUSD", "FDUSD", "DAI",
    "USDD", "USDP", "PYUSD", "EURT"
}

# =====================================================
# GLOBAL VARIABLES
# =====================================================

lock = threading.Lock()

_next_request = [0.0]

cache = {}
last_alert = {}
seen = {}

stats = {
    "scans": 0,
    "alerts": 0,
    "errors": 0
}


# =====================================================
# HTTP REQUEST
# =====================================================

def get(url, params=None, tries=3):

    for k in range(tries):

        with lock:

            wait = _next_request[0] - time.time()

            if wait > 0:
                time.sleep(wait)

            _next_request[0] = time.time() + 0.06

        try:

            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            if response.status_code == 429:

                time.sleep(2 + k * 2)
                continue

            if response.status_code == 200:
                return response.json()

            print(
                "HTTP error:",
                response.status_code,
                url
            )

        except Exception as e:

            print("Request error:", e)
            time.sleep(1)

    return None


# =====================================================
# PRICE FORMAT
# =====================================================

def fp(x):

    x = float(x)
    a = abs(x)

    if a >= 100:
        return f"{x:.2f}"

    if a >= 1:
        return f"{x:.4f}"

    if a >= 0.01:
        return f"{x:.5f}"

    return f"{x:.8f}"


# =====================================================
# FETCH CANDLES
# =====================================================

def fetch(pair, tf, limit):

    result = get(
        f"{BASE}/spot/candlesticks",
        {
            "currency_pair": pair,
            "interval": tf,
            "limit": limit
        }
    )

    if not result or len(result) < 30:
        return None

    try:

        df = pd.DataFrame(
            [
                [
                    float(x[0]),
                    float(x[5]),
                    float(x[3]),
                    float(x[4]),
                    float(x[2]),
                    float(x[1])
                ]
                for x in result
            ],
            columns=["t", "o", "h", "l", "c", "v"]
        )

        df = df.sort_values("t")

        df = df.drop_duplicates("t")

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        # Remove currently forming candle
        return df.iloc[:-1].reset_index(drop=True)

    except Exception as e:

        print("Candle error:", pair, e)
        return None


# =====================================================
# INDICATORS
# =====================================================

def ema(series, n):

    return series.ewm(
        span=n,
        adjust=False
    ).mean()


def rsi(close, n=14):

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/n,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/n,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - 100 / (1 + rs)


def atr(df, n=14):

    previous_close = df.c.shift()

    tr = pd.concat(
        [
            df.h - df.l,
            (df.h - previous_close).abs(),
            (df.l - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1/n,
        adjust=False
    ).mean()


def adx(df, n=14):

    up = df.h.diff()

    down = -df.l.diff()

    plus_dm = up.where(
        (up > down) & (up > 0),
        0
    )

    minus_dm = down.where(
        (down > up) & (down > 0),
        0
    )

    a = atr(df, n)

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1/n,
            adjust=False
        ).mean()
        / a.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1/n,
            adjust=False
        ).mean()
        / a.replace(0, np.nan)
    )

    denominator = (plus_di + minus_di).replace(
        0,
        np.nan
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1/n,
        adjust=False
    ).mean()


# =====================================================
# SUPERTREND
# =====================================================

def supertrend(df, n=10, m=3.0):

    a = atr(df, n).values

    hl2 = (
        (df.h + df.l) / 2
    ).values

    close = df.c.values

    upper = hl2 + m * a
    lower = hl2 - m * a

    final_upper = upper.copy()
    final_lower = lower.copy()

    direction = np.ones(len(df))

    for i in range(1, len(df)):

        final_upper[i] = (
            upper[i]
            if (
                upper[i] < final_upper[i-1]
                or close[i-1] > final_upper[i-1]
            )
            else final_upper[i-1]
        )

        final_lower[i] = (
            lower[i]
            if (
                lower[i] > final_lower[i-1]
                or close[i-1] < final_lower[i-1]
            )
            else final_lower[i-1]
        )

        if (
            direction[i-1] == 1
            and close[i] < final_lower[i]
        ):

            direction[i] = -1

        elif (
            direction[i-1] == -1
            and close[i] > final_upper[i]
        ):

            direction[i] = 1

        else:

            direction[i] = direction[i-1]

    return direction


# =====================================================
# HIGHER TIMEFRAME TREND
# =====================================================

def htf_trend(df):

    if df is None or len(df) < 55:
        return 0

    close = df.c

    e21 = ema(close, 21).iloc[-1]
    e50 = ema(close, 50).iloc[-1]

    price = close.iloc[-1]

    if price > e21 > e50:
        return 1

    if price < e21 < e50:
        return -1

    return 0


def get_trends(pair):

    trends = {}

    now = time.time()

    for tf, ttl in HTF.items():

        key = (pair, tf)

        if (
            key not in cache
            or now - cache[key][0] > ttl
        ):

            df = fetch(pair, tf, 120)

            cache[key] = (
                now,
                htf_trend(df)
            )

        trends[tf] = cache[key][1]

    return trends


# =====================================================
# PIVOTS
# =====================================================

def pivots(df, k=3):

    high = df.h.values
    low = df.l.values

    highs = []
    lows = []

    for i in range(k, len(df) - k):

        if high[i] == high[
            i-k:i+k+1
        ].max():

            highs.append((i, high[i]))

        if low[i] == low[
            i-k:i+k+1
        ].min():

            lows.append((i, low[i]))

    def dedupe(points, is_high):

        out = []

        for point in points:

            if (
                out
                and point[0] - out[-1][0] <= k
            ):

                if (
                    (point[1] > out[-1][1])
                    == is_high
                ):

                    out[-1] = point

            else:

                out.append(point)

        return out

    return (
        dedupe(highs, True),
        dedupe(lows, False)
    )


# =====================================================
# PATTERN DETECTION
# =====================================================

def detect_patterns(df, tol=0.008):

    out = []

    n = len(df)

    highs, lows = pivots(df)

    close = df.c.iloc[-1]

    highs = [
        p for p in highs
        if p[0] > n - 120
    ]

    lows = [
        p for p in lows
        if p[0] > n - 120
    ]

    def near(a, b):

        return (
            abs(a-b) / max(a, b)
            <= tol
        )

    def between_min(i, j):

        return df.l.values[i:j+1].min()

    def between_max(i, j):

        return df.h.values[i:j+1].max()

    def add(name, side, neck, height):

        breakout = (
            close < neck
            if side == "SHORT"
            else close > neck
        )

        target = (
            neck - height
            if side == "SHORT"
            else neck + height
        )

        out.append({
            "name": name,
            "side": side,
            "breakout": bool(breakout),
            "target": target
        })

    # Double top

    if (
        len(highs) >= 2
        and near(highs[-1][1], highs[-2][1])
        and highs[-1][0] - highs[-2][0] >= 5
        and n - highs[-1][0] <= 40
    ):

        neck = between_min(
            highs[-2][0],
            highs[-1][0]
        )

        top = max(
            highs[-1][1],
            highs[-2][1]
        )

        if (top-neck)/top > 0.012:

            add(
                "Double Top",
                "SHORT",
                neck,
                top-neck
            )

    # Triple top

    if (
        len(highs) >= 3
        and near(highs[-1][1], highs[-2][1])
        and near(highs[-2][1], highs[-3][1])
        and n - highs[-1][0] <= 40
    ):

        neck = between_min(
            highs[-3][0],
            highs[-1][0]
        )

        top = max(
            p[1] for p in highs[-3:]
        )

        if (top-neck)/top > 0.012:

            add(
                "Triple Top",
                "SHORT",
                neck,
                top-neck
            )

    # Double bottom

    if (
        len(lows) >= 2
        and near(lows[-1][1], lows[-2][1])
        and lows[-1][0] - lows[-2][0] >= 5
        and n - lows[-1][0] <= 40
    ):

        neck = between_max(
            lows[-2][0],
            lows[-1][0]
        )

        bottom = min(
            lows[-1][1],
            lows[-2][1]
        )

        if (neck-bottom)/neck > 0.012:

            add(
                "Double Bottom",
                "LONG",
                neck,
                neck-bottom
            )

    # Triple bottom

    if (
        len(lows) >= 3
        and near(lows[-1][1], lows[-2][1])
        and near(lows[-2][1], lows[-3][1])
        and n - lows[-1][0] <= 40
    ):

        neck = between_max(
            lows[-3][0],
            lows[-1][0]
        )

        bottom = min(
            p[1] for p in lows[-3:]
        )

        if (neck-bottom)/neck > 0.012:

            add(
                "Triple Bottom",
                "LONG",
                neck,
                neck-bottom
            )

    # Head and shoulders

    if (
        len(highs) >= 3
        and n - highs[-1][0] <= 40
    ):

        a, b, c = [
            p[1] for p in highs[-3:]
        ]

        if (
            b > a * 1.01
            and b > c * 1.01
            and near(a, c)
        ):

            neck = (
                between_min(
                    highs[-3][0],
                    highs[-2][0]
                )
                +
                between_min(
                    highs[-2][0],
                    highs[-1][0]
                )
            ) / 2

            add(
                "Head & Shoulders",
                "SHORT",
                neck,
                b-neck
            )

    # Inverse head and shoulders

    if (
        len(lows) >= 3
        and n - lows[-1][0] <= 40
    ):

        a, b, c = [
            p[1] for p in lows[-3:]
        ]

        if (
            b < a * 0.99
            and b < c * 0.99
            and near(a, c)
        ):

            neck = (
                between_max(
                    lows[-3][0],
                    lows[-2][0]
                )
                +
                between_max(
                    lows[-2][0],
                    lows[-1][0]
                )
            ) / 2

            add(
                "Inverse H&S",
                "LONG",
                neck,
                neck-b
            )

    # Triangles

    if (
        len(highs) >= 3
        and len(lows) >= 3
        and n - highs[-1][0] <= 30
    ):

        h3 = [
            p[1] for p in highs[-3:]
        ]

        l3 = [
            p[1] for p in lows[-3:]
        ]

        if (
            near(max(h3), min(h3))
            and l3[0] < l3[1] < l3[2]
        ):

            add(
                "Ascending Triangle",
                "LONG",
                max(h3),
                max(h3)-l3[0]
            )

        if (
            near(max(l3), min(l3))
            and h3[0] > h3[1] > h3[2]
        ):

            add(
                "Descending Triangle",
                "SHORT",
                min(l3),
                h3[0]-min(l3)
            )

    return out


# =====================================================
# STRICT SIGNAL ANALYSIS
# =====================================================

def analyze(pair):

    df = fetch(pair, ALERT_TF, 250)

    if df is None or len(df) < 120:
        return None

    ts = df.t.iloc[-1]

    if seen.get(pair) == ts:
        return None

    seen[pair] = ts

    close = df.c

    price = close.iloc[-1]

    e9 = ema(close, 9)
    e26 = ema(close, 26)

    r = rsi(close)

    a_series = atr(df)
    a = a_series.iloc[-1]

    st = supertrend(df)

    R = r.iloc[-1]

    if (
        np.isnan(R)
        or np.isnan(a)
        or a <= 0
        or price <= 0
    ):

        return None

    # ADX

    adx_value = adx(df).iloc[-1]

    if (
        np.isnan(adx_value)
        or adx_value < MIN_ADX
    ):

        return None

    # Volume

    avg_vol = df.v.iloc[-21:-1].mean()

    current_vol = df.v.iloc[-1]

    if (
        pd.isna(avg_vol)
        or avg_vol <= 0
    ):

        return None

    vol_ok = (
        current_vol >= avg_vol * VOL_MULT
    )

    if not vol_ok:
        return None

    # Candle quality

    candle = df.iloc[-1]

    candle_range = candle.h - candle.l

    body = abs(
        candle.c - candle.o
    )

    if candle_range <= 0:
        return None

    if body < a * MIN_BODY_ATR:
        return None

    close_position = (
        candle.c - candle.l
    ) / candle_range

    long_candle_ok = (
        close_position >= CLOSE_ZONE
    )

    short_candle_ok = (
        close_position <= 1-CLOSE_ZONE
    )

    # MACD

    ml = ema(close, 12) - ema(close, 26)

    hist = ml - ema(ml, 9)

    macd_long = (
        hist.iloc[-1] > 0
        and hist.iloc[-1] > hist.iloc[-2]
    )

    macd_short = (
        hist.iloc[-1] < 0
        and hist.iloc[-1] < hist.iloc[-2]
    )

    # Five indicators

    L = {

        "EMA": (
            e9.iloc[-1] > e26.iloc[-1]
            and price > e9.iloc[-1]
        ),

        "Supertrend": st[-1] == 1,

        "RSI": (
            LONG_RSI_MIN
            <= R
            <= LONG_RSI_MAX
        ),

        "MACD": macd_long,

        "Volume": vol_ok
    }

    S = {

        "EMA": (
            e9.iloc[-1] < e26.iloc[-1]
            and price < e9.iloc[-1]
        ),

        "Supertrend": st[-1] == -1,

        "RSI": (
            SHORT_RSI_MIN
            <= R
            <= SHORT_RSI_MAX
        ),

        "MACD": macd_short,

        "Volume": vol_ok
    }

    long_score = sum(L.values())
    short_score = sum(S.values())

    side = None

    if (
        SEND_LONG
        and long_score >= MIN_CONFIRM
        and long_score > short_score
        and long_candle_ok
    ):

        side = "LONG"
        indicators = L
        score = long_score

    elif (
        SEND_SHORT
        and short_score >= MIN_CONFIRM
        and short_score > long_score
        and short_candle_ok
    ):

        side = "SHORT"
        indicators = S
        score = short_score

    if side is None:
        return None

    # Cooldown

    if (
        time.time()
        - last_alert.get((pair, side), 0)
        < COOLDOWN_MIN * 60
    ):

        return None

    # HTF trend

    trends = get_trends(pair)

    want = 1 if side == "LONG" else -1

    aligned = sum(
        1 for v in trends.values()
        if v == want
    )

    opposite = sum(
        1 for v in trends.values()
        if v == -want
    )

    if aligned < MIN_HTF_ALIGN:
        return None

    if opposite > MAX_HTF_OPPOSITE:
        return None

    # Extension

    extension = (
        abs(price-e26.iloc[-1]) / a
    )

    if extension > MAX_EXTENSION_ATR:
        return None

    # Patterns

    patterns = detect_patterns(df)

    for p in patterns:

        if (
            p["breakout"]
            and p["side"] != side
        ):

            return None

    # Stop loss

    swing = df.tail(10)

    if side == "LONG":

        sl = min(
            price - ATR_MULT*a,
            swing.l.min() - 0.2*a
        )

    else:

        sl = max(
            price + ATR_MULT*a,
            swing.h.max() + 0.2*a
        )

    risk = abs(price-sl)

    risk_pct = (
        risk / price * 100
    )

    # Reject wide or unusually tight SL

    if risk_pct > MAX_RISK_PCT:
        return None

    if risk_pct < MIN_RISK_PCT:
        return None

    # Targets

    sign = 1 if side == "LONG" else -1

    t1 = price + sign*T1_R*risk
    t2 = price + sign*T2_R*risk

    # Support / resistance

    highs, lows = pivots(df)

    resistances = [
        p[1] for p in highs
        if p[1] > price
    ]

    supports = [
        p[1] for p in lows
        if p[1] < price
    ]

    resistance = (
        min(resistances)
        if resistances else None
    )

    support = (
        max(supports)
        if supports else None
    )

    # Obstacle before T1

    if side == "LONG":

        if (
            resistance is not None
            and resistance < t1
        ):

            return None

    else:

        if (
            support is not None
            and support > t1
        ):

            return None

    return {

        "pair": pair,
        "side": side,

        "price": price,
        "sl": sl,

        "t1": t1,
        "t2": t2,

        "risk_pct": risk_pct,

        "ind": indicators,
        "score": score,

        "rsi": R,
        "adx": adx_value,

        "trends": trends,
        "pats": patterns,

        "df": df,

        "e9": e9,
        "e26": e26,
        "rsi_s": r,

        "res": resistance,
        "sup": support
    }


# =====================================================
# CHART GENERATOR
# ==========
