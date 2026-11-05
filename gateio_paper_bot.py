
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==========================================
# GATE.IO LONG + SHORT PAPER SIGNAL BOT
# SIMULATION ONLY - NO REAL ORDERS
# ==========================================

BASE_URL = "https://api.gateio.ws/api/v4"

# Telegram details (optional)
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# Settings
TIMEFRAME = "15m"
SCAN_SECONDS = 120
MAX_COINS = 30
MIN_VOLUME_USDT = 1_000_000
MIN_CONFIRM = 5

# Backtest settings
BACKTEST_DAYS = 7
FEE_PERCENT = 0.10       # assumed fee per side
SLIPPAGE_PERCENT = 0.05  # assumed slippage per side

# Risk simulation settings
ATR_MULT = 1.8
MAX_RISK_PERCENT = 3.0
T1_R = 1.5
T2_R = 2.5

SEND_LONG = True
SEND_SHORT = True

STABLECOINS = {
    "USDC", "TUSD", "FDUSD", "DAI",
    "USDD", "USDP", "PYUSD", "EURT"
}

session = requests.Session()
session.headers.update({"Accept": "application/json"})

last_alert = {}


# ==========================================
# TELEGRAM
# ==========================================

def telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(message)
        return

    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(
            url,
            data={
                "chat_id": TG_CHAT_ID,
                "text": message
            },
            timeout=15
        )
    except Exception as e:
        print("Telegram error:", e)


# ==========================================
# GATE.IO DATA
# ==========================================

def get_candles(pair, interval="15m", limit=500):
    url = BASE_URL + "/spot/candlesticks"

    try:
        response = session.get(
            url,
            params={
                "currency_pair": pair,
                "interval": interval,
                "limit": limit
            },
            timeout=20
        )
        response.raise_for_status()
        raw = response.json()

        if not raw:
            return None

        # Gate.io candle:
        # timestamp, quote_volume, close, high, low, open

        rows = []

        for x in raw:
            rows.append({
                "time": int(x[0]),
                "open": float(x[5]),
                "high": float(x[3]),
                "low": float(x[4]),
                "close": float(x[2]),
                "volume": float(x[1])
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("time").drop_duplicates("time")
        df = df.reset_index(drop=True)

        # Exclude the currently forming candle
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        return df

    except Exception as e:
        print("Candle error:", pair, e)
        return None


def get_pairs():
    try:
        url = BASE_URL + "/spot/tickers"
        r = session.get(url, timeout=20)
        r.raise_for_status()

        data = r.json()
        selected = []

        for item in data:
            pair = item.get("currency_pair", "")

            if not pair.endswith("_USDT"):
                continue

            base = pair.split("_")[0]

            if base in STABLECOINS:
                continue

            volume = float(item.get("quote_volume", 0))

            if volume >= MIN_VOLUME_USDT:
                selected.append((pair, volume))

        selected.sort(key=lambda x: x[1], reverse=True)

        return [x[0] for x in selected[:MAX_COINS]]

    except Exception as e:
        print("Pair error:", e)
        return []


# ==========================================
# INDICATORS
# ==========================================

def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    df["macd_hist"] = macd - signal

    # ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move, 0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move, 0
    )

    atr_safe = df["atr"].replace(0, np.nan)

    plus_di = 100 * pd.Series(
        plus_dm, index=df.index
    ).ewm(alpha=1/14, adjust=False).mean() / atr_safe

    minus_di = 100 * pd.Series(
        minus_dm, index=df.index
    ).ewm(alpha=1/14, adjust=False).mean() / atr_safe

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["adx"] = dx.ewm(
        alpha=1/14, adjust=False
    ).mean().fillna(0)

    # Supertrend
    hl2 = (high + low) / 2
    upper = hl2 + 3 * df["atr"]
    lower_band = hl2 - 3 * df["atr"]

    st_direction = []
    direction = 1

    for i in range(len(df)):
        if i > 0:
            if close.iloc[i] > upper.iloc[i - 1]:
                direction = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction = -1

        st_direction.append(direction)

    df["supertrend_dir"] = st_direction

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ==========================================
# HIGHER TIMEFRAME TREND
# ==========================================

def htf_direction(pair, interval):
    df = get_candles(pair, interval, 150)

    if df is None or len(df) < 60:
        return 0

    close = df["close"]

    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    if ema21 > ema50:
        return 1

    if ema21 < ema50:
        return -1

    return 0


def get_htf_score(pair):
    directions = [
        htf_direction(pair, "1h"),
        htf_direction(pair, "4h"),
        htf_direction(pair, "1d")
    ]

    return directions


# ==========================================
# SIGNAL GENERATOR
# ==========================================

def make_signal(df, htf):
    if df is None or len(df) < 100:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(row["close"])
    atr = float(row["atr"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    long_score = 0
    short_score = 0

    # EMA
    if row["ema9"] > row["ema26"]:
        long_score += 1
    if row["ema9"] < row["ema26"]:
        short_score += 1

    # Supertrend
    if row["supertrend_dir"] == 1:
        long_score += 1
    if row["supertrend_dir"] == -1:
        short_score += 1

    # RSI
    if 50 < row["rsi"] <= 62:
        long_score += 1
    if 38 <= row["rsi"] < 50:
        short_score += 1

    # MACD
    if row["macd_hist"] > 0:
        long_score += 1
    if row["macd_hist"] < 0:
        short_score += 1

    # Volume
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    if avg_volume > 0 and row["volume"] >= avg_volume * 1.5:
        if row["close"] > row["open"]:
            long_score += 1
        elif row["close"] < row["open"]:
            short_score += 1

    # Candle confirmation
    candle_body = abs(row["close"] - row["open"])

    if candle_body < atr * 0.25:
        return None

    # ADX strength
    if row["adx"] < 20:
        return None

    # HTF confirmation
    long_aligned = sum(x == 1 for x in htf)
    short_aligned = sum(x == -1 for x in htf)

    long_opposite = sum(x == -1 for x in htf)
    short_opposite = sum(x == 1 for x in htf)

    side = None

    if (
        SEND_LONG
        and long_score >= MIN_CONFIRM
        and long_aligned >= 2
        and long_opposite == 0
        and 50 < row["rsi"] <= 62
    ):
        side = "LONG"

    elif (
        SEND_SHORT
        and short_score >= MIN_CONFIRM
        and short_aligned >= 2
        and short_opposite == 0
        and 38 <= row["rsi"] < 50
    ):
        side = "SHORT"

    if side is None:
        return None

    if side == "LONG":
        sl = price - ATR_MULT * atr
        risk = price - sl

        if risk / price * 100 > MAX_RISK_PERCENT:
            return None

        t1 = price + risk * T1_R
        t2 = price + risk * T2_R

    else:
        sl = price + ATR_MULT * atr
        risk = sl - price

        if risk / price * 100 > MAX_RISK_PERCENT:
            return None

        t1 = price - risk * T1_R
        t2 = price - risk * T2_R

    return {
        "side": side,
        "entry": price,
        "sl": sl,
        "t1": t1,
        "t2": t2,
        "score": long_score if side == "LONG" else short_score,
        "rsi": float(row["rsi"]),
        "adx": float(row["adx"]),
        "time": int(row["time"])
    }


# ==========================================
# LIVE PAPER ALERT SCAN
# ==========================================

def scan_market():
    pairs = get_pairs()
    alerts = 0

    print(
        "\nScanning", len(pairs),
        "pairs at", datetime.now().strftime("%H:%M:%S")
    )

    for pair in pairs:
        if alerts >= 2:
            break

        try:
            df = get_candles(pair, TIMEFRAME, 250)

            if df is None or len(df) < 100:
                continue

            df = add_indicators(df)

            htf = get_htf_score(pair)
            signal = make_signal(df, htf)

            if signal is None:
                continue

            key = (pair, signal["side"])

            if last_alert.get(key) == signal["time"]:
                continue

            last_alert[key] = signal["time"]

            message = (
                "📊 GATE.IO PAPER SIGNAL\n\n"
                f"Pair: {pair}\n"
                f"Side: {signal['side']}\n"
                f"Entry: {signal['entry']:.8g}\n"
                f"Stop Loss: {signal['sl']:.8g}\n"
                f"Target 1: {signal['t1']:.8g}\n"
                f"Target 2: {signal['t2']:.8g}\n\n"
                f"Score: {signal['score']}/5\n"
                f"RSI: {signal['rsi']:.2f}\n"
                f"ADX: {signal['adx']:.2f}\n\n"
                "SIMULATION ONLY - NO REAL ORDER"
            )

            telegram(message)
            alerts += 1

        except Exception as e:
            print("Scan error:", pair, e)

    print("Scan complete. Alerts:", alerts)


# ==========================================
# 7-DAY HISTORICAL PAPER BACKTEST
# ==========================================

def backtest_pair(pair):
    # 7 days = 672 candles of 15 minutes
    # Extra candles are requested for indicator warm-up.
    df = get_candles(pair, "15m", 1000)

    if df is None or len(df) < 200:
        return []

    df = add_indicators(df)

    cutoff = int(time.time()) - BACKTEST_DAYS * 86400
    start_index = next(
        (i for i, t in enumerate(df["time"]) if t >= cutoff),
        None
    )

    if start_index is None:
        return []

    results = []
    i = max(start_index, 100)

    while i < len(df) - 2:
        # Use historical HTF data available up to that candle.
        # Resample past 15m candles to avoid future-data leakage.
        history = df.iloc[:i + 1].copy()
        history = history.set_index(
            pd.to_datetime(history["time"], unit="s", utc=True)
        )

        htf_dirs = []

        for rule in ["1h", "4h", "1D"]:
            htf_df = history.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            if len(htf_df) < 55:
                htf_dirs.append(0)
                continue

            e21 = htf_df["close"].ewm(
                span=21, adjust=False
            ).mean().iloc[-1]

            e50 = htf_df["close"].ewm(
                span=50, adjust=False
            ).mean().iloc[-1]

            htf_dirs.append(
                1 if e21 > e50 else -1 if e21 < e50 else 0
            )

        # Generate signal from data available at candle close.
        signal = make_signal(
            df.iloc[:i + 1].reset_index(drop=True),
            htf_dirs
        )

        if signal is None:
            i += 1
            continue

        # Entry at next candle open, not signal candle close.
        entry = float(df.iloc[i + 1]["open"])

        # Adjust levels by the difference between signal close and entry.
        shift = entry - signal["entry"]

        if signal["side"] == "LONG":
            sl = signal["sl"] + shift
            t1 = signal["t1"] + shift
            t2 = signal["t2"] + shift
        else:
            sl = signal["sl"] + shift
            t1 = signal["t1"] + shift
            t2 = signal["t2"] + shift

        outcome = "OPEN"
        exit_price = entry
        exit_index = len(df) - 1

        for j in range(i + 1, len(df)):
            candle = df.iloc[j]

            if signal["side"] == "LONG":
                hit_sl = candle["low"] <= sl
                hit_t2 = candle["high"] >= t2
                hit_t1 = candle["high"] >= t1
            else:
                hit_sl = candle["high"] >= sl
                hit_t2 = candle["low"] <= t2
                hit_t1 = candle["low"] <= t1

            # Conservative assumption: if SL and target
            # occur in the same candle, count SL first.
            if hit_sl:
                outcome = "SL"
                exit_price = sl
                exit_index = j
                break

            if hit_t2:
                outcome = "T2"
                exit_price = t2
                exit_index = j
                break

            if hit_t1:
                outcome = "T1"
                exit_price = t1
                exit_index = j
                break

        if outcome == "OPEN":
            exit_price = float(df.iloc[-1]["close"])

        if signal["side"] == "LONG":
            gross_pct = (exit_price / entry - 1) * 100
        else:
            gross_pct = (entry / exit_price - 1) * 100

        total_cost = 2 * (FEE_PERCENT + SLIPPAGE_PERCENT)
        net_pct = gross_pct - total_cost

        results.append({
            "pair": pair,
            "side": signal["side"],
            "entry": entry,
            "exit": exit_price,
            "result": outcome,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
            "entry_time": int(df.iloc[i + 1]["time"])
        })

        # One position at a time per pair.
        i = exit_index + 1

    return results


def run_backtest():
    print("\nStarting 7-day paper backtest...")

    pairs = get_pairs()
    all_results = []

    for number, pair in enumerate(pairs, 1):
        print(f"Backtesting {number}/{len(pairs)}: {pair}")

        try:
            trades = backtest_pair(pair)
            all_results.extend(trades)
        except Exception as e:
            print("Backtest error:", pair, e)

        time.sleep(0.15)

    if not all_results:
        print("No historical signals found.")
        return

    results = pd.DataFrame(all_results)

    wins = results[results["net_pct"] > 0]
    losses = results[results["net_pct"] <= 0]

    total = len(results)
    win_rate = len(wins) / total * 100

    gross_profit = wins["net_pct"].sum()
    gross_loss = abs(losses["net_pct"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else float("inf")
    )

    # Approximate sequential equity curve.
    equity = 100.0
    peak = equity
    max_drawdown = 0.0

    for value in results["net_pct"]:
        equity *= (1 + value / 100)
        peak = max(peak, equity)

        if peak > 0:
            dd = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, dd)

    print("\n========== BACKTEST REPORT ==========")
    print("Period:", BACKTEST_DAYS, "days")
    print("Total trades:", total)
    print("LONG trades:", len(results[results.side == "LONG"]))
    print("SHORT trades:", len(results[results.side == "SHORT"]))
    print("T1 hits:", len(results[results.result == "T1"]))
    print("T2 hits:", len(results[results.result == "T2"]))
    print("SL hits:", len(results[results.result == "SL"]))
    print("Open at end:", len(results[results.result == "OPEN"]))
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Approx. max drawdown: {max_drawdown:.2f}%")
    print(f"Net compounded result: {equity - 100:.2f}%")
    print("=====================================")

    results.to_csv("gateio_backtest_results.csv", index=False)
    print("Saved: gateio_backtest_results.csv")


# ==========================================
# MAIN MENU
# ==========================================

def main():
    print("\nGATE.IO PAPER BOT")
    print("1 = Start paper alerts")
    print("2 = Run 7-day backtest")
    print("3 = Exit")

    choice = input("Select option: ").strip()

    if choice == "1":
        print("Paper alert bot started.")

        while True:
            try:
                scan_market()
                time.sleep(SCAN_SECONDS)
            except KeyboardInterrupt:
                print("Bot stopped.")
                break
            except Exception as e:
                print("Main loop error:", e)
                time.sleep(10)

    elif choice == "2":
        run_backtest()

    else:
        print("Exited.")


if __name__ == "__main__":
    main()
