import time
import statistics
import requests

# ---------- SETTINGS ----------
TELEGRAM_TOKEN = "YAHAN_BOT_TOKEN"
CHAT_ID = "YAHAN_CHAT_ID"

INTERVAL = "5m"        # 1m, 5m, 15m, 1h (chhote timeframe me signals zyada noisy hote hain)
SCAN_EVERY = 60        # kitne second me ek baar scan
MAX_SCAN = 20          # ek baar me kitne coins check (API limit se bachne ke liye)
MIN_SCORE = 4          # futures me zyada strict: kitne signals ek saath match hon
COOLDOWN = 900         # ek coin ka alert dobara 15 min baad

# chote coins ka filter
MAX_PRICE = 1.0
MIN_VOLUME = 50000
MAX_VOLUME = 20000000
BIG_COINS = {"BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TRX","TON","AVAX","LINK",
             "DOT","MATIC","LTC","BCH","SHIB","UNI","ATOM","ETC","XLM","USDC","FDUSD","DAI","GT"}

# TP / SL (ATR ke hisaab se)
SL_ATR = 1.5
TP1_ATR = 1.5
TP2_ATR = 3.0
# ------------------------------

API = "https://api.gateio.ws/api/v4/futures/usdt"
TREND_INTERVAL = "1h"   # bada trend filter: signal trend ke saath hi ho
last_alert = {}


def send(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("Telegram error:", e)


# ---------- INDICATORS ----------
def ema(vals, n):
    k = 2 / (n + 1)
    e = vals[0]
    out = [e]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(closes, n=14):
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr(h, l, c, n=14):
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    return sum(trs[-n:]) / n


def macd_hist(closes):
    m = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    s = ema(m, 9)
    return [x - y for x, y in zip(m, s)]


def bollinger(closes, n=20):
    w = closes[-n:]
    mid = sum(w) / n
    sd = statistics.pstdev(w)
    return mid - 2 * sd, mid + 2 * sd


# ---------- CANDLE PATTERNS ----------
def candle_patterns(o, h, l, c):
    """+ = bullish, - = bearish. Returns list of (naam, score)."""
    found = []
    body = lambda i: abs(c[i] - o[i])
    rng = lambda i: (h[i] - l[i]) or 1e-12
    red = lambda i: c[i] < o[i]
    green = lambda i: c[i] > o[i]

    if red(-2) and green(-1) and o[-1] <= c[-2] and c[-1] >= o[-2]:
        found.append(("Bullish Engulfing", 1))
    if green(-2) and red(-1) and o[-1] >= c[-2] and c[-1] <= o[-2]:
        found.append(("Bearish Engulfing", -1))

    lw = min(o[-1], c[-1]) - l[-1]
    uw = h[-1] - max(o[-1], c[-1])
    if lw >= 2 * body(-1) and uw <= body(-1) and c[-3] > c[-1]:
        found.append(("Hammer", 1))
    if uw >= 2 * body(-1) and lw <= body(-1) and c[-3] < c[-1]:
        found.append(("Shooting Star", -1))

    if red(-3) and body(-2) < body(-3) * 0.4 and green(-1) and c[-1] > (o[-3] + c[-3]) / 2:
        found.append(("Morning Star", 1))
    if green(-3) and body(-2) < body(-3) * 0.4 and red(-1) and c[-1] < (o[-3] + c[-3]) / 2:
        found.append(("Evening Star", -1))

    if body(-1) <= 0.1 * rng(-1):
        found.append(("Doji (indecision)", 0))
    return found


# ---------- CHART PATTERNS (double top / bottom) ----------
def double_pattern(h, l, c, tol=0.006):
    lows = [i for i in range(2, len(l) - 2) if l[i] == min(l[i - 2:i + 3])]
    highs = [i for i in range(2, len(h) - 2) if h[i] == max(h[i - 2:i + 3])]
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if b - a >= 5 and abs(l[a] - l[b]) / l[a] < tol:
            neck = max(h[a:b + 1])
            if c[-1] > neck:
                return ("Double Bottom breakout", 1)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if b - a >= 5 and abs(h[a] - h[b]) / h[a] < tol:
            neck = min(l[a:b + 1])
            if c[-1] < neck:
                return ("Double Top breakdown", -1)
    return None


# ---------- ANALYSIS ----------
def get_candles(pair, interval=None):
    r = requests.get(f"{API}/candlesticks",
                     params={"contract": pair, "interval": interval or INTERVAL, "limit": 120}, timeout=10)
    rows = r.json()[:-1]                 # chalu (adhuri) candle hata do
    c = [float(x["c"]) for x in rows]
    h = [float(x["h"]) for x in rows]
    l = [float(x["l"]) for x in rows]
    o = [float(x["o"]) for x in rows]
    v = [float(x.get("sum", x["v"])) for x in rows]
    return o, h, l, c, v


def analyze(pair):
    o, h, l, c, v = get_candles(pair)
    if len(c) < 60:
        return None
    score = 0
    reasons = []

    r = rsi(c)
    if r < 30:
        score += 1; reasons.append(f"RSI oversold ({r:.0f})")
    elif r > 70:
        score -= 1; reasons.append(f"RSI overbought ({r:.0f})")

    e9, e21 = ema(c, 9), ema(c, 21)
    if e9[-1] > e21[-1] and e9[-3] <= e21[-3]:
        score += 1; reasons.append("EMA 9/21 bullish cross")
    elif e9[-1] < e21[-1] and e9[-3] >= e21[-3]:
        score -= 1; reasons.append("EMA 9/21 bearish cross")

    mh = macd_hist(c)
    if mh[-1] > 0 >= mh[-2]:
        score += 1; reasons.append("MACD bullish")
    elif mh[-1] < 0 <= mh[-2]:
        score -= 1; reasons.append("MACD bearish")

    lo, hi = bollinger(c)
    if c[-1] < lo:
        score += 1; reasons.append("Bollinger lower band ke neeche")
    elif c[-1] > hi:
        score -= 1; reasons.append("Bollinger upper band ke upar")

    for name, s in candle_patterns(o, h, l, c):
        score += s
        reasons.append(name)

    dp = double_pattern(h, l, c)
    if dp:
        score += dp[1] * 2
        reasons.append(dp[0])

    avg_v = sum(v[-21:-1]) / 20
    if avg_v > 0 and v[-1] > 3 * avg_v:
        reasons.append("Volume spike (3x)")

    # bade timeframe ka trend filter (galat direction ke signals hata do)
    _, _, _, tc, _ = get_candles(pair, TREND_INTERVAL)
    if len(tc) < 55:
        return None
    trend_up = tc[-1] > ema(tc, 50)[-1]
    if (score > 0 and not trend_up) or (score < 0 and trend_up):
        return None
    reasons.append(f"{TREND_INTERVAL} trend {'UP' if trend_up else 'DOWN'} ke saath")

    a = atr(h, l, c)
    return score, reasons, c[-1], a


def format_alert(pair, score, reasons, price, a):
    if score > 0:
        side = "LONG 🟢"
        sl, tp1, tp2 = price - SL_ATR * a, price + TP1_ATR * a, price + TP2_ATR * a
    else:
        side = "SHORT 🔴"
        sl, tp1, tp2 = price + SL_ATR * a, price - TP1_ATR * a, price - TP2_ATR * a
    return (f"📊 {pair} [{INTERVAL}]\n"
            f"Signal: {side}\n"
            f"Entry: {price:.8g}\n"
            f"SL: {sl:.8g}\n"
            f"TP1: {tp1:.8g}\n"
            f"TP2: {tp2:.8g}\n"
            f"Score: {score}\n"
            f"Wajah: " + ", ".join(reasons) +
            "\nLeverage: 3x-5x se zyada mat lo, SL zaroor lagao."
            "\n⚠️ Ye sirf analysis hai, guarantee nahi.")


def candidates():
    data = requests.get(f"{API}/tickers", timeout=10).json()
    out = []
    for t in data:
        pair = t["contract"]
        if not pair.endswith("_USDT") or pair.split("_")[0] in BIG_COINS:
            continue
        try:
            price = float(t["last"])
            vol = float(t.get("volume_24h_quote") or t.get("volume_24h_settle") or 0)
            chg = abs(float(t["change_percentage"]))
        except (TypeError, ValueError, KeyError):
            continue
        if price <= 0 or price > MAX_PRICE or not (MIN_VOLUME <= vol <= MAX_VOLUME):
            continue
        out.append((chg, pair))
    out.sort(reverse=True)              # sabse active coins pehle
    return [p for _, p in out[:MAX_SCAN]]


def run():
    print("Signal bot start...")
    while True:
        try:
            for pair in candidates():
                if time.time() - last_alert.get(pair, 0) < COOLDOWN:
                    continue
                res = analyze(pair)
                time.sleep(0.3)          # API par load kam rakhne ke liye
                if not res:
                    continue
                score, reasons, price, a = res
                if abs(score) >= MIN_SCORE:
                    send(format_alert(pair, score, reasons, price, a))
                    last_alert[pair] = time.time()
        except Exception as e:
            print("Error:", e)
        time.sleep(SCAN_EVERY)


run()
