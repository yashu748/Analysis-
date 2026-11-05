    return [p for _, p in out[:MAX_COINS]]

def scan():
    pairs = coin_list(); sent = 0
    stats.clear()
    def work(p):
        try: return analyze(p)
        except Exception: traceback.print_exc(); return None
    with ThreadPoolExecutor(6) as ex:
        res = [s for s in ex.map(work, pairs) if s]
    # best quality pehle: pattern breakout wale, phir bade volume wale
    res.sort(key=lambda s: s["q"], reverse=True)
    now = time.time()
    day_alerts[:] = [t for t in day_alerts if now - t < 24 * 3600]
    room = max(0, MAX_ALERTS_PER_DAY - len(day_alerts))
    if res and room == 0: stats["daily limit poori (6/24h)"] = len(res)
    for s in res[:min(MAX_ALERTS_PER_SCAN, room)]:
        last_alert[(s["pair"], s["side"])] = now; day_alerts.append(now); send(s); sent += 1
        if TRACK_RESULTS:
            open_trades.append(dict(pair=s["pair"], side=s["side"], entry=s["price"], stop=s["sl"],
                                    t1=s["t1"], t2=s["t2"], since=float(s["df"].t.iloc[-1]) + 900, t1_hit=False))
    if TRACK_RESULTS and open_trades:
        try: track_results()
        except Exception: traceback.print_exc()
    blocked = ", ".join(f"{k}: {v}" for k, v in sorted(stats.items(), key=lambda x: -x[1]))
    print(time.strftime("%H:%M:%S"), f"scanned {len(pairs)} coins, alerts {sent} (24h: {len(day_alerts)}/{MAX_ALERTS_PER_DAY}), open {len(open_trades)}"
          + (f" | roke gaye -> {blocked}" if blocked else ""), flush=True)

def main():
    end = time.time() + RUN_MINUTES * 60
    print(f"Bot start (v2 filters, mode={MODE}): alerts on {ALERT_TF}, trend filter {list(HTF)}, runs {RUN_MINUTES:.0f} min", flush=True)
    while time.time() < end:
        t0 = time.time()
        try: scan()
        except Exception: traceback.print_exc()
        time.sleep(max(5, SCAN_EVERY - (time.time() - t0)))

if __name__ == "__main__":
    main()
  
