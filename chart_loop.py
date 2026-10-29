"""
Chart loop: chart_scanner.py ko lagataar chalata hai (default har 3 minute) jab tak job chalti hai,
taaki alert late na ho. GitHub job ek baar me ~6 ghante chalti hai, isliye workflow har 5 ghante
naya loop shuru karta hai aur purana khud band ho jata hai (kabhi gap nahi).

Ye chart_scanner ko import nahi karta, har round me `python chart_scanner.py` alag process me
chalata hai. Isliye scanner me main() ya koi extra function ki zarurat nahi.
Har round ke baad cooldown state file badli ho to repo me save (git push) hoti hai.

Safety: lagataar MAX_FAILS baar scan fail hua to job khud ruk jayegi (minutes waste nahi honge).

ENV (optional): RUN_MINUTES (340), INTERVAL (180 sec), MAX_FAILS (5), SCAN_TIMEOUT (600 sec),
                STATE_FILE (alert_state_chart.json) + chart_scanner ke saare ENV
"""
import os
import subprocess
import sys
import time

RUN_MIN = float(os.getenv("RUN_MINUTES", "340"))
INTERVAL = int(os.getenv("INTERVAL", "180"))
MAX_FAILS = int(os.getenv("MAX_FAILS", "5"))
SCAN_TIMEOUT = int(os.getenv("SCAN_TIMEOUT", "600"))
STATE_FILE = os.getenv("STATE_FILE", "alert_state_chart.json")


def git(*a):
    return subprocess.run(["git", *a], check=False, capture_output=True, text=True)


def push_state():
    """State file badli ho tabhi commit + push (alert gaya ho to hi badlegi)."""
    try:
        if not os.path.exists(STATE_FILE):
            return
        git("config", "user.name", "github-actions")
        git("config", "user.email", "actions@github.com")
        git("add", STATE_FILE)
        if git("diff", "--cached", "--quiet").returncode != 0:
            git("commit", "-m", "update alert state")
            r = git("push")
            if r.returncode != 0:  # conflict ho to ek baar rebase karke retry
                git("pull", "--rebase", "--autostash")
                r = git("push")
            print("state push:", r.returncode, flush=True)
    except Exception as e:
        print("state push fail:", repr(e)[:150], flush=True)


def run_scan():
    """Ek baar poora scan. Kamyab to True."""
    try:
        r = subprocess.run([sys.executable, "chart_scanner.py"], timeout=SCAN_TIMEOUT)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print("scan timeout", flush=True)
        return False


def main():
    end = time.time() + RUN_MIN * 60
    n = 0
    fails = 0
    while time.time() < end:
        t0 = time.time()
        n += 1
        print(f"=== scan #{n} ===", flush=True)
        try:
            ok = run_scan()
        except Exception as e:
            ok = False
            print("scan error:", repr(e)[:200], flush=True)
        if ok:
            fails = 0
            push_state()
        else:
            fails += 1
            print(f"scan fail ({fails}/{MAX_FAILS})", flush=True)
            if fails >= MAX_FAILS:
                print("Bahut baar fail hua, job band kar raha hoon.", flush=True)
                sys.exit(1)
        sleep_for = max(30, INTERVAL - (time.time() - t0))
        if time.time() + sleep_for >= end:
            break
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
