"""
Chart loop: chart_scanner ko lagataar chalata hai (default har 3 minute) jab tak job chalti hai,
taaki alert late na ho. GitHub job ek baar me ~6 ghante chalti hai, isliye workflow har 5 ghante
naya loop shuru karta hai aur purana khud band ho jata hai (kabhi gap nahi).
Har alert ke baad cooldown state repo me save (git push) hota hai.

ENV (optional): RUN_MINUTES (340), INTERVAL (180 sec), + chart_scanner ke saare ENV
"""
import os
import subprocess
import time

import chart_scanner as cs

RUN_MIN = float(os.getenv("RUN_MINUTES", "340"))
INTERVAL = int(os.getenv("INTERVAL", "180"))


def push_state():
    try:
        run = lambda *a: subprocess.run(a, check=False, capture_output=True, text=True)
        run("git", "config", "user.name", "github-actions")
        run("git", "config", "user.email", "actions@github.com")
        run("git", "add", cs.STATE_FILE)
        if run("git", "diff", "--cached", "--quiet").returncode != 0:
            run("git", "commit", "-m", "update alert state")
            r = run("git", "push")
            print("state push:", r.returncode)
    except Exception as e:
        print("state push fail:", repr(e)[:150])


def main():
    end = time.time() + RUN_MIN * 60
    n = 0
    while time.time() < end:
        t0 = time.time()
        n += 1
        print(f"=== scan #{n} ===")
        try:
            sent = cs.main()
            if sent:
                push_state()
        except Exception as e:
            print("scan error:", repr(e)[:200])
        sleep_for = max(30, INTERVAL - (time.time() - t0))
        if time.time() + sleep_for >= end:
            break
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
  
