#!/usr/bin/env python3
"""
Watchdog script to monitor the Elenchus benchmark process.
Checks for log activity and process liveness.
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

BENCHMARK_LOG = Path("benchmark.log")
WATCHDOG_LOG = Path("watchdog.log")
STALL_THRESHOLD_SEC = 600  # 10 minutes (DeepSeek R1 is slow)

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")

def find_benchmark_pid():
    try:
        # Find process running benchmark_probe.py
        cmd = ["pgrep", "-f", "benchmark_probe.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().splitlines()
            # Return the first one that isn't this watchdog itself (if invoked weirdly)
            return int(pids[0])
    except Exception:
        pass
    return None

def main():
    log("🚀 Watchdog started. Monitoring benchmark_probe.py...")
    
    pid = find_benchmark_pid()
    if not pid:
        log("❌ No benchmark process found! Exiting.")
        sys.exit(1)
        
    log(f"✅ Found benchmark process: PID {pid}")
    
    while True:
        # 1. Check if process is still running
        try:
            os.kill(pid, 0)
        except OSError:
            log("⚠️  Benchmark process died! It is no longer running.")
            sys.exit(1)
            
        # 2. Check log freshness
        if BENCHMARK_LOG.exists():
            mtime = BENCHMARK_LOG.stat().st_mtime
            age = time.time() - mtime
            
            if age > STALL_THRESHOLD_SEC:
                log(f"🚨 STALL DETECTED: No log output for {int(age)} seconds!")
                # Optional: trigger notification or restart logic here
            elif age > 60:
                # Heartbeat for slow but alive
                pass 
        else:
            log("⚠️  benchmark.log not found yet.")
            
        time.sleep(60)

if __name__ == "__main__":
    main()
