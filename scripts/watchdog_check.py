#!/usr/bin/env python3
"""Check Luma's systemd integration without systemd (SPEC-friochap CA-5).

Plays systemd's part: owns the NOTIFY_SOCKET, sets WATCHDOG_USEC, starts
`luma serve`, and checks that
  1. READY=1 arrives only once the HTTP listener answers;
  2. WATCHDOG=1 keeps arriving while Luma serves;
  3. with Luma frozen (SIGSTOP) the heartbeat stops, so systemd, which kills
     a unit after WatchdogSec without one, would restart it in time;
  4. STOPPING=1 arrives on SIGTERM and the process exits within 15 s.

Usage: watchdog_check.py <path to luma binary>
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

LUMA = sys.argv[1]
WATCHDOG_SEC = 2.0
work = tempfile.mkdtemp()
sock_path = os.path.join(work, "notify.sock")
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(sock_path)

env = dict(
    os.environ,
    NOTIFY_SOCKET=sock_path,
    WATCHDOG_USEC=str(int(WATCHDOG_SEC * 1_000_000)),
    DATA_DIR=os.path.join(work, "data"),
    SQLITE_DB_PATH=os.path.join(work, "data", "rustkiss.db"),
    LUMA_API_KEY="watchdog-check-key-0123456789abcdef",
    LUMA_MASTER_KEY="watchdog-check-master-0123456789abcdef",
    PORT_LUMA_VDB="7711",
    BIND_ADDR="127.0.0.1",
)
proc = subprocess.Popen([LUMA, "serve"], env=env, cwd=work,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def messages(seconds):
    """Every notification received within `seconds`."""
    got = []
    deadline = time.monotonic() + seconds
    while (left := deadline - time.monotonic()) > 0:
        sock.settimeout(left)
        try:
            got.append(sock.recv(4096).decode())
        except socket.timeout:
            break
    return got


def answers():
    try:
        return urllib.request.urlopen("http://127.0.0.1:7711/v1/health", timeout=1).status == 200
    except Exception:
        return False


failures = []
try:
    first = []
    deadline = time.monotonic() + 60
    while "READY=1" not in first and time.monotonic() < deadline:
        first += messages(0.5)
    if "READY=1" not in first:
        failures.append("no READY=1 within 60 s")
    elif not answers():
        failures.append("READY=1 sent but /v1/health does not answer")

    beats = [m for m in messages(3 * WATCHDOG_SEC) if m == "WATCHDOG=1"]
    if len(beats) < 3:
        failures.append(f"only {len(beats)} heartbeats in {3 * WATCHDOG_SEC} s while serving")

    proc.send_signal(signal.SIGSTOP)
    messages(WATCHDOG_SEC)  # anything already in flight
    frozen = [m for m in messages(2 * WATCHDOG_SEC) if m == "WATCHDOG=1"]
    if frozen:
        failures.append(f"{len(frozen)} heartbeats while frozen: systemd would not restart it")
    proc.send_signal(signal.SIGCONT)
    again = [m for m in messages(3 * WATCHDOG_SEC) if m == "WATCHDOG=1"]
    if not again:
        failures.append("heartbeat did not resume after SIGCONT")

    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    stopping = messages(1)
    if "STOPPING=1" not in stopping:
        failures.append("no STOPPING=1 on SIGTERM")
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        failures.append("did not exit within 15 s of SIGTERM")
    print(f"ready, {len(beats)} beats serving, 0 expected frozen ({len(frozen)} seen), "
          f"{len(again)} after resume, exit after {time.monotonic() - started:.1f} s")
finally:
    if proc.poll() is None:
        proc.kill()

for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
