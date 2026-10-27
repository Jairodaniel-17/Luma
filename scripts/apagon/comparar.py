#!/usr/bin/env python3
"""Same workload, same power cut, several engines (see comparar.sh).

`load` writes key = value from several threads until the engine goes away,
journalling `ok` (fsynced) only after the engine confirmed. `verify` counts
how many confirmed writes are missing after the cut. One client per engine:
RESP for Luma and Redis, psql for PostgreSQL. Standard library only.
"""

import os
import socket
import subprocess
import sys
import threading
import time

ENGINE = sys.argv[2]
RESP_PORT = {"luma": 6390, "redis-always": 6380, "redis-everysec": 6380}
AUTH = os.environ.get("LUMA_API_KEY") if ENGINE == "luma" else None


class Resp:
    def __init__(self):
        self.s = socket.create_connection(("127.0.0.1", RESP_PORT[ENGINE]), timeout=10)
        self.f = self.s.makefile("rb")
        if AUTH:
            self.cmd("AUTH", AUTH)

    def cmd(self, *args):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)
        line = self.f.readline()
        if not line:
            raise ConnectionError("closed")
        if line[:1] == b"$":
            n = int(line[1:])
            if n < 0:
                return None
            data = self.f.read(n + 2)[:-2]
            return data.decode()
        if line[:1] == b"-":
            raise RuntimeError(line.decode().strip())
        return line[1:].decode().strip()


def ready(c):
    """Redis accepts connections while it replays its AOF and answers
    -LOADING until done; Luma only listens once replayed."""
    for _ in range(600):
        try:
            if c.cmd("PING") == "PONG":
                return c
        except RuntimeError:
            time.sleep(0.1)
    raise RuntimeError("engine still loading after 60 s")


class Pg:
    """One psql in autocommit; `\\echo` after each INSERT prints only once the
    INSERT has committed, which is the confirmation."""

    def __init__(self):
        self.p = subprocess.Popen(
            ["psql", "-h", "/tmp", "-p", "5433", "-U", "postgres", "-qAt", "-v", "ON_ERROR_STOP=1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def put(self, key):
        self.p.stdin.write(f"INSERT INTO kv VALUES ('{key}') ON CONFLICT DO NOTHING;\n\\echo ok\n")
        self.p.stdin.flush()
        if self.p.stdout.readline().strip() != "ok":
            raise ConnectionError("closed")


def worker(n, cycle, journal, lock):
    try:
        client = Pg() if ENGINE.startswith("pg") else ready(Resp())
    except (OSError, RuntimeError):
        return
    i = 0
    while True:
        i += 1
        key = f"c{cycle}w{n}k{i}"
        try:
            if isinstance(client, Pg):
                client.put(key)
            else:
                client.cmd("SET", key, key)
        except (OSError, ConnectionError, RuntimeError, BrokenPipeError):
            return
        with lock:
            journal.write(key + "\n")
            journal.flush()
            os.fsync(journal.fileno())


def load(path, cycle, threads=6):
    lock = threading.Lock()
    with open(path, "a") as journal:
        pool = [threading.Thread(target=worker, args=(n, cycle, journal, lock)) for n in range(threads)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()


def verify(path):
    with open(path) as f:
        confirmed = [line.strip() for line in f if line.strip()]
    if ENGINE.startswith("pg"):
        out = subprocess.run(
            ["psql", "-h", "/tmp", "-p", "5433", "-U", "postgres", "-qAt", "-c", "SELECT k FROM kv"],
            capture_output=True, text=True, check=True).stdout
        present = set(out.split())
        lost = [k for k in confirmed if k not in present]
    else:
        c = ready(Resp())
        lost = [k for k in confirmed if c.cmd("GET", k) != k]
    print(f"{len(confirmed)} confirmed, {len(lost)} lost")
    return len(lost)


if __name__ == "__main__":
    mode, journal = sys.argv[1], sys.argv[3]
    if mode == "load":
        load(journal, int(sys.argv[4]))
    else:
        sys.exit(1 if verify(journal) else 0)
