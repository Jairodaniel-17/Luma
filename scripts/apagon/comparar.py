#!/usr/bin/env python3
"""Same workload, same power cut, several engines (see comparar.sh).

`load` writes key = value from several threads until the engine goes away,
journalling `ok` (fsynced) only after the engine confirmed. `verify` counts
how many confirmed writes are missing after the cut. One client per engine:
RESP for Luma and Redis, psql for PostgreSQL, the HTTP gateway for etcd, and
SQLite in-process (the load process is what the cut kills). Standard library only.
"""

import base64
import json
import os
import sqlite3
import urllib.request
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


SQLITE_DB = "/mnt/lazy/sqlite/kv.db"


class Sq:
    """Autocommit INSERT in WAL mode; `synchronous` from the engine name: FULL
    promises durability, NORMAL (the common default with WAL) does not."""

    def __init__(self):
        self.c = sqlite3.connect(SQLITE_DB, timeout=60, isolation_level=None)
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute(f"PRAGMA synchronous={ENGINE.split('-')[1].upper()}")

    def put(self, key):
        self.c.execute("INSERT OR IGNORE INTO kv VALUES (?)", (key,))


def b64(s):
    return base64.b64encode(s.encode()).decode()


def etcd(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:2379/v3/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


class Etcd:
    def put(self, key):
        etcd("kv/put", {"key": b64(key), "value": b64(key)})


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
        if ENGINE.startswith("pg"):
            client = Pg()
        elif ENGINE.startswith("sqlite"):
            client = Sq()
        elif ENGINE == "etcd":
            client = Etcd()
        else:
            client = ready(Resp())
    except (OSError, RuntimeError, sqlite3.Error):
        return
    i = 0
    while True:
        i += 1
        key = f"c{cycle}w{n}k{i}"
        try:
            if isinstance(client, Resp):
                client.cmd("SET", key, key)
            else:
                client.put(key)
        except (OSError, ConnectionError, RuntimeError, BrokenPipeError, sqlite3.Error):
            return
        with lock:
            journal.write(key + "\n")
            journal.flush()
            os.fsync(journal.fileno())


def load(path, cycle, threads=6):
    if ENGINE.startswith("sqlite"):
        os.makedirs(os.path.dirname(SQLITE_DB), exist_ok=True)
        sqlite3.connect(SQLITE_DB).execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY)")
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
    elif ENGINE.startswith("sqlite"):
        c = sqlite3.connect(SQLITE_DB)
        present = {r[0] for r in c.execute("SELECT k FROM kv")} if c.execute(
            "SELECT 1 FROM sqlite_master WHERE name='kv'").fetchone() else set()
        lost = [k for k in confirmed if k not in present]
    elif ENGINE == "etcd":
        kvs = etcd("kv/range", {"key": b64("c"), "range_end": b64("d"), "keys_only": True}).get("kvs", [])
        present = {base64.b64decode(kv["key"]).decode() for kv in kvs}
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
