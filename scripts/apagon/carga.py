#!/usr/bin/env python3
"""Workload and checker for the power-loss harness (see run.sh).

`load` writes through Luma's HTTP API from several threads until the server
goes away, journalling every operation twice: `try` before sending it and `ok`
once the server confirmed it. The journal lives outside the LazyFS mount, so
it survives the simulated power cut.

`verify` reads it back after the restart and checks the one promise that
matters: **everything confirmed is there, and nothing confirmed as deleted
came back.** An operation in flight at the cut may or may not have landed;
both outcomes are accepted, nothing else is.

Standard library only: it runs in the bare LazyFS container.
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("LUMA_URL", "http://127.0.0.1:7700")
KEY = os.environ.get("LUMA_API_KEY", "apagon")
# The soak (scripts/soak/run.sh) gives each cycle its own blob bucket, queue,
# vector collection and doc collection, so a cycle is verified on its own, and
# paces the threads; it also counts 5xx answers, which must not happen.
NS = os.environ.get("CARGA_NS", "apagon")
DOCS = os.environ.get("CARGA_DOCS", "ordenes")
PAUSE = float(os.environ.get("CARGA_PAUSE", "0"))
ERRORS = os.environ.get("CARGA_5XX")


def call(method, path, body=None, raw=False, timeout=10):
    data = None
    headers = {"Authorization": f"Bearer {KEY}"}
    if body is not None:
        if raw:
            data = body
            headers["Content-Type"] = "application/octet-stream"
        else:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        if e.code >= 500 and ERRORS:
            with open(ERRORS, "a") as f:
                f.write(f"{time.time():.0f} {method} {path} {e.code}\n")
        return e.code, e.read()


class Journal:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, *fields):
        # The time last: the disaster drill (verify-dr) needs to know what was
        # confirmed before a given moment.
        line = json.dumps([*fields, time.time()])
        with self.lock:
            self.f.write(line + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())


def worker(n, cycle, journal, stop):
    """One thread, its own keys: at most one operation per key is in flight."""
    i = 0
    while not stop.is_set():
        if PAUSE:
            stop.wait(PAUSE)
        i += 1
        kind = i % 7
        try:
            if kind in (0, 1):
                key = f"c{cycle}w{n}k{i}"
                value = {"n": i, "w": n}
                journal.write("try", "put", key, value)
                status, _ = call("PUT", f"/v1/state/{key}", {"value": value})
                if status == 200:
                    journal.write("ok", "put", key, value)
                if i % 4 == 1 and status == 200:
                    journal.write("try", "del", key, None)
                    status, _ = call("DELETE", f"/v1/state/{key}")
                    if status == 200:
                        journal.write("ok", "del", key, None)
            elif kind == 2:
                body = os.urandom(512 + i % 3000)
                key = f"c{cycle}w{n}b{i}"
                digest = hashlib.sha256(body).hexdigest()
                journal.write("try", "blob", key, digest)
                status, _ = call("PUT", f"/v1/blob/{NS}/{key}", body, raw=True)
                if status == 200:
                    journal.write("ok", "blob", key, digest)
                    if i % 5 == 2:
                        journal.write("try", "blobdel", key, None)
                        status, _ = call("DELETE", f"/v1/blob/{NS}/{key}")
                        if status == 200:
                            journal.write("ok", "blobdel", key, None)
            elif kind == 3:
                journal.write("try", "enq", None, None)
                status, raw = call("POST", f"/v1/queue/{NS}", {"body": {"i": i}})
                if status == 200:
                    msg_id = json.loads(raw)["id"]
                    journal.write("ok", "enq", msg_id, None)
                    if i % 3 == 0:
                        journal.write("try", "ack", msg_id, None)
                        status, _ = call("DELETE", f"/v1/queue/{NS}/{msg_id}")
                        if status == 200:
                            journal.write("ok", "ack", msg_id, None)
            elif kind == 5:
                vec_id = f"c{cycle}w{n}v{i}"
                meta = {"i": i, "w": n}
                journal.write("try", "vec", vec_id, meta)
                body = {"id": vec_id, "vector": [float(n + 1), float(i), 1.0, 0.5], "meta": meta}
                status, _ = call("POST", f"/v1/vector/{NS}/upsert", body)
                if status == 200:
                    journal.write("ok", "vec", vec_id, meta)
                    if i % 4 == 1:
                        journal.write("try", "vecdel", vec_id, None)
                        status, _ = call("POST", f"/v1/vector/{NS}/delete", {"id": vec_id})
                        if status == 200:
                            journal.write("ok", "vecdel", vec_id, None)
            else:
                doc_id = f"c{cycle}w{n}d{i}"
                estado = "abierto" if i % 2 else "cerrado"
                doc = {"estado": estado, "n": i}
                journal.write("try", "doc", doc_id, doc)
                status, _ = call("PUT", f"/v1/doc/{DOCS}/{doc_id}", doc)
                if status == 200:
                    journal.write("ok", "doc", doc_id, doc)
        except (urllib.error.URLError, ConnectionError, OSError):
            # The server is gone: the cut happened. What was in flight stays a
            # `try` without an `ok`.
            return


def load(journal_path, cycle, threads=6):
    # 409 when it already exists from an earlier cycle.
    call("POST", f"/v1/vector/{NS}", {"dim": 4, "metric": "cosine"})
    journal = Journal(journal_path)
    stop = threading.Event()
    pool = [
        threading.Thread(target=worker, args=(n, cycle, journal, stop), daemon=True)
        for n in range(threads)
    ]
    for t in pool:
        t.start()
    for t in pool:
        t.join()


def replay_journal(path):
    """Per subject: the last confirmed state and, if any, the operation that
    was in flight after it."""
    last_ok = {}
    pending = {}
    enq_ok = set()
    enq_tries = 0
    acked = set()
    ack_pending = set()
    with open(path) as f:
        for line in f:
            try:
                phase, op, subject, payload = json.loads(line)[:4]
            except ValueError:
                continue  # the journal line being written at the cut
            if op == "enq":
                if phase == "try":
                    enq_tries += 1
                else:
                    enq_ok.add(subject)
                continue
            if op == "ack":
                (ack_pending if phase == "try" else acked).add(subject)
                continue
            family = {
                "put": "kv",
                "del": "kv",
                "blob": "blob",
                "blobdel": "blob",
                "doc": "doc",
                "vec": "vec",
                "vecdel": "vec",
            }[op]
            slot = (family, subject)
            if phase == "try":
                pending[slot] = (op, payload)
            else:
                last_ok[slot] = (op, payload)
                pending.pop(slot, None)
    return last_ok, pending, enq_ok, enq_tries, acked, ack_pending - acked


def read(path):
    """GET that tells "absent" (404) from anything else going wrong. Only a
    404 may count as a missing value: a 429 or a 500 read as "missing" would
    report data loss that did not happen."""
    status, raw = call("GET", path)
    if status not in (200, 404):
        raise SystemExit(f"GET {path} answered {status}: {raw[:200]!r}")
    return raw if status == 200 else None


def observed(family, subject):
    if family == "kv":
        raw = read(f"/v1/state/{subject}")
        return ("put", json.loads(raw)["value"]) if raw is not None else ("del", None)
    if family == "blob":
        raw = read(f"/v1/blob/{NS}/{subject}")
        return ("blob", hashlib.sha256(raw).hexdigest()) if raw is not None else ("blobdel", None)
    if family == "vec":
        raw = read(f"/v1/vector/{NS}/get?id={subject}")
        return ("vec", json.loads(raw)["meta"]) if raw is not None else ("vecdel", None)
    raw = read(f"/v1/doc/{DOCS}/{subject}")
    return ("doc", json.loads(raw)["doc"]) if raw is not None else ("absent", None)


def verify(journal_path, queue_dir):
    last_ok, pending, enq_ok, enq_tries, acked, ack_unknown = replay_journal(journal_path)
    failures = []
    for slot, state in last_ok.items():
        allowed = [state]
        if slot in pending:
            allowed.append(pending[slot])
        got = observed(*slot)
        if got not in allowed:
            failures.append(f"{slot}: expected one of {allowed}, found {got}")

    # Documents must also be reachable through their index.
    by_state = {}
    for estado in ("abierto", "cerrado"):
        status, raw = call("POST", f"/v1/doc/{DOCS}/find", {"filter": {"estado": estado}, "limit": 100000})
        if status == 200:
            by_state[estado] = {d["id"] for d in json.loads(raw)["documents"]}
    for (family, subject), (op, payload) in last_ok.items():
        if family == "doc" and ("doc", subject) not in pending:
            if subject not in by_state.get(payload["estado"], set()):
                failures.append(f"doc {subject} not found through its index ({payload['estado']})")

    # Queue: read the message files directly. Receiving would put leases on
    # them and hide them from the next cycle's check.
    present = set()
    for name in os.listdir(queue_dir) if os.path.isdir(queue_dir) else []:
        if name.endswith(".json") and not name.startswith(".tmp-"):
            with open(os.path.join(queue_dir, name), "rb") as f:
                try:
                    present.add(json.load(f)["id"])
                except ValueError:
                    failures.append(f"queue message {name} is unreadable")
    missing = (enq_ok - acked - ack_unknown) - present
    resurrected = (acked & present)
    unknown = present - enq_ok
    if missing:
        failures.append(f"{len(missing)} confirmed queue messages lost, e.g. {sorted(missing)[:3]}")
    if resurrected:
        failures.append(f"{len(resurrected)} acked queue messages came back, e.g. {sorted(resurrected)[:3]}")
    if len(unknown) > enq_tries - len(enq_ok):
        failures.append(f"{len(unknown)} queue messages nobody enqueued")

    print(
        f"checked {len(last_ok)} keys/blobs/docs, {len(enq_ok)} enqueues, "
        f"{len(pending)} in flight at the cut: {len(failures)} failures"
    )
    for f in failures[:40]:
        print("  FAIL", f)
    return 1 if failures else 0


ABSENT = {"kv": ("del", None), "blob": ("blobdel", None), "vec": ("vecdel", None), "doc": ("absent", None)}
FAMILY = {"put": "kv", "del": "kv", "blob": "blob", "blobdel": "blob", "doc": "doc", "vec": "vec", "vecdel": "vec"}


def verify_dr(journal_path, queue_dir, killed_at, rpo):
    """After restoring on another machine from the remote: everything confirmed
    more than `rpo` seconds before the host died is there; what came later may
    or may not be. Also prints the RPO actually observed: the age of the newest
    confirmed write that did not make it."""
    lines = []
    with open(journal_path) as f:
        for line in f:
            try:
                phase, op, subject, payload, at = json.loads(line)
            except ValueError:
                continue
            lines.append((phase, op, subject, payload, at))

    slots = {(FAMILY[op], subject) for _, op, subject, _, _ in lines if op in FAMILY}
    seen = {slot: observed(*slot) for slot in slots}
    present = set()
    for name in os.listdir(queue_dir) if os.path.isdir(queue_dir) else []:
        if name.endswith(".json") and not name.startswith(".tmp-"):
            with open(os.path.join(queue_dir, name), "rb") as f:
                present.add(json.load(f)["id"])

    def failures(cutoff):
        """What is wrong if everything confirmed up to `cutoff` must be there."""
        state, allowed = {}, {}
        enq, acked, ack_tried = set(), set(), set()
        for phase, op, subject, payload, at in lines:
            if op == "enq":
                if phase == "ok" and at <= cutoff:
                    enq.add(subject)
                continue
            if op == "ack":
                (ack_tried if phase == "try" or at > cutoff else acked).add(subject)
                continue
            slot = (FAMILY[op], subject)
            if phase == "ok" and at <= cutoff:
                state[slot] = (op, payload)
                allowed[slot] = set()
            else:
                # In flight at the cutoff or later: may have made it or not.
                allowed.setdefault(slot, set()).add(json.dumps((op, payload)))
        bad = []
        for slot in slots:
            must = state.get(slot, ABSENT[slot[0]])
            ok = {json.dumps(must)} | allowed.get(slot, set())
            if json.dumps(seen[slot]) not in ok:
                bad.append(f"{slot}: expected {must}, found {seen[slot]}")
        lost = (enq - acked - ack_tried) - present
        if lost:
            bad.append(f"{len(lost)} queue messages confirmed before the cutoff are missing")
        back = acked & present
        if back:
            bad.append(f"{len(back)} queue messages acked before the cutoff came back")
        return bad

    promised = failures(killed_at - rpo)
    # The newest moment up to which everything is there, to 0.1 s.
    oks = sorted(at for phase, _, _, _, at in lines if phase == "ok" and at <= killed_at)
    lo, hi = 0, len(oks)
    while lo < hi:
        mid = (lo + hi) // 2
        if failures(oks[mid]):
            hi = mid
        else:
            lo = mid + 1
    newest = oks[lo - 1] if lo else None
    observed_rpo = killed_at - newest if newest else float("inf")
    print(
        f"{len(slots)} keys/blobs/docs/vectors, {len(present)} queue messages; "
        f"everything confirmed up to {observed_rpo:.1f} s before the cut is there; "
        f"promise (RPO {rpo} s): {len(promised)} failures"
    )
    for f in promised[:40]:
        print("  FAIL", f)
    return 1 if promised else 0


if __name__ == "__main__":
    mode, journal_path = sys.argv[1], sys.argv[2]
    if mode == "load":
        load(journal_path, int(sys.argv[3]))
    elif mode == "verify-dr":
        sys.exit(verify_dr(journal_path, sys.argv[3], float(sys.argv[4]), float(sys.argv[5])))
    else:
        sys.exit(verify(journal_path, sys.argv[3]))
