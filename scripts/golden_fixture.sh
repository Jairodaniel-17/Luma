#!/bin/bash
# Record a golden data_dir with a release binary (tests/golden_data_dir.rs).
#
#   golden_fixture.sh <luma binary of the release> <version> <output dir>
#
# Runs anywhere with python3; for a Linux binary from Windows, inside a
# container. Writes one record of each kind, each marked with the version, and
# stops the server with SIGTERM so the directory is what a clean stop leaves.
set -euo pipefail
LUMA=$1 VERSION=$2 OUT=$3
rm -rf "$OUT" && mkdir -p "$OUT"
export LUMA_API_KEY=golden-key LUMA_ALLOW_INSECURE=1 SQLITE_ENABLED=false
"$LUMA" serve --port 7790 --data-dir "$OUT" >/tmp/golden.log 2>&1 &
pid=$!
VERSION=$VERSION python3 - <<'PY'
import json, os, time, urllib.request
v = os.environ["VERSION"]
base = "http://127.0.0.1:7790"
def call(method, path, body=None, raw=False):
    data = body if raw else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Authorization": "Bearer golden-key",
                                          "Content-Type": "application/octet-stream" if raw else "application/json"})
    return urllib.request.urlopen(req, timeout=10).read()
for _ in range(300):
    try:
        call("GET", "/v1/health"); break
    except Exception:
        time.sleep(0.1)
call("PUT", "/v1/state/golden-kv", {"value": {"marker": f"kv-v{v}"}})
call("PUT", "/v1/blob/golden/obj.bin", b"golden-blob-bytes", raw=True)
call("POST", "/v1/queue/golden", {"body": {"marker": f"queue-v{v}"}})
call("POST", "/v1/vector/golden", {"dim": 4, "metric": "cosine"})
call("POST", "/v1/vector/golden/upsert", {"id": "golden-vec", "vector": [1, 0, 0, 0], "meta": {"marker": f"vector-v{v}"}})
call("PUT", "/v1/doc/ordenes/golden-doc", {"estado": "abierto", "marker": f"doc-v{v}"})
PY
kill -TERM "$pid"
wait "$pid" || true
echo "recorded $VERSION in $OUT:"; ls "$OUT"
