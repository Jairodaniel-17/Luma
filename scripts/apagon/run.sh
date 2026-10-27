#!/bin/bash
# Power-loss harness.
#
# Luma's data directory sits on LazyFS, a FUSE filesystem that keeps every
# write in its own cache until it is fsynced. `lazyfs::clear-cache` throws that
# cache away, which is what a power cut does to the page cache. Each cycle:
# load from several threads, SIGKILL Luma at a random moment, drop the unsynced
# data, restart, and check that every confirmed write is there (carga.py).
#
# A SIGKILL alone (tests/crash_recovery.rs) never catches a missing fsync: the
# kernel still holds the data and writes it out. This does.
#
# Runs inside the luma-apagon image (Dockerfile next to this file):
#   docker run --rm --privileged --device /dev/fuse \
#     -v <dir with the linux luma binary>:/luma -v <this dir>:/apagon \
#     luma-apagon /apagon/run.sh
set -euo pipefail

CYCLES=${CYCLES:-20}
LUMA=${LUMA:-/luma/luma}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/var/lazy-root
MNT=/mnt/lazy
FIFO=/tmp/faults.fifo
WORK=/var/apagon
JOURNAL=$WORK/journal.log

mkdir -p "$ROOT" "$MNT" "$WORK"
: > "$JOURNAL"
cat > /tmp/lazyfs.toml <<EOF
[faults]
fifo_path="$FIFO"
[cache]
apply_eviction=false
[cache.simple]
custom_size="2gb"
blocks_per_page=1
[filesystem]
log_all_operations=false
logfile=""
EOF

(cd /opt/lazyfs/lazyfs && ./scripts/mount-lazyfs.sh -c /tmp/lazyfs.toml -m "$MNT" -r "$ROOT" >"$WORK/lazyfs.log" 2>&1 &)
for _ in $(seq 50); do mountpoint -q "$MNT" && break; sleep 0.2; done
mountpoint -q "$MNT" || { echo "LazyFS did not mount"; cat "$WORK/lazyfs.log"; exit 1; }

export DATA_DIR=$MNT/data
export SQLITE_DB_PATH=$MNT/data/sqlite/rustkiss.db
# Throwaway secrets for a throwaway data dir: Luma refuses weak ones.
export LUMA_API_KEY=apagon-harness-key-0123456789abcdef  # gitleaks:allow (throwaway, test only)
export LUMA_MASTER_KEY=apagon-harness-master-0123456789abcdef
export PORT_LUMA_VDB=7700
export BIND_ADDR=127.0.0.1
export MAX_DOC_FIND=1000000
# No rate limit: the checker reads back thousands of keys in a burst.
export RATE_LIMIT_RPS=0
# Snapshots and WAL rotation inside the run, so they get cut too.
export SNAPSHOT_INTERVAL_SECS=3
export LUMA_URL=http://127.0.0.1:7700

# LUMA_OLD=<binary> checks an upgrade (SPEC-friochap CA-7): cycle 1 loads with
# the old binary, is stopped under load, and from then on $LUMA runs on the
# same data directory, as a one-way upgrade does.
LUMA_PID=
start_luma() {
    local bin=${1:-$LUMA}
    (cd "$WORK" && exec "$bin" serve >>"$WORK/luma.log" 2>&1) &
    LUMA_PID=$!
    python3 - <<'EOF'
import time, urllib.request
for _ in range(300):
    try:
        if urllib.request.urlopen("http://127.0.0.1:7700/v1/health", timeout=1).status == 200:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        time.sleep(0.2)
raise SystemExit("Luma did not become ready")
EOF
}

# CUT=term checks an orderly stop instead (SPEC-friochap F5): SIGTERM under
# load, the page cache kept, and the same check that nothing confirmed is lost.
CUT=${CUT:-power}
power_cut() {
    if [ "$CUT" = term ]; then
        local started=$SECONDS
        kill -TERM "$LUMA_PID" 2>/dev/null || true
        wait "$LUMA_PID" 2>/dev/null || true
        if [ $((SECONDS - started)) -gt 12 ]; then
            echo "SIGTERM took $((SECONDS - started)) s"
            exit 1
        fi
        return
    fi
    kill -9 "$LUMA_PID" 2>/dev/null || true
    wait "$LUMA_PID" 2>/dev/null || true
    echo "lazyfs::clear-cache" > "$FIFO"
    sleep 0.5
}

for cycle in $(seq "$CYCLES"); do
    if [ "$cycle" = 1 ]; then start_luma "${LUMA_OLD:-$LUMA}"; else start_luma; fi
    python3 "$HERE/carga.py" load "$JOURNAL" "$cycle" &
    LOAD=$!
    sleep "$((RANDOM % 4 + 1)).$((RANDOM % 10))"
    power_cut
    wait "$LOAD" || true

    start_luma
    echo -n "cycle $cycle: "
    if ! python3 "$HERE/carga.py" verify "$JOURNAL" "$DATA_DIR/queues/apagon"; then
        echo "--- last lines of luma.log"
        tail -40 "$WORK/luma.log"
        exit 1
    fi
    power_cut
done
echo "all $CYCLES cuts ($CUT) survived"
