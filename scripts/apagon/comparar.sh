#!/bin/bash
# Luma against other engines under the same power cut (LazyFS, see run.sh).
#
#   comparar.sh <engine>   engine: luma | redis-always | redis-everysec | pg-on | pg-off
#                                  | etcd | sqlite-full | sqlite-normal
#
# Same workload for all (comparar.py: key = value from six threads, journalled
# once the engine confirms), SIGKILL at a random moment, unsynced data dropped,
# restart, count the confirmed writes that are gone. The lossy settings
# (Redis appendfsync everysec, Postgres synchronous_commit=off) are the control:
# if they never lost anything the harness would not be cutting deep enough.
# SQLite runs inside the load process, so for it the cut kills that process.
# Fails when an engine that promises durability (luma, redis-always, pg-on,
# etcd, sqlite-full) loses a confirmed write.
set -euo pipefail

ENGINE=$1
CYCLES=${CYCLES:-20}
LUMA=${LUMA:-/luma/luma}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/var/lazy-root
MNT=/mnt/lazy
FIFO=/tmp/faults.fifo
WORK=/var/apagon
JOURNAL=$WORK/journal-$ENGINE.log

mkdir -p "$ROOT" "$MNT" "$WORK"
chmod 777 "$WORK"  # pg_ctl writes its log here as the postgres user
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
mountpoint -q "$MNT" || { echo "LazyFS did not mount"; exit 1; }

# Throwaway secrets for a throwaway data dir.
export LUMA_API_KEY=compare-harness-key-0123456789abcdef  # gitleaks:allow (test only)
export LUMA_MASTER_KEY=compare-harness-master-0123456789abcdef  # gitleaks:allow (test only)
PG=$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | tail -1)

wait_port() {
    python3 -c "
import socket, sys, time
for _ in range(600):
    try:
        socket.create_connection(('127.0.0.1', $1), timeout=1).close(); sys.exit(0)
    except OSError:
        time.sleep(0.1)
sys.exit('engine did not come up')"
}

PID=
start() {
    case $ENGINE in
    luma)
        DATA_DIR=$MNT/data SQLITE_DB_PATH=$MNT/data/sqlite/rustkiss.db PORT_LUMA_VDB=7700 \
            RESP_PORT=6390 BIND_ADDR=127.0.0.1 RATE_LIMIT_RPS=0 SNAPSHOT_INTERVAL_SECS=3 \
            "$LUMA" serve >>"$WORK/engine.log" 2>&1 &
        PID=$!; wait_port 6390 ;;
    redis-*)
        mkdir -p "$MNT/redis"
        redis-server --dir "$MNT/redis" --appendonly yes --appendfsync "${ENGINE#redis-}" \
            --save "" --port 6380 --bind 127.0.0.1 --protected-mode no >>"$WORK/engine.log" 2>&1 &
        PID=$!; wait_port 6380 ;;
    etcd)
        etcd --data-dir "$MNT/etcd" --listen-client-urls http://127.0.0.1:2379 \
            --advertise-client-urls http://127.0.0.1:2379 --listen-peer-urls http://127.0.0.1:2380 \
            >>"$WORK/engine.log" 2>&1 &
        PID=$!; wait_port 2379 ;;
    sqlite-*) PID= ;;
    pg-*)
        if [ ! -d "$MNT/pg" ]; then
            install -d -o postgres "$MNT/pg"
            su postgres -c "$PG/initdb -D $MNT/pg --auth=trust" >>"$WORK/engine.log" 2>&1
            FIRST=1
        fi
        su postgres -c "$PG/pg_ctl -D $MNT/pg -l $WORK/pg.log -w \
            -o '-p 5433 -k /tmp -c synchronous_commit=${ENGINE#pg-}' start" >/dev/null
        PID=$(head -1 "$MNT/pg/postmaster.pid")
        if [ "${FIRST:-}" = 1 ]; then
            psql -h /tmp -p 5433 -U postgres -qc "CREATE TABLE kv (k text PRIMARY KEY)"; FIRST=
        fi ;;
    esac
}

cut() {
    kill -9 "$PID" 2>/dev/null || true
    [[ $ENGINE == pg-* ]] && pkill -9 -u postgres postgres 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    sleep 0.3
    echo "lazyfs::clear-cache" > "$FIFO"
    sleep 0.5
}

lost=0
for cycle in $(seq "$CYCLES"); do
    start
    python3 "$HERE/comparar.py" load "$ENGINE" "$JOURNAL" "$cycle" &
    LOAD=$!
    [[ $ENGINE == sqlite-* ]] && PID=$LOAD
    sleep "$((RANDOM % 4 + 1)).$((RANDOM % 10))"
    cut
    wait "$LOAD" || true
    start
    echo -n "$ENGINE cut $cycle: "
    python3 "$HERE/comparar.py" verify "$ENGINE" "$JOURNAL" || lost=1
    cut
done

echo "$ENGINE: $(wc -l < "$JOURNAL") writes confirmed over $CYCLES cuts"
case $ENGINE in
luma | redis-always | pg-on | etcd | sqlite-full) [ "$lost" = 0 ] || { echo "$ENGINE promises durability and lost writes"; exit 1; } ;;
esac
