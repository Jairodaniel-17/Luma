#!/bin/bash
# Disaster drill (SPEC-friochap F6): the host is lost, a new one is rebuilt
# from the remote alone, and every write confirmed more than one shipping
# interval before the loss must be there.
#
#   dr.sh primary  — serve with remote backup and WAL shipping on, load from
#                    several threads, take a full backup midway, SIGKILL.
#   dr.sh restore  — on an empty data dir: `luma restore-remote`, serve, and
#                    check the journal (carga.py verify-dr). Prints the RTO.
#
# Both run in the luma-apagon image on a network with an S3 server (MinIO);
# only /var/dr/journal is shared between them. Driven by scripts/dr_drill.sh.
set -euo pipefail

LUMA=${LUMA:-/luma/luma}
HERE=$(cd "$(dirname "$0")" && pwd)
J=/var/dr/journal
RPO=${WAL_SHIP_INTERVAL_SECS:-10}
export LUMA_API_KEY=dr-drill-key-0123456789abcdef
export LUMA_MASTER_KEY=dr-drill-master-0123456789abcdef
export PORT_LUMA_VDB=7700 BIND_ADDR=127.0.0.1 RATE_LIMIT_RPS=0 MAX_DOC_FIND=1000000
export SNAPSHOT_INTERVAL_SECS=5 WAL_SHIP_INTERVAL_SECS=$RPO
export LUMA_URL=http://127.0.0.1:7700
export DATA_DIR=/var/dr/data SQLITE_DB_PATH=/var/dr/data/sqlite/rustkiss.db
export BACKUP_DIR=/var/dr/backups

ready() {
  python3 - <<'PY' || { tail -20 /var/dr/luma.log; exit 1; }
import time, urllib.request
for _ in range(600):
    try:
        if urllib.request.urlopen("http://127.0.0.1:7700/v1/health/ready", timeout=1).status == 200:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        time.sleep(0.1)
raise SystemExit("Luma did not become ready")
PY
}

case "$1" in
primary)
  mkdir -p "$J"; : > "$J/journal.log"
  "$LUMA" serve >/var/dr/luma.log 2>&1 &
  pid=$!
  ready
  python3 "$HERE/carga.py" load "$J/journal.log" 1 &
  # The full backup lands midway, so the restore needs both it and the WAL
  # shipped after it.
  sleep "${BEFORE_BACKUP:-15}"
  if python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(os.environ["LUMA_URL"] + "/v1/admin/backup", method="POST",
                             headers={"Authorization": "Bearer " + os.environ["LUMA_API_KEY"]})
print("full backup", json.load(urllib.request.urlopen(req, timeout=120))["path"])
PY
  then :; else tail -40 /var/dr/luma.log; exit 1; fi
  sleep "${AFTER_BACKUP:-25}"
  # Peak RSS under this load (SPEC-friochap F4).
  grep VmHWM "/proc/$pid/status"
  python3 -c 'import time; print(time.time())' > "$J/killed_at"
  kill -9 "$pid"
  wait || true
  grep -c '"ok"' "$J/journal.log" | sed 's/^/confirmed operations: /'
  grep -E "shipped|ERROR" /var/dr/luma.log | tail -3
  ;;
restore)
  start=$(date +%s.%N)
  "$LUMA" restore-remote
  "$LUMA" serve >/var/dr/luma.log 2>&1 &
  ready
  end=$(date +%s.%N)
  echo "RTO: $(python3 -c "print(f'{$end - $start:.1f}')") s (restore-remote + start until ready)"
  python3 "$HERE/carga.py" verify-dr "$J/journal.log" "$DATA_DIR/queues/apagon" \
    "$(cat "$J/killed_at")" "$RPO"
  ;;
esac
