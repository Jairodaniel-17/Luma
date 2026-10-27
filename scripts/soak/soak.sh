#!/bin/bash
# 72 h soak (SPEC-friochap F7, CA-9) on the machine itself, with Luma installed
# as scripts/sysrq/run.sh leaves it (production unit, /etc/luma/luma.env,
# /root/carga.py and /root/ready.py). Run as root under nohup; it does not
# need the machine that started it.
#
# Load paced from six threads (CARGA_PAUSE, ~12 ops/s by default) on KV, blobs,
# queues, vectors and documents, each cycle in its own namespace. Every 2 h the
# cycle ends with a `kill -9` (every sixth, a clean restart: the hot update of
# F7) and must have lost nothing; every 12 h all earlier cycles are checked
# again. Luma's RSS goes to rss.csv every minute, any 5xx to 5xx.log.
#
#   nohup soak.sh [hours] >/dev/null 2>&1 &      results in /root/soak
set -uo pipefail

HOURS=${1:-72}
CYCLE_SECS=${CYCLE_SECS:-7200}
W=/root/soak
mkdir -p "$W"
set -a; . /etc/luma/luma.env; set +a
export CARGA_PAUSE=${CARGA_PAUSE:-0.5} CARGA_5XX=$W/5xx.log
end=$(( $(date +%s) + HOURS * 3600 ))

log() { echo "$(date -Is) $*" >> "$W/soak.log"; }
check() {
    CARGA_NS=soak-c$1 CARGA_DOCS=soak-c$1-docs python3 /root/carga.py verify "$W/j$1.log" \
        "/var/lib/luma/queues/soak-c$1" >> "$W/soak.log" 2>&1 || { log "cycle $1: LOST DATA"; failed=1; }
}

( while :; do echo "$(date +%s),$(ps -o rss= -C luma | head -1)" >> "$W/rss.csv"; sleep 60; done ) &
RSS=$!
trap 'kill $RSS 2>/dev/null' EXIT

log "start: $HOURS h, cycles of $CYCLE_SECS s"
cycle=0 failed=0
while [ "$(date +%s)" -lt "$end" ]; do
    cycle=$((cycle + 1))
    python3 /root/ready.py
    CARGA_NS=soak-c$cycle CARGA_DOCS=soak-c$cycle-docs python3 /root/carga.py load "$W/j$cycle.log" "$cycle" \
        >/dev/null 2>&1 &
    LOAD=$!
    left=$(( end - $(date +%s) ))
    sleep $(( left < CYCLE_SECS ? (left > 60 ? left : 60) : CYCLE_SECS ))
    if (( cycle % 6 == 0 )); then
        log "cycle $cycle: clean restart"; systemctl restart luma
    else
        log "cycle $cycle: kill -9"; systemctl kill -s KILL luma
    fi
    wait "$LOAD"
    python3 /root/ready.py
    check "$cycle"
    if (( cycle % 6 == 0 )); then
        log "every cycle so far, again"
        for c in $(seq "$cycle"); do check "$c"; done
    fi
done
log "final check of all $cycle cycles"
for c in $(seq "$cycle"); do check "$c"; done

errors=$(wc -l < "$W/5xx.log" 2>/dev/null || echo 0)
python3 - "$W/rss.csv" >> "$W/soak.log" <<'EOF'
import sys
rows = [l.split(",") for l in open(sys.argv[1]) if l.strip().count(",") == 1]
kb = [int(r[1]) for r in rows if r[1].strip()]
hour = 60
if len(kb) > 4 * hour:
    first = sorted(kb[hour:2 * hour])[hour // 2]      # second hour: past the warm-up
    last = sorted(kb[-hour:])[hour // 2]
    print(f"RSS median 2nd hour {first // 1024} MB, last hour {last // 1024} MB, peak {max(kb) // 1024} MB")
EOF
log "done: $cycle cycles, 5xx answers: $errors, $([ "$failed" = 0 ] && echo 'nothing lost' || echo 'DATA LOST')"
[ "$failed" = 0 ] && [ "$errors" = 0 ]
