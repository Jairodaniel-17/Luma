#!/bin/bash
# etcd under the same real reboots as run.sh, on a machine you can SSH into as
# root: the comparison LazyFS cannot make, because bbolt's mmap faults on its
# FUSE mount (scripts/apagon/comparar.sh). Same workload as there
# (scripts/apagon/comparar.py), then `echo b > /proc/sysrq-trigger`.
#
#   HOST=<ip> KEY=<ssh key> CYCLES=20 etcd.sh
#   UNSAFE=1 ...   etcd --unsafe-no-fsync: the control, it should lose writes
#
# Fails when etcd (without UNSAFE) loses a confirmed write.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CYCLES=${CYCLES:-20}
KEY=$(realpath "$KEY")
V=v3.5.17
FLAGS=${UNSAFE:+--unsafe-no-fsync}
J=/root/journal-etcd${UNSAFE:+-unsafe}.log
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ConnectTimeout=5 -o LogLevel=ERROR "root@$HOST")
vm() { "${SSH[@]}" "$@"; }
log() { echo "$(date +%T) $*"; }
wait_ssh() {
    for _ in $(seq 120); do vm true 2>/dev/null && return 0; sleep 2; done
    echo "the machine did not come back"; exit 1
}
healthy() {
    vm 'for _ in $(seq 300); do curl -sf localhost:2379/health | grep -q "\"true\"" && exit 0; sleep 0.2; done
        journalctl -u etcd -n 30 --no-pager; exit 1'
}

log "installing etcd $V ${FLAGS}"
scp -q -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$HERE/../apagon/comparar.py" "root@$HOST:/root/"
vm "set -e
    A=\$(dpkg --print-architecture)
    [ -x /usr/local/bin/etcd ] || curl -sfL https://github.com/etcd-io/etcd/releases/download/$V/etcd-$V-linux-\$A.tar.gz \
        | tar xz -C /usr/local/bin --strip-components=1 etcd-$V-linux-\$A/etcd
    cat > /etc/systemd/system/etcd.service <<UNIT
[Service]
Environment=ETCD_UNSUPPORTED_ARCH=\$A
ExecStart=/usr/local/bin/etcd --data-dir /var/lib/etcd --listen-client-urls http://127.0.0.1:2379 --advertise-client-urls http://127.0.0.1:2379 $FLAGS
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl stop etcd 2>/dev/null || true
    rm -rf /var/lib/etcd $J
    systemctl enable --now etcd
    sync"

lost=0
for cycle in $(seq "$CYCLES"); do
    healthy
    vm "nohup python3 /root/comparar.py load etcd $J $cycle >/dev/null 2>&1 </dev/null &"
    sleep "$((RANDOM % 4 + 2)).$((RANDOM % 10))"
    # The connection dies with the kernel; that is the point.
    vm "echo b > /proc/sysrq-trigger" 2>/dev/null || true
    sleep 3
    wait_ssh
    healthy
    echo -n "reboot $cycle: "
    vm "python3 /root/comparar.py verify etcd $J" || lost=1
done
log "etcd ${FLAGS:-(fsync)}: $CYCLES reboots"
[ -n "${UNSAFE:-}" ] || [ "$lost" = 0 ] || { echo "etcd promises durability and lost writes"; exit 1; }
