#!/bin/bash
# Real reboots without flushing (SPEC-friochap CA-2), in a VM of the host's
# architecture: arm64 on an Ampere A1 (what CA-2 asks for), x86_64 on GitHub's
# runners, the only hosted ones with KVM (without it QEMU emulates, and not
# even one boot fits in a 15 min job).
#
# LazyFS (scripts/apagon) simulates a lost page cache; this one loses it for
# real: a Debian 13 VM under QEMU, Luma installed with the production
# unit (deploy/luma.service), load from several threads, and then
# `echo b > /proc/sysrq-trigger`: the kernel reboots on the spot, without
# syncing or unmounting anything. After the reboot systemd starts Luma, and
# everything confirmed must be there (scripts/apagon/carga.py verify).
#
# The disk is cache=none: what the guest did not flush is gone, what it did
# reached the host disk, as with a real drive that honours flushes.
#
#   run.sh <luma binary for this host's architecture>      (Linux host with KVM)
set -euo pipefail

LUMA=$(realpath "$1")
CYCLES=${CYCLES:-10}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${WORK:-/tmp/sysrq}
mkdir -p "$W" && cd "$W"

[ -e /dev/kvm ] || { echo "no /dev/kvm: emulated reboots are too slow to be worth it"; exit 1; }
sudo chmod 666 /dev/kvm
case $(uname -m) in
    aarch64) IMG=arm64 QEMU=(qemu-system-aarch64 -M virt -bios /usr/share/qemu-efi-aarch64/QEMU_EFI.fd) ;;
    x86_64) IMG=amd64 QEMU=(qemu-system-x86_64 -M q35) ;;
    *) echo "unsupported host $(uname -m)"; exit 1 ;;
esac

[ -f base.qcow2 ] || curl -sfL -o base.qcow2 \
    "https://cloud.debian.org/images/cloud/trixie/latest/debian-13-genericcloud-$IMG.qcow2"
rm -f disk.qcow2 key key.pub
qemu-img create -q -f qcow2 -F qcow2 -b base.qcow2 disk.qcow2 8G
ssh-keygen -q -t ed25519 -N '' -f key
cat > user-data <<CFG
#cloud-config
disable_root: false
ssh_authorized_keys: ["$(cat key.pub)"]
CFG
cloud-localds seed.img user-data

"${QEMU[@]}" -enable-kvm -cpu host -m 2048 -smp 2 \
    -drive if=virtio,file=disk.qcow2,cache=none,discard=unmap \
    -drive if=virtio,file=seed.img,format=raw \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22 \
    -display none -serial file:console.log -daemonize

SSH=(ssh -i key -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ConnectTimeout=5 -o LogLevel=ERROR root@127.0.0.1)
vm() { "${SSH[@]}" "$@"; }
wait_ssh() {
    for _ in $(seq 120); do vm true 2>/dev/null && return 0; sleep 2; done
    echo "the VM did not come back"; tail -30 console.log; exit 1
}

wait_ssh
# Throwaway secrets for a throwaway VM: Luma refuses weak ones.
KEYS="LUMA_API_KEY=sysrq-harness-key-0123456789abcdef"  # gitleaks:allow (test only)
MASTER="LUMA_MASTER_KEY=sysrq-harness-master-0123456789abcdef"  # gitleaks:allow (test only)
scp -q -i key -P 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$LUMA" "$REPO/deploy/luma.service" "$REPO/scripts/apagon/carga.py" "$REPO/scripts/sysrq/ready.py" \
    root@127.0.0.1:/root/
vm "set -e
    install -m 755 /root/luma /usr/local/bin/luma
    install -m 644 /root/luma.service /etc/systemd/system/luma.service
    useradd --system --home /var/lib/luma luma
    install -d -o luma -g luma /var/lib/luma
    mkdir -p /etc/luma
    cat > /etc/luma/luma.env <<ENV
$KEYS
$MASTER
DATA_DIR=/var/lib/luma
SQLITE_DB_PATH=/var/lib/luma/sqlite/rustkiss.db
PORT_LUMA_VDB=7700
BIND_ADDR=127.0.0.1
RATE_LIMIT_RPS=0
MAX_DOC_FIND=1000000
SNAPSHOT_INTERVAL_SECS=3
ENV
    systemctl daemon-reload
    systemctl enable --now luma"

for cycle in $(seq "$CYCLES"); do
    vm python3 /root/ready.py
    vm "$KEYS nohup python3 /root/carga.py load /root/journal.log $cycle >/dev/null 2>&1 </dev/null &"
    sleep "$((RANDOM % 4 + 2)).$((RANDOM % 10))"
    # The connection dies with the kernel; that is the point.
    vm "echo b > /proc/sysrq-trigger" 2>/dev/null || true
    sleep 3
    wait_ssh
    vm python3 /root/ready.py
    echo -n "reboot $cycle: "
    if ! vm "$KEYS python3 /root/carga.py verify /root/journal.log /var/lib/luma/queues/apagon"; then
        vm "journalctl -u luma -n 40 --no-pager" || true
        exit 1
    fi
done
echo "all $CYCLES sysrq reboots survived"
