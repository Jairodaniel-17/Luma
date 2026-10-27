#!/bin/bash
# Run tests/fsync_eio.rs on a disk that starts returning EIO mid-run
# (SPEC-friochap CA-4): a device-mapper device whose table is switched from
# `linear` to `error` by the test, then back.
#
#   docker run --rm --privileged -v <repo>:/src -w /src rust:1-slim-bookworm \
#     bash scripts/fsync_eio.sh
set -euo pipefail
apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev e2fsprogs dmsetup >/dev/null 2>&1
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/src/target-docker}
cargo test --no-run --test fsync_eio 2>&1 | tail -1

truncate -s 256M /tmp/eio.img
LOOP=$(losetup -f --show /tmp/eio.img)
SECTORS=$(blockdev --getsz "$LOOP")
export LUMA_EIO_DM=luma-eio
export LUMA_EIO_TABLE_OK="0 $SECTORS linear $LOOP 0"
export LUMA_EIO_TABLE_ERR="0 $SECTORS error"
dmsetup create "$LUMA_EIO_DM" --table "$LUMA_EIO_TABLE_OK"
# No udev in a container to create the node.
dmsetup mknodes "$LUMA_EIO_DM"
cleanup() {
    umount /mnt/eio 2>/dev/null || true
    dmsetup remove "$LUMA_EIO_DM" 2>/dev/null || true
    losetup -d "$LOOP" 2>/dev/null || true
}
trap cleanup EXIT
mkfs.ext4 -q "/dev/mapper/$LUMA_EIO_DM"
mkdir -p /mnt/eio
mount "/dev/mapper/$LUMA_EIO_DM" /mnt/eio
export LUMA_EIO_MNT=/mnt/eio LUMA_EIO_RECORD=/tmp/eio-confirmed.txt

LUMA_EIO_PHASE=1 cargo test --test fsync_eio -- --nocapture 2>&1 | grep -vE "^(warning|note|   Compiling|    Finished)"

# What a reboot does: the filesystem is checked and mounted again.
umount /mnt/eio || umount -l /mnt/eio
e2fsck -fy "/dev/mapper/$LUMA_EIO_DM" >/dev/null || true
mount "/dev/mapper/$LUMA_EIO_DM" /mnt/eio

LUMA_EIO_PHASE=2 cargo test --test fsync_eio -- --nocapture 2>&1 | grep -vE "^(warning|note|   Compiling|    Finished)"
