#!/bin/bash
# Run tests/disk_full.rs on a real 64 MB ext4 filesystem (SPEC-friochap CA-3).
#
#   docker run --rm --privileged -v <repo>:/src -w /src rust:1-slim-bookworm \
#     bash scripts/disk_full.sh
set -euo pipefail
apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev e2fsprogs >/dev/null 2>&1
truncate -s 64M /tmp/small.img
mkfs.ext4 -q /tmp/small.img
mkdir -p /mnt/small
mount -o loop /tmp/small.img /mnt/small
LUMA_SMALL_FS=/mnt/small CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/src/target-docker} \
    cargo test --test disk_full -- --nocapture --test-threads=1
