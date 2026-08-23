#!/usr/bin/env bash
# F4.4 — honest RESP benchmark against a real Redis 7.
#
# Honest means three things this script enforces:
#
#   1. The same client drives both. `redis-benchmark` runs from a container, so
#      the numbers are comparable to each other even where they are not
#      comparable to a bare-metal Redis.
#   2. PING is measured too. It is the control: it crosses the same network and
#      does no work, so it separates "Luma is slow" from "this network is slow".
#   3. SET is measured with and without pipelining. If pipelining does not help,
#      the ceiling is a serialization inside the server, not round-trip latency.
#
# What the first run found, and why it matters more than the totals:
#
#   PING            33 613/s     (control: the network path itself)
#   GET             33 112/s     — reads track the control, so reads are free
#   SET             785/s        with wal_sync_mode = per_write
#   SET             1 625/s      with wal_sync_mode = group (no fsync per write)
#   SET -P 16       1 747/s      pipelined — barely better than unpipelined
#
# Two facts, and together they name the cause: removing the fsync entirely bought
# only 2x, and pipelining bought nothing. So the ceiling was neither the disk nor
# the round trip — it was a serialization inside the server.
#
# ## What that turned out to be, and where it ended
#
# This comment used to stop here, concluding "the write ceiling is ~1 700/s, it
# is one redb write transaction per operation, and fixing it is worth doing only
# if a real workload needs more". Two of those three were wrong, and the third
# was only the first of four layers:
#
#   785    -> 4 648     group commit: the WAL fsync is shared across a batch
#   4 648  -> 8 893     the projection applies the whole batch in one transaction
#   8 893  -> 22 989    block_in_place: dispatch was blocking a Tokio worker, so
#                       at most `workers` commands were ever in flight (20 here),
#                       and group commit cannot batch writers that never arrive
#
# 29x, with `wal_sync_mode` still `per_write` and no durability guarantee moved.
#
# ## Two things this script is here to stop you from concluding
#
# **The control is not decoration.** PING says 26 178/s, so SET at 22 989 is at
# 88% of what this transport allows, and no server-side change can show up here
# any more. A later swap of the KV projection to an LSM was briefly credited with
# 24 242 -> 26 178 in the docs; both numbers were wrong — 26 178 is this control —
# and the real gain only shows in-process, 26 379 -> 35 622 writes/s at 128
# writers. Re-running the benchmark is what caught it; re-reading it never would.
#
# **The disk decides.** Same binary, same config, same client: 22 989 SET/s with
# data_dir on NVMe, 3 142/s on a spinning disk, and GET unchanged in both (24 671
# vs 25 685) because reads never touch it. Publish the device or publish nothing.
#
# ## Against a real Redis, on the same route
#
# Redis measured over its own container loopback does 147 783 SET/s; measured
# across the Docker NAT to the host — the route this script uses for Luma — it
# does 28 517, with its control moving 150 754 -> 28 763. That gap is transport,
# not engine, which is exactly why the script drives both from the same client.
#
#   Redis 7   PING 28 763   SET 28 517 (99% of control)   GET 27 298
#   Luma      PING 26 178   SET 22 989 (88% of control)   GET 25 685
#
# So Luma sits at 81% of Redis's SET and 94% of its GET — while fsyncing every
# confirmed write, which Redis out of the box does not.
#
# The full per-layer table, and the two designs that were measured and rejected,
# are in `docs/referencia/BENCHMARKS.md`.
#
# Usage:
#   scripts/resp-benchmark.sh <luma-resp-port> <luma-api-key> [redis-container]
set -euo pipefail

PORT="${1:?usage: resp-benchmark.sh <luma-resp-port> <luma-api-key> [redis-container]}"
KEY="${2:?an api key: RESP AUTH maps to Luma's api keys}"
REDIS_CONTAINER="${3:-luma-diff-redis}"
N="${N:-5000}"
CLIENTS="${CLIENTS:-32}"

run() { docker exec "$REDIS_CONTAINER" redis-benchmark "$@" -q; }

echo "── control: PING (same network, no work) ────────────────────────────────"
run -h host.docker.internal -p "$PORT" -a "$KEY" -t ping -n "$N" -c "$CLIENTS" | tail -1

echo "── Luma ─────────────────────────────────────────────────────────────────"
run -h host.docker.internal -p "$PORT" -a "$KEY" -t set,get -n "$N" -c "$CLIENTS" | tail -2
echo "   pipelined (-P 16): if this is flat, the ceiling is serialization"
run -h host.docker.internal -p "$PORT" -a "$KEY" -t set -n "$N" -c "$CLIENTS" -P 16 | tail -1

echo "── Redis 7, same client ─────────────────────────────────────────────────"
run -h 127.0.0.1 -p 6379 -t set,get -n "$N" -c "$CLIENTS" | tail -2
