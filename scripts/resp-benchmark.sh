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
# So the write ceiling is ~1 700/s and **fsync is not the cause**: removing it
# entirely bought 2x, and pipelining bought nothing. What remains is one redb
# write transaction per operation, serialized behind the global append lock that
# keeps offset order equal to WAL order. Fixing that is an architecture change
# (an in-memory index with redb as the durable projection), not a config knob —
# and it is worth doing only if a real workload needs more than ~1 700 writes/s.
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
