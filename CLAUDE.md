# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Luma** (crate: `rust-kiss-vdb`) is a high-performance convergent data engine for AI applications. It unifies vector search, key-value state, relational SQL, and pub/sub events in a single Rust binary.

## Common Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test
cargo test <test_name>           # Run a single test

# Lint & Format
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Benchmarks
cargo bench
cargo bench --bench vector_bench
cargo bench --bench sqlite_bench
cargo bench --bench vector_mmap_bench

# Run server
./target/release/luma serve

# CLI utilities
./target/release/luma vacuum --collection <name>
./target/release/luma diskann-build <params>
./target/release/luma diskann-status --collection <name>
```

## Architecture

The system has two API levels:

- **Level 1** (`/v1/vector`, `/v1/state`, `/v1/doc`, `/v1/sql`, `/v1/events`): Primitive operations on individual subsystems.
- **Level 2** (`/v1/db`): `LumaDatabase` hub that orchestrates auto-chunking, embedding generation, hybrid SQL+vector search, and auto-schema indexing.

### Key Modules

**`src/engine/mod.rs`** — Core `Engine` struct. Coordinates all subsystems, handles WAL replay on startup, TTL expiration, and periodic snapshots. All mutations are published as events with monotonic offsets (event sourcing pattern).

**`src/engine/hub.rs`** — `LumaDatabase` orchestrator. Handles document ingestion (chunking → embedding → upsert) and hybrid search (SQL pre-filter → vector search → hydration). Background task auto-creates SQL indexes on metadata fields.

**`src/vector/mod.rs`** — Vector store with three indexing strategies:
- **HNSW** (default): In-memory approximate nearest neighbor
- **IVF_FLAT_Q8**: Inverted file index with 8-bit quantization refinement
- **DiskANN**: Disk-based Vamana graph for massive collections

Collections are split into segments (~8,192 vectors each). Active segment receives upserts; frozen segments are read-only. Search merges results across all segments. Vectors are persisted as binary (`vectors.bin`) with zero-copy mmap support.

**`src/engine/persist.rs`** — Segmented WAL (`events-XXXXXX.log`, JSON lines) + periodic snapshots (`snapshot.json`). Snapshot triggers WAL rotation and cleanup.

**`src/engine/state.rs` / `state_db.rs`** — In-memory KV store with JSON values, per-key TTL, and optimistic locking via `if_revision` (compare-and-swap). Optional `redb`-backed persistence.

**`src/engine/events.rs`** — Pub/Sub bus using `tokio::sync::broadcast`. SSE clients stream live events; lagging clients receive "gap" notifications.

**`src/sqlite/mod.rs` + `actor.rs`** — Embedded SQLite in WAL mode. Accessed via actor pattern (tokio MPSC channel) for thread-safe async queries without blocking.

**`src/api/mod.rs`** — Axum HTTP router with Bearer token auth, CORS, request timeouts, and body size limits. Routes split by concern into `routes_vector.rs`, `routes_state.rs`, `routes_doc.rs`, `routes_sql.rs`, `routes_hub.rs`, `routes_events.rs`.

**`src/engine/embeddings.rs`** — Embedding provider abstraction supporting Ollama, OpenAI, and Mock (for tests/CI).

### Data Layout on Disk

```
data/
├── events-000001.log          # Segmented WAL (JSON lines)
├── snapshot.json              # Latest state snapshot
├── vectors/
│   └── <collection>/
│       ├── manifest.json      # Collection metadata (dim, metric)
│       ├── vectors.bin        # Binary vector storage (mmap)
│       └── diskann/           # DiskANN graph (if built)
└── sqlite/
    └── rustkiss.db            # Relational + auth + docstore
```

### Configuration

`luma.toml` at the repo root (auto-generated if missing). Key sections:
- **Server**: `port`, `bind_addr`, `api_key`
- **Storage**: `data_dir`, `snapshot_interval_secs`, `wal_segment_max_bytes`
- **Vector**: `index_kind` (`HNSW` | `IVF_FLAT_Q8` | `DiskANN`), `max_vector_dim`, `simd_enabled`
- **IVF**: `ivf_clusters`, `ivf_nprobe`, `q8_refine_topk`
- **DiskANN**: `diskann_max_degree`, `diskann_build_threads`
- **Embeddings**: `embedding_provider` (`none` | `ollama` | `openai` | `mock`), `embedding_model`, `embedding_dim`

Config source: `src/config.rs`. Environment variables override TOML values.

## CI

CI runs: rustfmt check → clippy (strict, `-D warnings`) → tests → `cargo-audit`. Releases trigger on `v*` tags and produce cross-platform binaries (Linux, Windows, macOS).

Use the `mock` embedding provider in tests to avoid external service dependencies.

## Work Log

### In Progress

- Harden planner execution so `sql_first` and `vector_first` use different code paths, with `_plan` exposed only when explicitly requested.
- Add embedding provider guardrails: bounded non-batch concurrency and cache keys namespaced by provider/model.
- Make HNSW segment compaction conservative by default and avoid long write locks during rebuild.
- Make `/v1/metrics` cheaper by caching RSS outside the scrape path.

### Done In This Turn

- Reviewed the PR1-PR8 series against code and tests, identified blocking issues in planner, batching, compaction, metrics, and release gates.
- Hardened WAL replay with checksummed envelopes, corruption stop-at-tail semantics, and idempotent recovery paths for `state_db` plus vector replay.
- Added range scans for KV (`start`/`end` end-exclusive), cached metrics rendering off the scrape path, and explicit SSE gap signaling when requested offsets are no longer retained.
