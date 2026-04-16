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

The system has three API levels:

- **Level 1** (`/v1/vector`, `/v1/state`, `/v1/doc`, `/v1/sql`, `/v1/events`): Primitive operations on individual subsystems.
- **Level 2** (`/v1/db`): `LumaDatabase` hub that orchestrates auto-chunking, embedding generation, hybrid SQL+vector search, and auto-schema indexing.
- **Level 3** (`/v1/memory/{namespace}/`): NS-Mem agent memory layer — `ingest_event`, `upsert_fact`, `query`, `timeline/{entity_id}`, `upsert_procedure`, `next_step`, `edges`, `beliefs/{fact_key}/history`, `graph/centrality`.

### Key Modules

**`src/memory/`** — NS-Mem: agent memory layer built on top of Luma's primitives. Four memory types:
- **episodic**: concrete events/interactions → vector store + `memory_records` SQLite table.
- **semantic**: stable facts/preferences → vector store + `memory_records` (type `semantic`).
- **procedural**: DAG-based procedures with typed edges and constraint evaluation → SQLite (`procedures`, `proc_nodes`, `proc_edges`, `proc_constraints`).
- **working**: ephemeral session context → KV store with TTL.

Consolidation pipeline: `ingest_event` → LLM (or local heuristic) extracts `FactCandidate` → persisted as `semantic` (`active` if confidence ≥ threshold, else `draft`). Automatically creates a `TriggeredBy` edge (episodic → semantic). LLM providers: `none`, `mock`, `openai`, `ollama`. See `docs/NS_MEM.md` for full API reference.

**`src/memory/graph.rs`** — `GraphService`: typed edge CRUD (`memory_edges` table), semantic walk BFS, simplified PageRank (15 iter, damping 0.85), belief versioning (`memory_history` table).

**`src/memory/graph_api.rs`** — Public graph methods on `MemoryService`: `create_edge`, `node_edges`, `remove_edge`, `get_belief_history`, `refresh_centrality`.

Recall algorithm (v2.0.0): K-NN seeds → BFS expansion over typed edges → score = `cosine × edge_factor × (1 + centrality)` → filter archived → return top-k. Edge factors: `supports=1.0`, `triggered_by=0.8`, `related_to=0.7`, `contradicts=-0.5` (skip), `supersedes=0.0` (skip).

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
                               # NS-Mem tables: memory_records,
                               # memory_edges, memory_history,
                               # procedures, proc_nodes, proc_edges,
                               # proc_constraints, memory_versions
```

### Configuration

`luma.toml` at the repo root (auto-generated if missing). Key sections:
- **Server**: `port`, `bind_addr`, `api_key`
- **Storage**: `data_dir`, `snapshot_interval_secs`, `wal_segment_max_bytes`
- **Vector**: `index_kind` (`HNSW` | `IVF_FLAT_Q8` | `DiskANN`), `max_vector_dim`, `simd_enabled`
- **IVF**: `ivf_clusters`, `ivf_nprobe`, `q8_refine_topk`
- **DiskANN**: `diskann_max_degree`, `diskann_build_threads`
- **Embeddings**: `embedding_provider` (`none` | `ollama` | `openai` | `mock`), `embedding_model`, `embedding_dim`
- **Memory / NS-Mem**: `memory_consolidation_enabled`, `memory_working_ttl_secs`, `memory_default_limit`, `memory_fact_promotion_threshold` (default 0.85), `llm_provider` (`none` | `mock` | `openai` | `ollama`), `llm_model`, `llm_url`, `llm_api_key`
- **Graph / Semantic Walk**: `memory_walk_max_hops` (default 2), `memory_walk_min_similarity` (default 0.65), `memory_walk_max_nodes` (default 40), `memory_centrality_enabled` (default true), `memory_centrality_update_interval_secs` (default 300)

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

### Done — v2.0.0

- **NS-Mem Graph Layer**: replaced flat K-NN recall with semantic walk BFS over typed memory graph (`memory_edges`). Nodes ranked by `cosine × edge_factor × (1 + PageRank)`.
- **`memory_edges` table**: typed weighted edges with auto-creation on consolidation (`TriggeredBy`) and on `upsert_fact` overwrite (`Supersedes`).
- **`memory_history` table**: append-only belief versioning. Every `upsert_fact` on an existing fact snapshots the old version before overwriting.
- **PageRank centrality**: 15-iter simplified PageRank, stored in `memory_records.centrality_score`, recomputable via `POST graph/centrality`.
- **5 new endpoints**: `edges` CRUD, `beliefs/{fact_key}/history`, `graph/centrality`.
- **Security**: updated `quinn-proto` → 0.11.14, `rustls-webpki` → 0.103.12.
- **CI**: all checks pass — fmt, clippy `-D warnings`, 99/99 tests, 0 audit errors.

### Done — v1.4.0

- Reviewed the PR1-PR8 series against code and tests, identified blocking issues in planner, batching, compaction, metrics, and release gates.
- Hardened WAL replay with checksummed envelopes, corruption stop-at-tail semantics, and idempotent recovery paths for `state_db` plus vector replay.
- Added range scans for KV (`start`/`end` end-exclusive), cached metrics rendering off the scrape path, and explicit SSE gap signaling when requested offsets are no longer retained.
- Implemented NS-Mem (`src/memory/`): episodic/semantic/procedural/working memory layer with consolidation pipeline, LLM fact extraction, DAG procedure engine with constraint evaluation, and REST API at `/v1/memory/{namespace}/`. Full docs in `docs/NS_MEM.md`.
