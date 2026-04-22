# CHANGELOG

## v2.0.1 (2026-04-22)

### Security
- **CVE fix**: bump `rustls-webpki` 0.103.12 → 0.103.13 (RUSTSEC-2026-0104 — reachable panic in CRL parsing via DoS).
- Remove API key support via query parameter — keys were leaking into access logs and proxy logs.
- Enforce admin role check on all `/v1/auth/keys` endpoints; previously any valid key could create or revoke keys.

### Robustness
- Switch `EmbeddingClient` and `Metrics` from `std::sync::Mutex` to `parking_lot::Mutex` — eliminates mutex-poison cascade panics.
- Add bounded concurrency semaphore to `EmbeddingClient` (`with_limits`) — caps inflight HTTP requests to embedding providers.
- Chunk OpenAI `embed_batch` into ≤ 96 texts per request to avoid rate limits and token overflow.
- Log `TriggeredBy` edge failures in consolidator with `tracing::warn!` instead of silently swallowing errors.
- Raise default HNSW compaction tombstone ratio from 0.2 → 0.5 (conservative by default, avoids compaction under normal write load).

### CI
- Remove `continue-on-error: true` from test and fmt jobs — failing tests now block merges.
- Add MSRV check job (Rust 1.75).
- Add `cargo-deny` job with `deny.toml` for license, source, and advisory enforcement.

---

## v2.0.0 (2026-04-20)

### Added
- **NS-Mem Graph Layer**: replaced flat K-NN recall with semantic walk BFS over typed memory graph (`memory_edges`).
  - Recall score: `cosine × edge_factor × (1 + PageRank)`.
  - Edge factors: `supports=1.0`, `triggered_by=0.8`, `related_to=0.7`, `contradicts=-0.5` (skip), `supersedes=0.0` (skip).
- **`memory_edges` table**: typed weighted edges with auto-creation on consolidation (`TriggeredBy`) and on `upsert_fact` overwrite (`Supersedes`).
- **`memory_history` table**: append-only belief versioning — every `upsert_fact` on an existing fact snapshots the old version.
- **Simplified PageRank** (15 iterations, damping 0.85) stored in `memory_records.centrality_score`; recomputable via `POST /v1/memory/{namespace}/graph/centrality`.
- **5 new API endpoints**:
  - `POST /v1/memory/{namespace}/edges` — create/update edge
  - `GET /v1/memory/{namespace}/edges/{memory_id}` — list edges for a node
  - `POST /v1/memory/{namespace}/edges/{edge_id}/delete` — remove edge
  - `GET /v1/memory/{namespace}/beliefs/{fact_key}/history` — belief version history
  - `POST /v1/memory/{namespace}/graph/centrality` — trigger PageRank recomputation

### Security
- Updated `quinn-proto` → 0.11.14, `rustls-webpki` → 0.103.12.

### CI
- All checks pass: fmt, clippy `-D warnings`, 99/99 tests, 0 audit errors.

---

## v1.4.0 (2026-03-15)

### Added
- **NS-Mem** (`src/memory/`): episodic / semantic / procedural / working memory layer.
  - Consolidation pipeline: `ingest_event` → LLM fact extraction → semantic promotion.
  - Procedural memory: DAG-based procedures with typed edges, priority ordering, and constraint evaluation.
  - LLM providers: `none`, `mock`, `openai`, `ollama`.
  - Full REST API at `/v1/memory/{namespace}/`.
- **WAL hardening**: checksummed envelopes, corruption stop-at-tail semantics, idempotent recovery for `state_db` and vector replay.
- **Range scans for KV**: `start`/`end` parameters (end-exclusive) on `GET /v1/state`.
- **Cached metrics**: RSS read outside the scrape path; `GET /v1/metrics` returns pre-rendered Prometheus text.
- **SSE gap signaling**: explicit `gap` events when clients request offsets no longer retained.

### Documentation
- `docs/NS_MEM.md` — full API reference for the memory layer.

---

## v1.3.0 (2025-12-20)

### Added
- **LumaDatabase hub** (`/v1/db/{namespace}/`): orchestrates auto-chunking, embedding generation, hybrid SQL+vector search, and auto-schema indexing.
- **Hybrid search planner**: decides `sql_first` vs `vector_first` strategy based on filter selectivity and collection size.
- **Multi-tenant isolation**: per-namespace vector collections and SQL tables.
- **DiskANN index**: disk-based Vamana graph for collections beyond HNSW memory limits. CLI: `diskann-build`, `diskann-status`.

---

## v1.2.0 (2025-10-05)

### Added
- **IVF_FLAT_Q8 index**: inverted file index with 8-bit quantization refinement for large-scale approximate search.
- **SIMD acceleration**: AVX2 dot-product and int8 dot-product intrinsics with runtime feature detection.
- **mmap vector storage**: zero-copy memory-mapped access to frozen segments.
- **Segment compaction**: background compaction triggered by tombstone ratio threshold.

---

## v1.1.0 (2025-08-15)

### Added
- **HNSW segmentation**: collections split into ~8,192-vector segments; active segment receives upserts, frozen segments are read-only. Search merges results across all segments.
- **Pub/Sub events** (`/v1/events`, `/v1/stream`): SSE-based event bus with monotonic offsets and gap detection.
- **TTL for KV state**: per-key TTL with background expiration.
- **Optimistic locking**: `if_revision` compare-and-swap for KV put/delete.
- **OpenAPI spec** (`docs/openapi.yaml`): generated from route definitions.

---

## v0.2.0 (2025-12-14)

### Added
- Secure bind (`127.0.0.1` default; `--bind`/`--unsafe-bind` flags for explicit exposure).
- Consistent error responses (`{ "error": "...", "message": "..." }` across all routes).
- Batch APIs: `/v1/state/batch_put`, `/v1/vector/*_batch`.
- DocStore over KV (`/v1/doc/*` + metadata find).
- SQLite module (`/v1/sql/query`, `/v1/sql/exec`).
- Vector store segmentation + metadata keyword index.
- `vacuum` CLI command for vector collection compaction.

---

## v0.1.0 (2025-10-01)

Initial release. Core primitives:
- Vector store with HNSW (in-memory, single segment).
- Key-value state store with JSON values.
- WAL persistence with periodic snapshots.
- Axum HTTP server with Bearer token auth.
- Basic `cargo bench` for vector add/search.
