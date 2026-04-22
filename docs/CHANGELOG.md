# CHANGELOG

## 🚀 Novedades en v3.0.0 (Search, Observabilidad y NS-Mem Avanzado)

### Búsqueda y exportación

- **Batch search** (`POST /v1/vector/{collection}/search_batch`): hasta 100 queries ejecutadas en paralelo internamente (`rayon`). Reduce 100 round-trips a 1.
- **Scroll / cursor API** (`GET /v1/vector/{collection}/scroll`): paginación lexicográfica con cursor opaco para exportar colecciones completas sin límite de `k`.
- **Reranking por coseno** (`POST /v1/vector/{collection}/rerank`): recibe IDs + query (texto o vector), embebe si es texto, reordena por coseno real. Ideal para pipeline search-then-rerank.
- **Aggregations** (`POST /v1/vector/{collection}/aggregate`): `{ "group_by": "campo", "filter": {...}, "limit": N }` — cuenta ítems por valor usando el keyword index. Fast path O(1) para campos indexados.
- **Pre-filter threshold configurable** (`pre_filter_threshold`, default 10 000): brute-force automático sobre subconjuntos filtrados, más eficiente que HNSW + post-filter para corpus muy filtrados.

### Embeddings

- **4 nuevos providers**: `azure` (Azure OpenAI), `cohere` (v1/embed con input_type), `huggingface` (Inference API), añadidos a los existentes `openai`, `ollama`, `mock`.
- **Retry con backoff exponencial + jitter**: `EMBEDDING_RETRY_ATTEMPTS` (default 3) y `EMBEDDING_RETRY_INITIAL_MS` (default 200). Resiste 429 y 503 transitorios sin fallar la request.

### Observabilidad y operaciones

- **Audit log** (`GET /v1/admin/audit`): cada request queda registrada en SQLite con `ts`, `api_key_id`, `ip`, `method`, `path`, `status`, `latency_ms`. Filtra por rango de tiempo, key e id.
- **Backup endpoint** (`POST /v1/admin/backup`): dispara snapshot WAL y retorna `{ "ok": true, "offset": N }`. Ambos requieren rol `admin`.
- **Bench CI**: `cargo bench --no-run` como job en GitHub Actions — bloquea merges que rompen compilación de benchmarks.
- **RBAC tests** (`tests/auth_rbac.rs`): 7 tests de integración que cubren 401/403 en todas las combinaciones de token ausente, inválido, revocado y rol `user` vs `admin`.

### NS-Mem — Memoria de agentes más inteligente

- **Deduplicación de facts**: el consolidador verifica similitud semántica (cosine ≥ 0.95) antes de insertar. Facts redundantes extraídos de eventos distintos no se acumulan.
- **Decay exponencial** (`memory_decay_enabled`): campo `decay_score` en cada fact semántico. Decae con semivida configurable (`memory_decay_half_life_days`, default 30d). Facts por debajo del umbral se archivan automáticamente.
- **Detección de contradicciones**: al sobrescribir un fact (`upsert_fact`), si la similitud semántica entre el contenido viejo y el nuevo es < 0.55, se crea una arista `Contradicts` en lugar de `Supersedes` y se emite log de contradicción detectada.

## v2.1.0 (2026-04-22)

### Added
- **Typed `MetadataFilter` system** (`src/vector/filter.rs`): composable filter tree replacing the flat `{"field": "value"}` object.
  - Operators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `any_of`, `contains`, `starts_with`, `exists`.
  - Logical combinators: `and`, `or`, `not` (arbitrarily nested).
  - New `filter` field on `SearchOptions` (typed); legacy `filters` field kept for backward compatibility — both are merged with AND when both are present.
- **`any_of` operator** for array-valued metadata fields: returns true when the field (JSON array) contains at least one of the query values. Example: `tax_system: ["suitetax","legacy"]` matches `any_of: ["suitetax"]`. Supports multi-system queries (`any_of: ["suitetax","v3"]`).
- **Keyword index extended to array fields**: `add_meta_to_index` now indexes each string element of array-valued fields individually, enabling the keyword fast-path for `any_of` and `eq` queries on array fields without a full scan.
- **Keyword index fast path for `any_of`**: resolves candidate ID sets directly from the index (union of all query values), same O(1) lookup as `eq`.
- **SQL translation for hub pre-filtering** (`to_sql_where`): `any_of` maps to `EXISTS (SELECT 1 FROM json_each(metadata, '$.field') WHERE value IN (...))`.
- **`from_legacy()`**: automatic conversion of old flat-object filters to typed `And([Eq(...)])` — no breaking change for existing API clients.

### Internal
- Removed `matches_filters()` free function and `Collection::keyword_candidates()` method; replaced by `filter::evaluate_filter()` and `filter::index_candidates()`.
- 7 new tests for `any_of` covering: single-value match, multi-value OR semantics, scalar-field non-match, missing field, keyword index fast path, AND combination, SQL translation.

---

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
