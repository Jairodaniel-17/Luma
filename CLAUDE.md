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

**`src/engine/embeddings.rs`** — Embedding provider abstraction supporting Ollama, OpenAI, Azure OpenAI, Cohere, HuggingFace, and Mock (for tests/CI). Includes LRU cache keyed by `provider::model::dim::text`, bounded concurrency semaphore, and configurable exponential-backoff retry with jitter.

**`src/memory/decay.rs`** — Background decay task for NS-Mem semantic facts. Applies exponential decay (`exp(-λt)`) to `decay_score` field; archives facts below threshold automatically. Opt-in via `memory_decay_enabled`.

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
- **Embeddings**: `embedding_provider` (`none` | `ollama` | `openai` | `azure` | `cohere` | `huggingface` | `mock`), `embedding_model`, `embedding_dim`, `embedding_retry_attempts` (default 3), `embedding_retry_initial_ms` (default 200)
- **Azure OpenAI**: `embedding_azure_api_base`, `embedding_azure_deployment`, `embedding_azure_api_version` (default `2024-02-01`)
- **Cohere**: `embedding_cohere_input_type` (`search_document` | `search_query`)
- **Search**: `pre_filter_threshold` (default 10 000) — filtered candidate sets ≤ this size use brute-force instead of HNSW + post-filter
- **Memory / NS-Mem**: `memory_consolidation_enabled`, `memory_working_ttl_secs`, `memory_default_limit`, `memory_fact_promotion_threshold` (default 0.85), `llm_provider` (`none` | `mock` | `openai` | `ollama`), `llm_model`, `llm_url`, `llm_api_key`
- **Graph / Semantic Walk**: `memory_walk_max_hops` (default 2), `memory_walk_min_similarity` (default 0.65), `memory_walk_max_nodes` (default 40), `memory_centrality_enabled` (default true), `memory_centrality_update_interval_secs` (default 300)
- **NS-Mem Decay**: `memory_decay_enabled` (default false), `memory_decay_half_life_days` (default 30.0), `memory_decay_archive_threshold` (default 0.1), `memory_decay_interval_secs` (default 3600)

Config source: `src/config.rs`. Environment variables override TOML values.

## CI

CI runs: rustfmt check → clippy (strict, `-D warnings`) → tests → `cargo-audit` → `cargo-deny` → MSRV check (1.88) → `cargo bench --no-run` (bench compilation gate). Releases trigger on `v*` tags and produce cross-platform binaries (Linux, Windows, macOS).

Use the `mock` embedding provider in tests to avoid external service dependencies.

## Work Log

### Done — Enterprise (multi-tenancy, admin panel, seguridad)

Capa empresarial **aditiva** sobre el core existente (API keys/RBAC/audit ya presentes).

- **Cuentas y sesiones** (`src/api/accounts.rs`, `routes_accounts.rs`): tablas `sys_orgs`, `sys_users`, `sys_sessions`, `sys_collections`, `sys_audit_events`. Login email+password con **Argon2id** (`src/crypto.rs`); token de sesión opaco `lums_…` (se persiste solo su hash SHA-256; TTL 7 días). Rutas: `/v1/auth/register|login|logout|refresh`, `/v1/admin/orgs`, `/v1/admin/users`, `/v1/admin/stats`, `/v1/admin/audit-events`. Tablas creadas *lazily* vía `OnceCell` (sin tocar `RouterDeps`; `AccountsService` se construye dentro de `router()` desde `sqlite`).
- **Roles** owner/admin/member/viewer integrados en `rbac.rs` (`role_level`: viewer/readonly=10, member/user=20, admin=30, owner=40). Roles semilla añadidos con herencia por padre.
- **Auth extendido** (`auth.rs`): además de API keys y clave estática, resuelve tokens de sesión → `TenantContext { user_id, tenant_id=org_id, role }`. Se añadió `user_id: Option<String>` a `TenantContext`.
- **Aislamiento por org** (`tenant_isolation_middleware` en `api/mod.rs`): propiedad first-touch de colecciones/doc/blob en `sys_collections`; cross-tenant → 404. Hub (`/v1/db`) y NS-Mem (`/v1/memory`) se excluyen porque ya aíslan internamente por `tenant_id` (ver `tests/multitenant_hub.rs`).
- **Cifrado en reposo** (`src/crypto.rs`): `SecretBox` con **ChaCha20-Poly1305**, clave maestra de `LUMA_MASTER_KEY` (SHA-256). Ciphertext auto-descriptivo `enc:v1:<b64(nonce||ct)>`.
- **Cabeceras de seguridad** (`security_headers`): CSP estricta (script-src 'self', sin unsafe-inline; jsdelivr permitido para Scalar docs), `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, **HSTS**.
- **Respaldos** (`src/backup.rs`): `VACUUM INTO` + snapshot + WAL a `backups/<ts>/` con retención. CLI `luma backup` / `luma restore <path>`; tarea de fondo opt-in (`backup_enabled`). Config nueva en `luma.toml`: `backup_enabled`, `backup_dir`, `backup_interval_secs`, `backup_retention` (con `#[serde(default)]` para compatibilidad).
- **Panel React+Vite+TS** en `admin-ui/`, compilado a `ui/dist` e incrustado con `rust-embed` (`routes_ui.rs` embebe `ui/dist/`, con `spa_fallback`). Páginas: login/registro, dashboard de stats, usuarios, orgs, API keys, auditoría, salud. Rutas relativas `/v1/*` (sin localhost hardcodeado).
- **Tests de seguridad** (`tests/security_enterprise.rs`, `tests/backup_restore.rs`): login/sesión, sesión inválida/revocada → 401, aislamiento cross-org → 404, RBAC viewer → 403, XSS servido como JSON escapado + CSP, inyección SQL neutralizada (params vinculados), cabeceras de seguridad, panel embebido servido, backup↔restore roundtrip. Suite completa verde; clippy `-D warnings` limpio.

### Done — v3.0.0

**Fase 2 — Calidad de resultados y completitud funcional**

- **Batch search** (`POST /v1/vector/{collection}/search_batch`): hasta 100 queries en paralelo vía `rayon::par_iter()`. Devuelve `{ "results": [{ "hits": [...] }] }`.
- **Scroll / cursor API** (`GET /v1/vector/{collection}/scroll?cursor=&limit=&include_vectors=`): paginación lexicográfica por ID con cursor opaco. Permite exportar colecciones completas.
- **Reranking por coseno** (`POST /v1/vector/{collection}/rerank`): recibe `{ "query_text" | "query_vector", "ids": [...] }`, embebe la query (si es texto), recupera vectores almacenados y devuelve IDs reordenados por coseno.
- **Aggregations** (`POST /v1/vector/{collection}/aggregate`): `{ "group_by": "field", "filter": {...}, "limit": 100 }` — cuenta ítems por valor de campo usando `keyword_index`. Soporta filtro tipado con fast path por índice.
- **Nuevos providers de embedding**:
  - `AzureOpenAI`: endpoint propio, `api-key` header, batching ≤ 96.
  - `Cohere`: `/v1/embed` con `input_type` configurable (`search_document` / `search_query`).
  - `HuggingFace`: `pipeline/feature-extraction/{model}`, maneja respuesta `[f32]` y `[[f32]]`.
- **Retry con backoff exponencial + jitter** en `EmbeddingClient`: `embedding_retry_attempts` (default 3), `embedding_retry_initial_ms` (default 200). Jitter basado en `subsec_millis` para evitar thundering herd.
- **Pre-filter threshold configurable** (`pre_filter_threshold`, default 10 000): reemplaza el hardcoded 512. Cuando el conjunto filtrado tiene ≤ threshold candidatos, se usa brute-force en lugar de HNSW + post-filter.
- **Audit log completo** (`src/api/audit.rs`): middleware que registra `ts, api_key_id, ip, method, path, status, latency_ms` en SQLite. `AuditKeyId` propagado desde `auth_middleware` vía extensions. Endpoint `GET /v1/admin/audit?from_ms=&to_ms=&key=&limit=`.
- **Backup endpoint** (`POST /v1/admin/backup`): dispara `engine.force_snapshot()` y retorna `{ "ok": true, "offset": u64 }`. Requiere rol admin.
- **`routes_admin.rs`**: módulo dedicado para handlers admin (backup + audit). Patrón consistente con `routes_*.rs` existentes.
- **`server.rs`** actualizado: inicializa `AuditLog` con `init().await` antes del router; wiring de los 3 nuevos providers de embedding.

**Fase 3 — Robustez operacional**

- **NS-Mem deduplicación de facts** (`src/memory/consolidator.rs`): antes de `upsert_fact()`, el consolidador busca facts semánticamente equivalentes (cosine ≥ 0.95) en la colección semántica del namespace. Si existe un duplicado distinto al fact que se crearía, lo omite y loguea en `DEBUG`.
- **NS-Mem decay** (`src/memory/decay.rs`): campo `decay_score: f32` (default 1.0) en `MemoryRecord` y `memory_records`. Decay exponencial `exp(-ln(2)/half_life_days * elapsed_days)`. Facts por debajo de `archive_threshold` se archivan automáticamente. Background task opt-in (`memory_decay_enabled`). Migración via `ALTER TABLE` en schema.
- **NS-Mem detección de contradicciones** (`src/memory/ingest.rs`): en `upsert_fact()`, si hay un fact existente con el mismo ID, compara embeddings del contenido viejo y nuevo. Si coseno < 0.55, crea arista `Contradicts` en lugar de `Supersedes` y emite `tracing::info!`. Threshold: 0.55 (separación semántica significativa).
- **Bench CI** (`.github/workflows/ci.yml`): nuevo job `bench-compile` — `cargo bench --no-run` — bloquea merges que rompen la compilación de benchmarks sin ejecutarlos en CI.
- **RBAC tests** (`tests/auth_rbac.rs`): 7 tests de integración — token ausente → 401, token inválido → 401, rol `user` → 403 en `list_keys`/`create_key`/`revoke_key`, rol `admin` → 200, key revocada → 401.

### Done — v2.1.0

- **Typed MetadataFilter** (`src/vector/filter.rs`): composable filter tree with `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `any_of`, `contains`, `starts_with`, `exists` and logical `and`/`or`/`not`.
- **`any_of` operator**: array-field membership — `tax_system AnyOf ["suitetax"]` matches `["suitetax","legacy"]`. Useful for multi-tenant documents shared across systems.
- **Keyword index extended**: arrays are now indexed per-element; `any_of` resolves via keyword index fast path (union of sets), no full scan.
- **`SearchOptions.filter`** field (typed, alongside legacy `filters`); `effective_filter()` merges both with AND.
- **SQL hub pre-filter**: `any_of` → `json_each(metadata, '$.field')` for SQLite.
- Legacy `filters: {"field": "value"}` still works — converted transparently via `from_legacy()`.

### Done — v2.0.1

- CVE fix: `rustls-webpki` 0.103.12 → 0.103.13 (RUSTSEC-2026-0104).
- Removed API key via query param (credential leakage in logs).
- Admin role enforcement on `/v1/auth/keys` endpoints.
- `parking_lot::Mutex` in `EmbeddingClient` and `Metrics` (no poison panics).
- Bounded concurrency semaphore for embedding providers; OpenAI batch chunked to ≤ 96 texts.
- `tracing::warn!` on `TriggeredBy` edge failures in consolidator.
- HNSW compaction tombstone ratio default 0.2 → 0.5.
- CI: removed `continue-on-error`, added MSRV (1.88) and `cargo-deny` jobs.
- 20 procedural DAG tests (`tests/ns_mem_procedural.rs`).

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

<!-- tooling:log -->
- 2026-07-19 — Agregada linea final «test» al README y publicada al remoto (archivos clave: README.md)
- 2026-07-19 — Añadida nueva linea final «test» al README y publicada (push) al remoto (archivos clave: README.md)
- 2026-07-19 — Eliminada la última linea «test» del README y publicada (push) al remoto (archivos clave: README.md)
<!-- /tooling:log -->
