<!-- RustKissVDB: the Rust convergent data engine powering Luma (vector search + KV + SQL + events in a single binary). -->
# Luma: The Convergent Data Platform

[![CI](https://github.com/Jairodaniel-17/rust-kiss-vdb/actions/workflows/ci.yml/badge.svg)](https://github.com/Jairodaniel-17/rust-kiss-vdb/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/Jairodaniel-17/rust-kiss-vdb?label=release&sort=semver)](https://github.com/Jairodaniel-17/rust-kiss-vdb/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Python SDK](https://img.shields.io/badge/python-luma--vdb-3775A9.svg)](sdk/)
[![TS SDK](https://img.shields.io/badge/npm-luma--vdb-CB3837.svg)](sdk/typescript/)

**Luma** (Cargo crate `luma`, version **4.24.0**, *powered by RustKissVDB*) is not just a vector database. It is a **Convergent Data Engine** written in **Rust** that unifies, in a **single binary** (`luma`), the primitives a modern AI application needs:

- **Vector search** (ANN) with swappable HNSW / IVF-FLAT-Q8 / DiskANN indexes.
- **Key-value state** (KV) with TTL, compare-and-swap and secondary indexes.
- **Embedded relational SQL** (SQLite in WAL mode), with an optional remote libSQL/Turso backend.
- **JSON document store** and R2-style binary **object storage**.
- Durable **queues** and on-the-fly transformed **images**.
- **Pub/sub event bus** with SSE streaming.
- **NS-Mem**: a memory layer for autonomous agents (episodic, semantic, procedural and working).
- **Enterprise layer**: accounts/organizations, roles, Argon2id login, audit trail, backups, encryption at rest, and a **React admin panel embedded in the binary itself**.
- **Official SDKs**: Python (async + sync), TypeScript/JS, and a **LangChain** integration.

Everything runs in one process, eliminating network latency between subsystems and reducing deployment to a single executable.

*Léelo en [español](README.md).*

---

## Table of contents

- [Installation](#installation)
- [60-second quickstart](#quickstart)
- [Official SDKs](#sdks)
- [Why Luma?](#why-luma)
- [Platform surface](#surface)
- [Measured performance](#performance)
- [Architecture by module](#architecture)
- [Enterprise layer](#enterprise)
- [API levels](#api)
- [Embeddings (BYOM)](#embeddings)
- [Configuration](#configuration)
- [Key technologies](#tech)
- [On-disk layout](#layout)
- [Current project status](#status)
- [Documentation](#documentation)
- [License](#license)

---

<a id="installation"></a>

## 📦 Installation

### Prebuilt binary (fastest path)

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.sh | bash

# Specific version or custom destination
curl -fsSL .../install.sh | bash -s -- --version v4.24.0 --dest ~/.local/bin
```

```powershell
# Windows
irm https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.ps1 | iex
```

The scripts download the GitHub release binary for your platform (Linux, Windows, macOS) and place it on your `PATH`.

### Docker

```bash
# docker-compose (includes a persistent volume at ./data_storage)
LUMA_API_KEY=my-key docker compose up -d

# or a direct image
docker build -t luma:latest .
docker run -p 1234:1234 -v $PWD/data_storage:/data \
  -e DATA_DIR=/data -e LUMA_API_KEY=my-key -e LUMA_MASTER_KEY=strong-key \
  luma:latest
```

Two Dockerfiles are provided: `Dockerfile` (glibc) and `Dockerfile.musl` (static, minimal image). `docker-compose.yml` already exposes the rate limiting, TLS, embeddings and backup variables.

> ⚠️ **Durability requires mounting a persistent volume at `DATA_DIR`.** Without it, the WAL and vectors live in the container's ephemeral layer.

### From source

Requirements: **Rust 1.88+** (edition 2021). SQLite is *bundled* — no external install needed.

```bash
cargo build --release

# (Optional) rebuild the admin panel and embed it into ui/dist
cd admin-ui && npm ci && npm run build && cd ..
```

---

<a id="quickstart"></a>

## ⚡ 60-second quickstart

```bash
# 1. Start (serves API + panel at http://127.0.0.1:1234/)
LUMA_MASTER_KEY="strong-secret-key" LUMA_API_KEY="my-api-key" \
  ./target/release/luma serve
```

```bash
# 2. Create a vector collection (dimension + metric)
curl -X POST localhost:1234/v1/vector/docs \
  -H 'authorization: Bearer my-api-key' -H 'content-type: application/json' \
  -d '{"dim":4,"metric":"cosine"}'

# 3. Insert a vector with metadata
curl -X POST localhost:1234/v1/vector/docs/upsert \
  -H 'authorization: Bearer my-api-key' -H 'content-type: application/json' \
  -d '{"id":"doc-1","vector":[0.1,0.2,0.3,0.4],"meta":{"kind":"contract","year":2024}}'

# 4. Search the k nearest neighbours
curl -X POST localhost:1234/v1/vector/docs/search \
  -H 'authorization: Bearer my-api-key' -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4],"k":5}'

# 5. Search with a typed metadata filter
curl -X POST localhost:1234/v1/vector/docs/search \
  -H 'authorization: Bearer my-api-key' -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4],"k":5,
       "options":{"filter":{"eq":{"field":"kind","value":"contract"}}}}'
```

### Admin panel

```bash
# Create your organization and sign in
curl -X POST localhost:1234/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"org_name":"Acme","email":"owner@acme.com","password":"a-strong-password"}'
```

Open `http://127.0.0.1:1234/` for the React panel, or `http://127.0.0.1:1234/docs` for the interactive API docs (Scalar).

> **Production:** always set `LUMA_MASTER_KEY` (encryption) and `LUMA_API_KEY` (bootstrap). Without `LUMA_MASTER_KEY` a well-known development key is used and the server warns about it in the logs. The default port is **1234** bound to `127.0.0.1`; environment variables override `luma.toml`.

### Binary subcommands

| Subcommand | Description |
| :--- | :--- |
| `luma serve` | Starts the HTTP server (default when no subcommand is given). |
| `luma vacuum --collection <name>` | Compacts a vector collection. |
| `luma diskann build …` / `tune …` / `status <collection>` | Builds, tunes or inspects a DiskANN graph. |
| `luma backup` | Produces a consistent backup (SQLite + snapshot + WAL). |
| `luma restore <path>` | Restores from a backup directory. |

---

<a id="sdks"></a>

## 🐍 Official SDKs

Three clients maintained in this repo, all against the same HTTP API.

### Python — `sdk/` (package `luma-vdb`)

Async and sync, `py.typed`. Sub-clients: `vector`, `state`, `doc`, `admin`, `auth`, `stream`, `config`, `hub(ns)`, `memory(ns)`, `meta(c)`, `diskann(c)`.

```python
from luma import Luma          # async
from luma import SyncLuma      # sync

luma = Luma("http://localhost:1234", api_key="my-api-key")
await luma.vector.acreate("embeddings", dim=1536)
await luma.vector.aupsert("embeddings", "doc-1", [0.1] * 1536, meta={"kind": "contract"})
hits = await luma.vector.asearch("embeddings", [0.1] * 1536, k=5)

# Agent memory (NS-Mem)
mem = luma.memory("my-agent")
await mem.aingest_event(text="The user prefers short answers")
recall = await mem.aquery(text="how do they like to be answered?")
```

```python
with SyncLuma("http://localhost:1234", api_key="my-api-key") as db:
    db.vector.create("embeddings", dim=1536)
    hits = db.vector.search("embeddings", [0.1] * 1536, k=5)
```

### TypeScript / JavaScript — `sdk/typescript/` (package `luma-vdb`)

Node 18+ (uses native `fetch`). Also works in browsers and edge runtimes (Cloudflare Workers, Deno, Bun).

```bash
npm install luma-vdb
```

```typescript
import { LumaClient } from 'luma-vdb';

const client = new LumaClient({ baseUrl: 'http://localhost:1234', apiKey: 'my-api-key' });

await client.vector.createCollection('docs', 384, 'cosine');
await client.vector.upsert('docs', 'item-1', vector, { category: 'tech' });
await client.vector.upsertBatch('docs', [
  { id: 'a', vector: [0.1, 0.2], meta: { tag: 'ai' } },
]);
```

### LangChain — `sdk/langchain_luma/`

`LumaVectorStore` implements the `langchain_core` `VectorStore` interface, including **MMR** (maximal marginal relevance). It creates the collection on its own if it does not exist.

```python
from langchain_luma import LumaVectorStore

store = LumaVectorStore(
    url="http://localhost:1234", api_key="my-api-key",
    collection="rag", embedding=my_embedding, dim=1536,
)
store.add_texts(["...", "..."])
docs = store.max_marginal_relevance_search("my question", k=4)
```

Full Python SDK guide: [`docs/empezar/SDK_PYTHON.md`](docs/empezar/SDK_PYTHON.md).

---

<a id="why-luma"></a>

## 🚀 Why Luma?

The premise is simple: **AI needs more than vectors.** Where the traditional architecture fragments the stack (PostgreSQL for data, Redis for cache/queues, a separate service for vectors), Luma converges those primitives into one Rust binary, with memory safety, Tokio-based concurrency and zero internal latency between engines.

---

<a id="surface"></a>

## 🧰 Platform surface: what each primitive replaces

Luma is not only a vector engine — it is a platform services layer. Every primitive is mounted in the router today and covered by tests:

| Primitive | Endpoint | Replaces | Key semantics |
|---|---|---|---|
| **Object storage** | `/v1/blob/:bucket/:key` | S3 / R2 | buckets and binary objects |
| **KV** | `/v1/state` | Redis (data) / DynamoDB | TTL, revision compare-and-swap, secondary indexes, 16 shards |
| **Durable queues** | `/v1/queue/:queue` | SQS / Cloudflare Queues | at-least-once, visibility timeout, attempts, disk-backed |
| **Event bus** | `/v1/stream` (SSE) | SNS / EventBridge | pub/sub with WAL offsets, replay via `since=` |
| **Images** | `/v1/image/:bucket/:key` | CloudFront + image Lambda | on-the-fly transformation |
| **Document store** | `/v1/doc/:collection` | MongoDB / DocumentDB | JSON with filter-based find |
| **Vectors + hybrid RAG** | `/v1/vector`, `/v1/db` | Qdrant / pgvector / OpenSearch | HNSW · IVF-FLAT-Q8 · DiskANN, rerank, text search |
| **Agent memory** | `/v1/memory` (NS-Mem) | — (no managed equivalent exists) | typed graph, versioned beliefs, contradictions |
| **SQL** | embedded | SQLite WAL (or remote libSQL/Turso) | lightweight relational; does not aim to replace PostgreSQL |
| **Accounts and access control** | `/v1/auth`, `/v1/admin` | Cognito (partial) | orgs, roles, Argon2id, audit, per-organization isolation |

All primitives share the checksummed segmented WAL, snapshots, backups, encryption at rest and per-organization multi-tenant isolation.

> **Product roadmap:** the hardening master plan — verified durability, replication/WAL shipping, S3-compatible API, PostgreSQL CDC connector, operability and GA criteria — lives in [`docs/SPEC-producto.md`](docs/SPEC-producto.md). The **Redis protocol (RESP) compatibility** track — letting Celery, arq, redis-py or ioredis point at Luma **without code changes** (`REDIS_URL=redis://luma:6379`) — has its own phased SPEC in [`docs/SPEC-resp.md`](docs/SPEC-resp.md), and is **not implemented yet**. What we protect and from whom: [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md).

---

<a id="performance"></a>

## 📊 Measured performance

Comparison against **Qdrant** and **Milvus**: same machine, same dataset (50k × 768, cosine, k=10), exact brute-force ground truth, all three engines at their factory configuration.

| Engine / mode | Query (qps) | Latency | Recall@10 | RAM |
|---|---:|---:|---:|---:|
| Qdrant (HNSW, ef=64) | 237 | 4.22 ms | 0.416 | 320 MB |
| Milvus (HNSW, ef=64) | 476 | 2.09 ms | 0.086 | 846 MB |
| **Luma — DiskANN** | **515** | **1.94 ms** | 0.056 | **133 MB** |
| **Luma — HNSW** (ef=32) | 199 | 5.02 ms | **0.474** | 406 MB |
| **Luma — HNSW** (ef=128, default) | 99 | 10.08 ms | 0.853 | 464 MB |

Three headlines:

- 🏆 **RAM** — DiskANN runs 50k vectors in **133 MB**; no competitor goes below 320 MB. This is the design goal, and it holds under measurement.
- 🏆 **Query latency** — **1.94 ms / 515 qps** on DiskANN, the fastest of the group.
- 🏆 **Accuracy at equal speed** — at Qdrant's latency (`ef=32`), Luma HNSW delivers **more recall** (0.474 vs 0.416), and scales to 0.947 as `ef` rises.

> **How to read the recall column:** the dataset is uniformly random vectors with no cluster structure — an adversarial case for any ANN (even Qdrant tops out at 0.42). That column measures **the speed↔accuracy trade-off point each engine picks by default**, not the index's absolute quality. With real embeddings all recalls go up. Comparing recall across engines is only meaningful at equivalent latency.

Full methodology, the `ef` curve, the HNSW calibration (3× throughput at identical recall), the ingestion parallelization (293 → 926 vec/s) and what still lags behind: **[`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md)**.

---

<a id="architecture"></a>

## 🏛️ Architecture by module

The server (`src/server.rs`) validates configuration, initializes subsystems and starts the HTTP router (`src/api/mod.rs`) on **axum 0.7 / hyper 1**.

### Core Engine — `src/engine/`
The native high-performance Rust core. It coordinates subsystems, replays the WAL at startup, expires TTLs and publishes every mutation as an event with a monotonic offset (event sourcing).
- **State (KV):** `state.rs` / `state_db.rs`. In-memory JSON value store with per-key TTL and compare-and-swap via `if_revision`; optional persistence backed by **redb**.
- **Event bus:** `events.rs`. Pub/Sub over `tokio::sync::broadcast`; SSE clients receive the live stream plus a "gap" signal when they fall behind the buffer.
- **Persistence:** `persist.rs`. Segmented WAL (`events-XXXXXX.log`, JSON lines) with periodic snapshots (`snapshot.json`); a snapshot triggers WAL rotation and cleanup.
- **Embeddings:** `embeddings.rs`. HTTP client with swappable providers (`none`/`mock`/`ollama`/`openai`/`azure`/`cohere`/`huggingface`), LRU cache, concurrency semaphore and retries with exponential backoff + jitter.
- **Parsing and chunking:** `parser.rs` (PDF/DOCX/images via `pdf-extract`, `docx-rs`, `quick-xml`, `zip`, `image`) and `chunking.rs` to split text before embedding.
- **Hub (`hub.rs`):** the `LumaDatabase` orchestrator (see Level 2).

### Vector engine — `src/vector/`
Vector CRUD and k-NN search with three index strategies, switchable by config (`index_kind`):
- **HNSW** — in-memory approximate ANN (`hnsw_rs`).
- **IVF_FLAT_Q8** *(default)* — inverted file index with 8-bit quantization refinement (`ivf.rs`, `q8.rs`).
- **DiskANN** — on-disk Vamana graph for massive collections (`diskann/`).

Collections are split into segments (~8,192 vectors); the active segment receives upserts and frozen ones are read-only. Vectors persist as binary (`vectors.bin`) with **zero-copy mmap** support (`mmap.rs`) and SIMD optimizations (`simd.rs`). Composable typed filtering lives in `filter.rs`.

### SQL service — `src/sqlite/`
Embedded **SQLite** (`rusqlite` *bundled*) in **WAL** mode, accessed through an **actor pattern** (Tokio MPSC channel) for non-blocking async queries (`actor.rs`, `pool.rs`). If `LIBSQL_URL` is set, SQL is routed to a remote **libSQL/Turso** backend over Hrana on HTTPS (`hrana.rs`). It backs the hub pre-filter, NS-Mem, authentication and the enterprise layer.

### Text search engine — `src/search/`
`SearchEngine` with its own storage (`storage.rs`), grouping (`grouping.rs`) and scoring engine (`engine.rs`), exposed at `/search` and `/search/ingest`.

### HTTP orchestration layer — `src/api/`
**axum** router with authentication (Bearer API key, static keys and session tokens), configurable CORS, timeouts, body size limits, optional rate limiting (`tower_governor`) and optional TLS (`rustls`). Routes are split by domain across `routes_*.rs`. Interactive **Scalar** docs served at `/docs` from the embedded `docs/openapi.yaml`.

### Agent memory (NS-Mem) — `src/memory/`
Memory layer for autonomous agents; see **Level 3** in the API section.

---

<a id="enterprise"></a>

## 🏢 Enterprise layer: multi-tenancy, panel and security

An **additive** "enterprise" layer built on top of the core primitives. Everything lives in the same binary: no Node, no panel source code, no external services at runtime. `AccountsService` and the `sys_*` tables are created *lazily* on first use, as long as SQLite is enabled.

### Accounts, sessions and roles (`src/api/accounts.rs`, `routes_accounts.rs`)
- **Organizations and users** in SQLite (`sys_orgs`, `sys_users`), plus `sys_sessions` for session tokens and `sys_collections` for resource ownership.
- **Email + password login**: passwords are hashed with **Argon2id** (`src/crypto.rs`). Login issues an **opaque session token** (`lums_…`) of which only its SHA-256 hash is stored; 7-day TTL.
- **Roles**: `owner` > `admin` > `member` > `viewer`, integrated with the existing RBAC (`rbac.rs`, levels viewer=10, member=20, admin=30, owner=40). A middleware enforces a minimum role per route.
- **Multi-organization**: a user can belong to several orgs (`/v1/admin/users/:id/orgs`), with invitations (`/v1/admin/orgs/:id/invite`) and per-org member management.

### Per-organization data isolation (`tenant_isolation_middleware`)
Every collection/document/blob is bound to the organization that created it (*first-touch* in `sys_collections`). Another organization trying to reach that name gets a `404` — existence stays hidden across tenants. The hub (`/v1/db`) and NS-Mem (`/v1/memory`) **share a namespace on purpose** and already isolate internally by the token's `tenant_id`, so exclusive ownership is not imposed on them.

### Admin panel (React + Vite + TypeScript)
- Source lives in `admin-ui/` (`App.tsx`, `api.ts`, `main.tsx`); it is built (`npm ci && npm run build`) into `ui/dist/` (a real JS + CSS bundle under `ui/dist/assets/`) and **embedded in the binary** with `rust-embed` (`routes_ui.rs`). Axum serves it at `/` with an SPA *fallback* for client-side routes.
- It covers login/registration, usage dashboard (`/v1/admin/stats`), user and organization management, API keys, the audit log and health status. It uses relative `/v1/*` paths (no hardcoded hosts).
- React escapes content by default and API responses are always JSON, mitigating reflected/stored XSS.

### Backups (`src/backup.rs`)
- **Consistent** copy of SQLite (`VACUUM INTO`) + `snapshot.json` + WAL segments into `backups/<timestamp>/`, with **configurable retention**.
- CLI: `luma backup` and `luma restore <path>`. Optional background task (`backup_enabled`) with `backup_interval_secs`.

### Audit and encryption
- **Access audit** (`src/api/audit.rs`): middleware recording `ts, api_key_id, ip, method, path, status, latency_ms` in SQLite; queryable at `GET /v1/admin/audit`. Business-level ("semantic") audit events (login, user create/delete) go to `sys_audit_events` and are queried at `/v1/admin/audit-events`.
- **Encryption at rest** for sensitive fields with **ChaCha20-Poly1305** (AEAD), master key derived from `LUMA_MASTER_KEY`. Self-describing ciphertext `enc:v1:<b64(nonce||ct)>`.
- **Security headers** on every response: strict `Content-Security-Policy` (no `unsafe-inline` for scripts; jsdelivr allowed for the Scalar docs), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy` and **HSTS**.

Vulnerability reporting policy: [`SECURITY.md`](SECURITY.md). Threat model: [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md).

---

<a id="api"></a>

## 🧭 API levels

The router (`src/api/mod.rs`) mounts the routes below. All require `Authorization: Bearer <api_key|token>` except `register`/`login`, `/v1/health`, `/docs` and the panel assets.

### Level 1: primitive endpoints

Each engine works in isolation, for maximum speed and minimum overhead.

- **Vector** — `/v1/vector/...`: list/create/delete collections and read details; `add`, `upsert`, `upsert_batch`, `update`, `delete`, `delete_batch`, `get`; `search`, `search_batch` (up to 100 queries in parallel via `rayon`), `scroll` (cursor pagination), `rerank` (cosine reordering), `aggregate` (per-field counts); `diskann/build`, `diskann/tune`, `diskann/status`.
- **JSON documents** — `/v1/doc/{collection}/{id}` (`PUT`/`GET`/`DELETE`) and `/v1/doc/{collection}/find`.
- **Collection metadata** — `POST /v1/meta/{collection}/execute`: queries over a collection's metadata (`routes_meta.rs`).
- **Key-value** — `/v1/state/...`: `GET`/`PUT`/`DELETE` by key, `batch_put`, indexes (`indexes`, `index/{field}/{value}`), listing and TTL/CAS.
- **Object storage (R2-like)** — `/v1/blob/{bucket}/{key}` (`PUT`/`GET`/`DELETE`) and per-bucket listing. Atomic writes, hardened against path traversal.
- **Queues** — `/v1/queue/{queue}` (enqueue and stats), `/receive` (*at-least-once* delivery with *visibility timeout*), `DELETE /{id}` (ack).
- **Images** — `GET /v1/image/{bucket}/{key}?w=&h=&format=&quality=`: resize (Lanczos3) + convert (`png`/`jpeg`) over objects already in the blob store.
- **Events** — `GET /v1/events` and `GET /v1/stream` (live SSE with gap signal).
- **Text search** — `POST /search` and `POST /search/ingest` (the `src/search/` engine).

> Relational SQL is used **internally** (hub pre-filter, NS-Mem, auth, audit) and optionally against a remote libSQL/Turso backend; no raw SQL query route is exposed in the current router.

### Level 2: LumaDatabase Hub (hybrid RAG) — `/v1/db/{namespace}`

The `LumaDatabase` orchestrator (`src/engine/hub.rs`) fuses the engines: it splits large documents (chunking), connects to the configured embedding model, creates the collection if missing, stores vectors and persists metadata in SQLite transactionally (with *rollback* on I/O failure).

- **`POST /v1/db/{namespace}/ingest`** — ingests `{ id, text, metadata }`: chunking → embedding → upsert.
- **`POST /v1/db/{namespace}/search`** — hybrid search: strict **SQL pre-filter** (100% precision) before the vector phase, then chunk collapse and hydrated parent document return.

```json
{
  "query": "clause about the rental price",
  "limit": 5,
  "sql_filter": "json_extract(metadata, '$.kind') = 'rental' AND json_extract(metadata, '$.year') = 2024"
}
```

### Level 3: NS-Mem — agent memory — `/v1/memory/{namespace}`

A complete memory layer for autonomous agents (`src/memory/`), built on the convergent stack.

| Type | Storage | Description |
| :--- | :--- | :--- |
| **episodic** | Vector + SQLite | Concrete events and interactions indexed for semantic recall |
| **semantic** | Vector + SQLite | Stable facts and preferences, promoted from episodic via LLM |
| **procedural** | SQLite (DAG) | Flows with nodes, typed edges and constraint evaluation |
| **working** | KV + TTL | Ephemeral session context, expires automatically |

**Consolidation pipeline**: `ingest_event` → fact extraction (LLM or local heuristic) → `semantic` (`active` if confidence ≥ 0.85, otherwise `draft`), creating a `TriggeredBy` edge (episodic → semantic).

**Recall (semantic walk)**: K-NN seeds → BFS expansion over typed edges → ranking by `cosine × edge_factor × (1 + PageRank centrality)` → filter archived → top-k.

**Endpoints**:
- `POST ingest_event` · `POST upsert_fact` · `POST upsert_procedure`
- `POST query` (hybrid recall) · `POST next_step` (next valid DAG node)
- `GET timeline/{entity_id}`
- `POST edges` · `GET edges/{memory_id}` · `POST edges/{edge_id}/delete`
- `GET beliefs/{fact_key}/history` · `POST graph/centrality`

Also: fact deduplication (cosine ≥ 0.95), opt-in exponential decay (`memory_decay_enabled`) and contradiction detection (a `Contradicts` edge when old↔new similarity < 0.55). LLM providers: `none`, `mock`, `openai`, `ollama`. See [`docs/integrar/NS_MEM.md`](docs/integrar/NS_MEM.md).

### Administration, auth and health

- **Health and metrics**: `GET /v1/health`, `GET /v1/metrics` (p50/p95/p99 percentiles).
- **Accounts**: `POST /v1/auth/register` · `login` · `logout` · `refresh`; `GET /v1/auth/sessions` (active sessions).
- **Orgs**: `/v1/admin/orgs` (list/create), `/v1/admin/orgs/:id` (detail/update), `/members`, `/invite`, `/members/:user_id`.
- **Users**: `/v1/admin/users` (list/create), `/v1/admin/users/:id`, `/:id/role`, `/:id/orgs`.
- **API keys**: `/v1/auth/keys` (list/create), `/keys/:id` (revoke), `/keys/:id/role`.
- **RBAC**: `/v1/auth/roles` (list/create), `/roles/:id`, `/roles/:id/permissions`, `POST /roles/check`.
- **Operations**: `POST /v1/admin/backup` (triggers a snapshot), `GET /v1/admin/audit` (filterable access log), `GET /v1/admin/stats`, `GET /v1/admin/audit-events` — all require the `admin` role.
- **Hot configuration**: `GET`/`PUT /v1/config`, `POST /v1/config/embedding/probe` (verifies the embedding provider responds).

Full reference: [`docs/integrar/API.md`](docs/integrar/API.md) and the OpenAPI spec at [`docs/openapi.yaml`](docs/openapi.yaml) (served at `/docs`).

**All the documentation, arranged by what you are trying to do — getting started, operating, integrating, reference — is in [`docs/README.md`](docs/README.md).**

---

<a id="embeddings"></a>

## 🔌 Embeddings (BYOM — Bring Your Own Model)

To avoid bloating the binary with heavy C++ libraries, Luma uses a lightweight HTTP client with automatic retries. It supports 6 providers besides `none` (no server-side embedding, the default):

| Provider | Variables | Notes |
| :--- | :--- | :--- |
| `ollama` | `EMBEDDING_URL`, `EMBEDDING_MODEL` | Local, no API key |
| `openai` | `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | Batching ≤ 96 |
| `azure` | `EMBEDDING_AZURE_API_BASE`, `EMBEDDING_AZURE_DEPLOYMENT` | `api-key` header |
| `cohere` | `EMBEDDING_API_KEY`, `EMBEDDING_COHERE_INPUT_TYPE` | `search_document` / `search_query` |
| `huggingface` | `EMBEDDING_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | Inference API |
| `mock` | `EMBEDDING_DIM` | Tests/CI without network |

Retry with exponential backoff + jitter: `EMBEDDING_RETRY_ATTEMPTS` (default 3), `EMBEDDING_RETRY_INITIAL_MS` (default 200). You can validate the connection with `POST /v1/config/embedding/probe`.

---

<a id="configuration"></a>

## ⚙️ Configuration

`luma.toml` at the repo root (auto-generated if missing). Environment variables **override** the TOML. Source: `src/config.rs`.

| Section | Keys |
| :--- | :--- |
| Server | `port` (1234), `bind_addr` (127.0.0.1), `api_key` |
| Storage | `data_dir`, `snapshot_interval_secs`, `wal_segment_max_bytes` |
| Vector | `index_kind` (**`IVF_FLAT_Q8` by default** · `HNSW` · `DiskANN`), `max_vector_dim`, `simd_enabled`, `HNSW_SEARCH_EF` (128) |
| IVF | `ivf_clusters`, `ivf_nprobe`, `q8_refine_topk` |
| DiskANN | `diskann_max_degree`, `diskann_build_threads` |
| Embeddings | `embedding_provider`, `embedding_model`, `embedding_dim`, `embedding_retry_*` |
| Search | `pre_filter_threshold` (10,000) |
| NS-Mem | `memory_consolidation_enabled`, `memory_working_ttl_secs`, `memory_fact_promotion_threshold` (0.85), `llm_provider` |
| Graph | `memory_walk_max_hops` (2), `memory_walk_min_similarity` (0.65), `memory_centrality_enabled` |
| Decay | `memory_decay_enabled` (false), `memory_decay_half_life_days` (30), `memory_decay_archive_threshold` (0.1) |
| Backups | `backup_enabled`, `backup_dir`, `backup_interval_secs`, `backup_retention` |
| Operations | `rate_limit_rps` (0 = off), `rate_limit_burst`, `TLS_CERT_PATH`, `TLS_KEY_PATH`, `LIBSQL_URL` |

It can be read and updated at runtime via `GET`/`PUT /v1/config`. Full detail: [`docs/operar/CONFIG.md`](docs/operar/CONFIG.md).

---

<a id="tech"></a>

## 🧰 Key technologies

| Component | Crate | Role |
| :--- | :--- | :--- |
| Async runtime / HTTP | `tokio`, `axum` 0.7, `hyper` 1 | Non-blocking I/O and router |
| Core KV | `redb` | Pure-Rust ACID persistence |
| Relational | `rusqlite` (SQLite bundled) | Embedded SQL in WAL mode |
| Vectors | `hnsw_rs` + custom IVF/DiskANN | Swappable ANN indexes |
| Embedded panel | `rust-embed` + React/Vite | SPA served from `ui/dist` |
| Security | `argon2`, `chacha20poly1305`, `rustls` | Hashing, encryption, TLS |
| Parsing | `pdf-extract`, `docx-rs`, `image` | Rich format ingestion |

---

<a id="layout"></a>

## 🗂️ On-disk layout

```
data/
├── events-000001.log          # Segmented WAL (JSON lines)
├── snapshot.json              # Latest state snapshot
├── vectors/<collection>/      # manifest.json, vectors.bin (mmap), diskann/
└── sqlite/rustkiss.db         # Relational + auth + docstore + NS-Mem and sys_* tables
backups/<timestamp>/           # Backups (VACUUM INTO + snapshot + WAL)
```

---

<a id="status"></a>

## ✅ Current project status

Implemented and verifiable in today's code (crate `luma` v4.24.0):

- **Convergent core**: vector engine (HNSW / IVF-FLAT-Q8 / DiskANN), KV with TTL/CAS, segmented WAL + snapshots, embedded SQLite via actor, SSE event bus, hybrid RAG hub and text search engine. All mounted in the router and covered by `tests/`.
- **Object storage, queues and images**: R2 + Queues + Images style primitives already mounted at `/v1/blob`, `/v1/queue`, `/v1/image`.
- **NS-Mem**: agent memory with typed graph, semantic walk BFS, PageRank, belief versioning, deduplication and contradiction detection. Opt-in decay.
- **Enterprise layer**: accounts/orgs/users, owner/admin/member/viewer roles, Argon2id login + session tokens, per-organization multi-tenant isolation, audit, encryption at rest, backups (CLI + background task) and a **React admin panel genuinely built and embedded** (`admin-ui/` → `ui/dist`).
- **SDKs**: Python client (async + sync, `py.typed`), TypeScript client and `LumaVectorStore` for LangChain, all under `sdk/`.
- **Operations**: optional TLS, opt-in rate limiting, configurable CORS, timeouts, security headers, installers for Linux/macOS/Windows, two Dockerfiles + compose, and Scalar docs at `/docs`.

Honesty notes:

- Several heavyweight capabilities ship **disabled by default** (`embedding_provider = "none"`, memory consolidation/decay/centrality, `rate_limit_rps = 0`) — that is a development profile; in production they are enabled via config/environment.
- The remote **libSQL/Turso** backend only activates when `LIBSQL_URL` is set; otherwise local SQLite is used.
- Durability depends on mounting a persistent volume for `data_dir` when running in a container.
- The **Qdrant/Milvus comparison** was produced with ad-hoc scripts that are **not versioned** in this repo: the figures are the ones observed, but they cannot currently be re-run with a repository command. The *internal* benchmarks (`src/bin/bench.rs`) are reproducible.
- **Redis protocol (RESP) compatibility** is specified in `docs/SPEC-resp.md` but **not implemented** yet.

---

<a id="documentation"></a>

## 📚 Documentation

Documentation is currently written in Spanish.

| Document | Contents |
| :--- | :--- |
| [`docs/integrar/API.md`](docs/integrar/API.md) | Endpoint reference |
| [`docs/openapi.yaml`](docs/openapi.yaml) | OpenAPI spec (served at `/docs`) |
| [`docs/referencia/ARCHITECTURE.md`](docs/referencia/ARCHITECTURE.md) | Internal architecture |
| [`docs/operar/CONFIG.md`](docs/operar/CONFIG.md) | Every configuration key |
| [`docs/operar/CLI.md`](docs/operar/CLI.md) | Binary subcommands |
| [`docs/referencia/DATA_MODELS.md`](docs/referencia/DATA_MODELS.md) | Data models and schemas |
| [`docs/referencia/VECTOR_STORAGE.md`](docs/referencia/VECTOR_STORAGE.md) | Segments, mmap, quantization |
| [`docs/integrar/RESP.md`](docs/integrar/RESP.md) | Redis protocol compatibility: commands, divergences, how to validate |
| [`docs/integrar/NS_MEM.md`](docs/integrar/NS_MEM.md) | Agent memory (full API) |
| [`docs/empezar/SDK_PYTHON.md`](docs/empezar/SDK_PYTHON.md) | Python SDK guide |
| [`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md) | Qdrant/Milvus comparison + internal benchmarks |
| [`docs/referencia/BENCH.md`](docs/referencia/BENCH.md) | How to run the bench binary |
| [`docs/empezar/FEATURES.md`](docs/empezar/FEATURES.md) | Feature inventory |
| [`docs/operar/SECURITY.md`](docs/operar/SECURITY.md) · [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md) | Security and threat model |
| [`docs/operar/PROD_READINESS.md`](docs/operar/PROD_READINESS.md) | Production checklist |
| [`docs/PLAN-MAESTRO.md`](docs/PLAN-MAESTRO.md) | **Unified execution plan** (blocks, order, status) |
| [`docs/SPEC-producto.md`](docs/SPEC-producto.md) · [`docs/SPEC-resp.md`](docs/SPEC-resp.md) · [`docs/SPEC-roadmap.md`](docs/SPEC-roadmap.md) | Source SPECs (detail and acceptance criteria) |
| [`docs/referencia/CHANGELOG.md`](docs/referencia/CHANGELOG.md) | Version history |
| [`docs/empezar/DEMO.md`](docs/empezar/DEMO.md) | Demo script |

---

<a id="license"></a>

## 📄 License

[MIT](LICENSE) © Luma contributors.

Vulnerability reports: see [`SECURITY.md`](SECURITY.md).

---

## 🏁 Closing

Luma redefines the AI backend through **convergence**: it orchestrates best-in-class engines (vector indexes, SQLite, redb) plus a complete enterprise layer into one cohesive platform and a single binary.

> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**
