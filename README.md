# RustKissVDB

**RustKissVDB** es una **base de datos multimodelo, local-first y event-sourced**, expuesta como un **servicio HTTP**.
Combina **Key-Value con revisiones**, **Event Log con snapshots**, **Vector Store (HNSW)**, **Document Store** y **SQLite embebido**, con **streaming de cambios vía SSE**.

> Filosofía: **KISS**, almacenamiento explícito, cero magia oculta, recuperación determinística.

---

## Características principales

- ✅ **Event-sourced storage**
  - WAL segmentado (`events-*.log`)
  - Reproducción determinística del estado
  - Snapshots para fast-boot

- ✅ **Key-Value Store**
  - Revisiones (`revision: u64`)
  - CAS (`if_revision`)
  - TTL / expiración
  - Persistencia en `redb`

- ✅ **Vector Database**
  - Colecciones por dimensión y métrica
  - Índices **HNSW** (`hnsw_rs`)
  - Persistencia binaria por colección

- ✅ **Document Store**
  - Documentos JSON schemaless
  - Revisiones por documento

- ✅ **Event Streaming (CDC)**
  - SSE con replay + tail
  - Filtros por tipo, key prefix o colección

- ✅ **SQL embebido**
  - SQLite (`rusqlite`, bundled)
  - SELECT / DDL / DML vía API

- ✅ **Single-node, local-first**
  - Sin clustering
  - Sin dependencias externas

---

## Arquitectura de alto nivel

```text
                   ┌─────────────┐
                   │  HTTP API   │  (Axum)
                   └─────┬───────┘
                         │
              ┌──────────┴──────────┐
              │     Engine Core     │
              │─────────────────────│
              │ Event Log (WAL)     │
              │ State Materializer  │
              │ Vector Engine       │
              │ Doc Store           │
              │ SQLite Adapter      │
              └──────────┬──────────┘
                         │
     ┌───────────────────┴───────────────────┐
     │            Storage Layer              │
     │───────────────────────────────────────│
     │ events-XXXX.log   → WAL segmentado    │
     │ snapshot.json     → Snapshot          │
     │ state.redb        → KV materializado  │
     │ vectors/*         → Vector segments   │
     └───────────────────────────────────────┘
```

---

## Layout de datos en disco

```text
data/
├─ events-003605.log
├─ events-003606.log
├─ events-003607.log
├─ ...
├─ snapshot.json
├─ state.redb
└─ vectors/
   ├─ collection_a/
   │  ├─ manifest.json
   │  └─ vectors.bin
   └─ collection_b/
      ├─ manifest.json
      └─ vectors.bin
```

### Significado

| Archivo / carpeta         | Propósito                                 |
| ------------------------- | ----------------------------------------- |
| `events-*.log`            | WAL append-only, fuente de verdad         |
| `snapshot.json`           | Estado materializado para recovery rápido |
| `state.redb`              | KV store persistente (redb)               |
| `vectors/*/manifest.json` | Metadata de colección vectorial           |
| `vectors/*/vectors.bin`   | Datos binarios + HNSW                     |

---

## Modelos de datos soportados

### 1. Key-Value Store

- `key: string`
- `value: any`
- `revision: u64`
- `ttl_ms`
- CAS con `if_revision`

### 2. Event Store

- Append-only
- Offset incremental
- Replay desde offset arbitrario

### 3. Vector Store

- Métricas: `cosine`, `dot`
- Índice HNSW
- Top-K search
- Batch upsert / delete

### 4. Document Store

- JSON schemaless
- Colecciones + ID
- Revisión por documento

### 5. SQL

- SQLite embebido
- Consultas parametrizadas
- DDL/DML controlado

---

## API

- OpenAPI 3.0: [`docs/openapi.yaml`](docs/openapi.yaml)
- Prefijo: `/v1/*`
- Autenticación: Bearer token
- SSE: `text/event-stream`

Ejemplos:

- `/v1/state/{key}`
- `/v1/vector/{collection}/search`
- `/v1/stream`
- `/v1/doc/{collection}/{id}`
- `/v1/sql/query`

---

## Estructura del código

```text
src/
├─ api/            → HTTP handlers (Axum)
├─ engine/
│  ├─ events.rs    → WAL + offsets
│  ├─ persist.rs  → snapshots
│  ├─ state.rs    → KV materializer
│  └─ state_db.rs → redb adapter
├─ vector/
│  ├─ mod.rs
│  └─ persist.rs  → HNSW + binarios
├─ docstore/       → Document store
├─ sqlite/         → SQLite adapter
├─ bin/
│  └─ bench.rs
├─ config.rs
├─ lib.rs
└─ main.rs
```

---

## Dependencias clave

- **Axum** → HTTP API
- **Tokio** → Async runtime
- **redb** → KV persistente
- **hnsw_rs** → Vector indexing
- **rusqlite (bundled)** → SQL embebido
- **bincode** → Serialización binaria
- **SSE (async-stream)** → CDC

---

## Filosofía de diseño

- ✔ Single-node
- ✔ Determinista
- ✔ Event-sourced
- ✔ Observabilidad explícita
- ✔ Persistencia clara (archivos visibles)
- ❌ No clustering
- ❌ No sharding
- ❌ No consenso distribuido

---

## Casos de uso ideales

- RAG local / privado
- Memoria de agentes
- Sistemas offline-first
- Prototipos de DB engines
- Investigación en arquitecturas event-sourced
- Sustituto ligero de Redis + Vector DB + SQLite

---

## Estado del proyecto

- Versión: **0.1.1**
- Estado: **Activo / experimental**
- Enfoque: Correctitud, claridad, KISS
