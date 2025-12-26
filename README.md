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
  - Colecciones `IVF_FLAT_Q8` por dimensión y métrica
  - KMeans++ + `nprobe` para limitar clusters consultados
  - Cuantización **Q8** persistente con refinamiento opcional `f32`
  - Runs segmentados + `manifest.json` + compactación incremental

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
   │  ├─ centroids.json
   │  ├─ centroids.bin
   │  └─ runs/
   │     ├─ run-000001.log
   │     └─ run-000002.log
   └─ collection_b/
      ├─ manifest.json
      ├─ centroids.*
      └─ runs/
```

### Significado

| Archivo / carpeta         | Propósito                                 |
| ------------------------- | ----------------------------------------- |
| `events-*.log`            | WAL append-only, fuente de verdad         |
| `snapshot.json`           | Estado materializado para recovery rápido |
| `state.redb`              | KV store persistente (redb)               |
| `vectors/*/manifest.json` | Metadata y settings persistidos           |
| `vectors/*/centroids.*`   | Centroides IVF (kmeans++)                 |
| `vectors/*/runs/run-*.log`| Runs segmentados con checksum y compactación incremental |

### Bloque `disk_index` en `manifest.json`

Cada colección persiste el estado de su índice on-disk (DiskANN/Vamana) en el campo:

```json
"disk_index": {
  "kind": "diskann_vamana",
  "version": 1,
  "last_built_ms": 1702843350123,
  "graph_files": ["diskann/graph-1702843350.json"],
  "build_params": {
    "max_degree": 64,
    "build_threads": 8,
    "search_list_size": 128
  }
}
```

Los comandos disponibles desde el `Engine` son:

- `vector_build_disk_index(collection, DiskAnnBuildParams)` → genera/actualiza los archivos declarados y sincroniza el manifest.
- `vector_drop_disk_index(collection)` → borra los archivos listados y limpia el bloque.
- `vector_disk_index_status(collection)` → reporta si existen artefactos en disco (se usa para decidir si se puede servir desde SSD).

---

## Variables de entorno clave

`INDEX_KIND` admite `HNSW`, `IVF_FLAT_Q8` (por defecto) y `DISKANN` (modo experimental con índice en disco).

### Persistencia (runs)

| Variable                            | Descripción                                                                 | Valor por defecto |
| ----------------------------------- | --------------------------------------------------------------------------- | ----------------- |
| `RUN_TARGET_BYTES`                  | Tamaño objetivo de cada run/log de vectores (gira al superarlo).            | `134217728` (128MB) |
| `RUN_RETENTION`                     | Cantidad máxima de runs activos antes de forzar una compactación.          | `8`               |
| `COMPACTION_TRIGGER_TOMBSTONE_RATIO` | Umbral de basura (tombstones / records) que dispara una compactación.       | `0.2`             |
| `COMPACTION_MAX_BYTES_PER_PASS`     | Límite de bytes a reescribir por pasada de compactación incremental.       | `1073741824` (1GB) |

Valores sugeridos:
- **Pequeño / dev**: `RUN_TARGET_BYTES=33554432`, `RUN_RETENTION=4`, `COMPACTION_TRIGGER_TOMBSTONE_RATIO=0.15`, `COMPACTION_MAX_BYTES_PER_PASS=67108864`.
- **Mediano (single node 10M vectores)**: `RUN_TARGET_BYTES=134217728`, `RUN_RETENTION=8`, `COMPACTION_TRIGGER_TOMBSTONE_RATIO=0.2`, `COMPACTION_MAX_BYTES_PER_PASS=268435456`.
- **Grande (50M+)**: `RUN_TARGET_BYTES=536870912`, `RUN_RETENTION=12`, `COMPACTION_TRIGGER_TOMBSTONE_RATIO=0.25`, `COMPACTION_MAX_BYTES_PER_PASS=1073741824`.

### IVF / Q8 (tuning fino)

| Variable                    | Descripción                                                                 | Valor por defecto |
| --------------------------- | --------------------------------------------------------------------------- | ----------------- |
| `IVF_CLUSTERS`              | Cantidad de centroides (`centroid_count`). Define el tamaño de cada cluster. | `4096`            |
| `IVF_NPROBE`                | Número de centroides a inspeccionar por búsqueda (`nprobe`).                | `16`              |
| `IVF_TRAINING_SAMPLE`       | Máximo de vectores muestreados para kmeans++ offline.                       | `200000`          |
| `IVF_MIN_TRAIN_VECTORS`     | Mínimo de vectores vivos antes de entrenar por primera vez.                 | `1024`            |
| `IVF_RETRAIN_MIN_DELTAS`    | Delta de upserts antes de reentrenar los centroides.                        | `50000`           |
| `Q8_REFINE_TOPK`            | Cantidad de candidatos Q8 que se re-scorean en `f32` para el top-K final.   | `512`             |

Valores sugeridos:
- **Pequeño / dev (≤1M vectores)**: `IVF_CLUSTERS=256`, `IVF_NPROBE=8`, `IVF_MIN_TRAIN_VECTORS=2048`, `Q8_REFINE_TOPK=256`.
- **Mediano (≈10M vectores)**: `IVF_CLUSTERS=1024`, `IVF_NPROBE=16`, `IVF_MIN_TRAIN_VECTORS=8192`, `IVF_RETRAIN_MIN_DELTAS=100000`, `Q8_REFINE_TOPK=512`.
- **Grande (50M+)**: `IVF_CLUSTERS=4096-8192`, `IVF_NPROBE=24-32`, `IVF_MIN_TRAIN_VECTORS=50000`, `IVF_RETRAIN_MIN_DELTAS=250000`, `Q8_REFINE_TOPK=1024`.

### DiskANN (experimental)

| Variable                   | Descripción                                                         | Valor por defecto |
| -------------------------- | ------------------------------------------------------------------- | ----------------- |
| `DISKANN_MAX_DEGREE`       | Grado máximo del grafo on-disk (vecinos por vector).                | `48`              |
| `DISKANN_BUILD_THREADS`    | Hilos usados al construir el grafo.                                 | `# CPUs`          |
| `DISKANN_SEARCH_LIST_SIZE` | Nodos que el buscador explora antes de devolver resultados.         | `64`              |

> Cada upsert/delete invalida el índice en disco; vuelve a ejecutar `vector_build_disk_index` para refrescarlo antes de servir con `INDEX_KIND=DISKANN`.

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

### Control de DiskANN

- `POST /v1/vector/:collection/diskann/build`: reconstruye el grafo en disco. Cuerpo opcional (`max_degree`, `build_threads`, `search_list_size`).
- `POST /v1/vector/:collection/diskann/tune`: persiste nuevos `DiskAnnBuildParams` sin regenerar inmediatamente.
- `GET /v1/vector/:collection/diskann/status`: reporta `available`, `last_built_ms`, archivos y knobs vigentes.

CLI rápido:

```
rust-kiss-vdb diskann build  --collection docs [--max-degree 96 --build-threads 8 --search-list 256]
rust-kiss-vdb diskann tune   --collection docs [--search-list 192]
rust-kiss-vdb diskann status --collection docs
```

Los par metros se guardan en `manifest.json.disk_index.build_params`, as¡ sobrevive cualquier reinicio y se usa como default para la pr¢xima corrida.

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

## Benchmarks IVF vs f32

El binario src/bin/bench.rs genera colecciones sintéticas y compara tres modos:

- aseline_f32: índice HNSW + vectores 32
- ivf_flat_q8: IVF + cuantización Q8 + refinamiento 32
- diskann: grafo on-disk + cache paginada (requiere DATA_DIR)

Ejemplo rápido:

`ash
cargo run --release --bin bench -- --rows 200000 --search-queries 5000 --dims 768,1024,4096 --modes baseline_f32,ivf_flat_q8 --ivf-clusters 2048 --ivf-nprobe 24
`

La salida reporta insert/search (p50/p95/p99, throughput), 
ecall@k relativo al baseline, RAM usada y tamaño en disco por modo. Los flags --ivf-clusters, --ivf-nprobe, --q8-refine-topk, --ivf-min-train-vectors y --ivf-retrain-min-deltas permiten probar distintos tunings sin reiniciar el servicio.

Para capturar una comparativa DISKANN vs IVF vs baseline (sin castigar el SSD) usa el pipeline prepare → build-index → run-queries ya optimizado:

`
cargo run --release --bin bench --   --rows 20000 --search-queries 1000 --dims 4096   --modes baseline_f32,ivf_flat_q8,diskann   --ivf-clusters 2048 --ivf-nprobe 16 --q8-refine-topk 512   --diskann-max-degree 64 --diskann-search-list-size 256   --reuse-data --reuse-index --keep-data
`

- --reuse-data genera el dataset binario solo una vez por dim/seed.
- --reuse-index evita rebuilds si el fingerprint coincide.
- --keep-data conserva 	arget/bench/<hash> hasta que lo borres.

Smoke test (<100 vectores) que no toca disco:

`
cargo run --release --bin bench -- --rows 100 --search-queries 20 --dims 768 --modes diskann --diskann-max-degree 32 --diskann-search-list-size 64 --in-mem
`

## Interfaces de índice (preparación DiskANN/Vamana)

En `src/vector/index.rs` se definen dos traits:

- `VectorIndex`: superficie mínima que cualquier índice en memoria debe exponer (`create_collection`, `upsert`, `search`, `compact`, `retrain_ivf`, etc.).
- `DiskVectorIndex`: extensión para índices on-disk; incluye `warm_collection` y `sync_collection` para preparar caches y fsync/manifest antes del salto a DiskANN.

`VectorStore` implementa ambos traits, de modo que el motor puede intercambiar implementaciones (IVF, HNSW, futuros índices híbridos o DiskANN) sin reescribir el Engine/API.
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





