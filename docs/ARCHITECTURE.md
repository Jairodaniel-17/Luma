# Arquitectura (v1)

Objetivo: DB single-node KISS con **State Store + Event Store + SSE + Vector Store**.

## Componentes

### API HTTP (Control Plane)
- `axum` + `tokio`.
- Endpoints: `state`, `vector`, `doc`, `sql`, `events (SSE)`, `health`, `metrics`.

### Engine (State + Events)
- State in-memory `key -> {value, revision, expires_at_ms?}`.
- TTL: se purga periódicamente (cada 1s) y en acceso.
- Versionado: `revision` monotónico por key; `if_revision` opcional (CAS simple).

### EventBus (SSE + replay)
- Cada mutación publica un evento con `offset` global u64 incremental.
- `broadcast` para fan-out “tail” a clientes SSE.
- Si `DATA_DIR` está habilitado, el replay es desde WAL segmentado; el buffer in-memory queda como fallback si no hay disco.
- El replay in-memory ahora detecta offsets ya no retenidos y emite `event: gap` antes de reanudar desde el primer offset disponible.

### Persistencia (opcional)
Si `DATA_DIR` está definido:
- WAL segmentado: `DATA_DIR/events-000001.log`, `events-000002.log`.
- Cada línea WAL nueva es un envelope versionado con `crc32` del `EventRecord`; el lector sigue aceptando líneas legacy para no romper upgrades.
- Al detectar corrupción parcial o tail truncado, el replay se detiene en el primer registro inválido y conserva el prefijo válido. No continúa sobre bytes dudosos.
- Snapshot: `DATA_DIR/snapshot.json` (checkpoint de `last_offset` para recuperación idempotente).
- Snapshot periódico (`SNAPSHOT_INTERVAL_SECS`) fuerza flush del buffer WAL, escribe snapshot por rename atómico y luego rota segmento.

Invariante: el evento se emite “en vivo” **después** de persistirse en WAL (cuando `DATA_DIR` está habilitado).
Invariante: `state_db` y vector store tratan eventos con `offset <= applied_offset` como idempotentes durante replay.

### Vector Store (v1.1)
- Colecciones: `{dim, metric}` con `hnsw_rs`, ahora divididas en segmentos (`DEFAULT_SEGMENT_MAX=8192`).
- Layout en disco (por colecci¢n, cuando `DATA_DIR` est  definido):
  - `vectors/<collection>/manifest.json`: `{dim, metric, applied_offset, live_count, total_records, upsert_count, file_len}`.
  - `vectors/<collection>/vectors.bin`: stream binario `[u32 len][bincode<Record>]` (Upsert/Delete). No se usa `mmap`.
- Segmentos:
  - Activo: recibe upserts nuevos.
  - Fr­os: sólo lectura hasta vacuum/compaction.
- Search = merge top-k de cada segmento (HNSW) + filtrado por metadata (índice exact-match in-memory).
- Deletes = tombstone hasta que se ejecute `rust-kiss-vdb vacuum --collection <name>`.
- Arranque:
  1. Leer `manifest`.
  2. Reproducir `vectors.bin` normalizando vectores `dot`.
  3. Reconstruir segmentos + HNSW usando `live_count` como baseline.

### DocStore / SQLite
- DocStore vive sobre el KV (`doc:{collection}:{id}` + `docidx:{collection}:{field}:{value}`), expone `/v1/doc/*`.
- SQLite (opcional) usa un `rusqlite::Connection` en WAL mode y expone `/v1/sql/query|exec`.
- Ambos módulos comparten engine/config pero no el mismo WAL (SQLite mantiene su archivo independiente).

### Hub híbrido (`/v1/db`)
- El planner híbrido decide entre dos caminos reales:
  - `sql_first`: `COUNT(*)` del filtro SQL, fetch de ids, `allowed_ids` en vector search, luego colapso e hydration.
  - `vector_first`: vector search ampliado, colapso a documentos candidatos, post-filter SQL con `id IN (...)`, luego hydration.
- El plan se expone solo con `include_plan=true`.
- Los diagnósticos operativos se exponen con `include_diagnostics=true` e incluyen tiempos por etapa y tamaños de candidate sets.

## SSE
- Endpoint: `GET /v1/stream?since=...&types=...&key_prefix=...&collection=...`
- Reconexión: soporta `Last-Event-ID` o `since` (u64).
- `event:` = tipo, `id:` = id incremental, `data:` = JSON del evento.
- Backpressure: ante `Lagged`, emite `event: gap` con rango de offsets perdidos.
