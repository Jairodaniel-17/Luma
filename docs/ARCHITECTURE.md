# Arquitectura (v1)

Objetivo: DB single-node KISS con **State Store + Event Store + SSE + Vector Store**.

## Componentes

### API HTTP (Control Plane)
- `axum` + `tokio`.
- Auth v1: `Authorization: Bearer <API_KEY>`.
- Endpoints: `state`, `vector`, `events (SSE)`, `health`, `metrics`.

### Engine (State + Events)
- State in-memory `key -> {value, revision, expires_at_ms?}`.
- TTL: se purga periódicamente (cada 1s) y en acceso.
- Versionado: `revision` monotónico por key; `if_revision` opcional (CAS simple).

### EventBus (SSE + replay)
- Cada mutación publica un evento con `offset` global u64 incremental.
- `broadcast` para fan-out “tail” a clientes SSE.
- Si `DATA_DIR` está habilitado, el replay es desde WAL segmentado; el buffer in-memory queda como fallback si no hay disco.

### Persistencia (opcional)
Si `DATA_DIR` está definido:
- WAL segmentado: `DATA_DIR/events-000001.log`, `events-000002.log` (JSON lines, append-only).
- Snapshot: `DATA_DIR/snapshot.json` (estado + vectores + `last_offset`).
- Snapshot periódico (`SNAPSHOT_INTERVAL_SECS`) bloquea momentáneamente el WAL, escribe snapshot y rota truncando el WAL.

Invariante: el evento se emite “en vivo” **después** de persistirse en WAL (cuando `DATA_DIR` está habilitado).

### Vector Store (v1)
- Colecciones: `{dim, metric}` con `hnsw_rs` por colecci¢n.
- Layout en disco (por colecci¢n, cuando `DATA_DIR` est  definido):
  - `vectors/<collection>/manifest.json`: `{dim, metric, applied_offset, live_count, total_records, upsert_count, file_len}`.
  - `vectors/<collection>/vectors.bin`: stream binario `[u32 len][bincode<Record>]` (Upsert/Delete). No se usa `mmap` (s¢lo read/append).
- Operaciones: create/add/upsert/update/delete/get/search.
- Search: s¢lo HNSW (`cosine` o `dot`), `k` limitado por config; filtros por igualdad exacta sobre `meta` JSON.
- Deletes = tombstone en `vectors.bin` (queda deuda de compactaci¢n).
- Arranque:
  1. Leer `manifest`.
  2. Reproducir `vectors.bin` en orden de append (normalizando vectores `dot`).
  3. Reconstruir HNSW usando `upsert_count` como baseline de capacidad.

## SSE
- Endpoint: `GET /v1/stream?since=...&types=...&key_prefix=...&collection=...`
- Reconexión: soporta `Last-Event-ID` o `since` (u64).
- `event:` = tipo, `id:` = id incremental, `data:` = JSON del evento.
- Backpressure: ante `Lagged`, emite `event: gap` con rango de offsets perdidos.
