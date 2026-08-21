# Prod Readiness (KISS)

## Requisitos mínimos

- **Durabilidad**: define `DATA_DIR` en producción (habilita WAL segmentado + snapshots).
- **Integridad WAL**: los segmentos nuevos incluyen checksum por registro. Si el proceso cae a mitad de append, Luma recupera el prefijo válido y contabiliza corrupción detectada en métricas.
- **Bind seguro**: sin flags el binario sólo escucha en `127.0.0.1`. Usa `--bind 0.0.0.0` o `--unsafe-bind` sólo si lo pones detrás de un proxy.
- **Auth**: exporta `RUSTKISS_API_KEY`/`API_KEY` para exigir `Authorization: Bearer …`. Si no lo haces, las rutas quedan abiertas (útil para laboratorio, no prod).

## Durabilidad por plataforma (W1.2, parcial)

Tabla en construcción — la auditoría completa de fsync por primitiva es el ítem
W1.2 del [plan maestro](PLAN-MAESTRO.md). Lo verificado hasta ahora:

| Operación | Linux / macOS | Windows |
|---|---|---|
| Escritura del manifest de una colección vectorial (`write_manifest`) | `write` → `sync_data` del fichero temporal → `rename` → **fsync del directorio** | igual, pero **sin fsync del directorio**: NTFS rechaza `FlushFileBuffers` sobre un handle de directorio, así que la durabilidad de la entrada del directorio depende del journaling de metadatos de NTFS |

Cómo se comprobó en Windows: `File::open()` sobre un directorio devuelve
`PermissionDenied`; con `FILE_FLAG_BACKUP_SEMANTICS` el handle sí abre, pero
`sync_all()` sobre él vuelve a devolver `PermissionDenied`. No es un fallo de
E/S: es que la operación no existe en esa plataforma, y por eso no puede hacer
fallar la escritura que la solicitó.

**Recomendación operativa:** el objetivo de despliegue es Linux (imagen musl).
Windows es plataforma de desarrollo soportada, no de producción con garantías
de durabilidad equivalentes.

## SSE tuning

- `LIVE_BROADCAST_CAPACITY`: sube si hay bursts (default `4096`).
- Clientes lentos: el servidor no se cae; emite `event: gap` y el cliente debe reconectar usando `since=<last_offset>`.
- Si el `since` pedido ya no está retenido por buffer o WAL, Luma emite `event: gap` con `from_offset`/`to_offset` antes del primer evento disponible.
- Proxies: asegúrate de permitir `text/event-stream` y deshabilitar buffering (p.ej. nginx `proxy_buffering off`).

## Límites Anti-DoS

- `MAX_BODY_BYTES`: límite duro de request body.
- `MAX_JSON_BYTES`: límite duro para `value/meta/filters`.
- `MAX_VECTOR_DIM`, `MAX_K`, `MAX_KEY_LEN`, `MAX_ID_LEN`, `MAX_COLLECTION_LEN`.

## Retención del log

- `WAL_SEGMENT_MAX_BYTES`: tamaño de segmento.
- `WAL_RETENTION_SEGMENTS`: cantidad de segmentos retenidos.
- Si necesitas replay largo, aumenta `WAL_RETENTION_SEGMENTS` o reduce snapshot interval.
- Observa `wal_replay_corrupt_total`, `wal_gap_total` y `wal_rotation_total` en `/v1/metrics`.

## CORS

- Dev: sin `CORS_ALLOWED_ORIGINS` (acepta Any).
- Prod: define `CORS_ALLOWED_ORIGINS=https://tuapp.com,https://admin.tuapp.com`.

## Timeouts

- `REQUEST_TIMEOUT_SECS` aplica a requests HTTP normales; SSE mantiene keepalive.

## Logs

- Usa `--logs info|warning|error|critical` para ajustar el nivel sin tocar `RUST_LOG`. En producción se recomienda `--logs warning` + redirect estándar a tu stack centralizado.

## Vector Store

- En disco se mantiene `vectors/<collection>/{manifest.json,vectors.bin}`. Cada mutaci¢n es append-only; borra = tombstone. Usa `rust-kiss-vdb vacuum --collection <name>` para compactar sin reiniciar.
- Rebuild al arranque = leer `vectors.bin` + recrear segmentos/HNSW. Costo observado: ~120 ms por cada 10k vectores (dim 384) en laptop m3.
- Límites recomendados (v1): `dim <= 1536`, `k <= 200`, `<= 1e6` vectores por colección (más allá considera sharding o instancias extra).
- SSE vectorial expone `collection` en `data` y respeta `?collection=foo` en `/v1/stream`. También se envía `event: vector_*` para auditar ingestas.
- Los filtros por metadata usan un índice exact-match; dimensiona la RAM según tu cardinalidad.
- La compactación HNSW en memoria es opt-in (`HNSW_SEGMENT_COMPACTION_ENABLED=1`) y ahora rebuilda fuera del lock de escritura, intercambiando segmentos solo si el `applied_offset` no cambió durante el rebuild.

## Estado / KV

- `/v1/state` soporta `?prefix=` para scans por prefijo y `?start=&end=` para scans lexicográficos end-exclusive.
- CAS (`if_revision`) y replay aplican revisiones de forma idempotente; una revisión más antigua no pisa una más nueva.

## DocStore / SQL

- DocStore vive sobre KV (`doc:{collection}:{id}` + `docidx:*`). Ideal para dashboards/configuraciones ligeras.
- SQLite embebido (`SQLITE_ENABLED=1`) comparte proceso pero NO WAL; respáldalo como parte del backup del `DATA_DIR`.
- Ambos módulos reutilizan el middleware de auth/API key; si no los necesitas mantenlos desactivados.
- El hub híbrido publica métricas por etapa: `hybrid_sql_prefilter_duration_ms`, `hybrid_vector_duration_ms`, `hybrid_hydration_duration_ms`, `hybrid_chunking_duration_ms`, `hybrid_vector_write_duration_ms`, `hybrid_sql_write_duration_ms`.
- También expone contadores/gauges útiles para explicar consultas: `hybrid_sql_first_total`, `hybrid_vector_first_total`, `hybrid_last_sql_candidates`, `hybrid_last_vector_candidates`, `hybrid_last_doc_candidates`, `hybrid_last_hydrated_docs`.
