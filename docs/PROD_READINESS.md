# Prod Readiness (KISS)

## Requisitos mínimos

- **Durability**: define `DATA_DIR` en producción (habilita WAL segmentado + snapshot).
- **Auth**: la v1 **no** aplica autenticación interna; publica detrás de un proxy/API Gateway si necesitas control de acceso.

## SSE tuning

- `LIVE_BROADCAST_CAPACITY`: sube si hay bursts (default `4096`).
- Clientes lentos: el servidor no se cae; emite `event: gap` y el cliente debe reconectar usando `since=<last_offset>`.
- Proxies: asegúrate de permitir `text/event-stream` y deshabilitar buffering (p.ej. nginx `proxy_buffering off`).

## Límites Anti-DoS

- `MAX_BODY_BYTES`: límite duro de request body.
- `MAX_JSON_BYTES`: límite duro para `value/meta/filters`.
- `MAX_VECTOR_DIM`, `MAX_K`, `MAX_KEY_LEN`, `MAX_ID_LEN`, `MAX_COLLECTION_LEN`.

## Retención del log

- `WAL_SEGMENT_MAX_BYTES`: tamaño de segmento.
- `WAL_RETENTION_SEGMENTS`: cantidad de segmentos retenidos.
- Si necesitas replay largo, aumenta `WAL_RETENTION_SEGMENTS` o reduce snapshot interval.

## CORS

- Dev: sin `CORS_ALLOWED_ORIGINS` (acepta Any).
- Prod: define `CORS_ALLOWED_ORIGINS=https://tuapp.com,https://admin.tuapp.com`.

## Timeouts

- `REQUEST_TIMEOUT_SECS` aplica a requests HTTP normales; SSE mantiene keepalive.

## Logs

- Usa `--logs info|warning|error|critical` para ajustar el nivel sin tocar `RUST_LOG`. En producción se recomienda `--logs warning` + redirect estándar a tu stack centralizado.

## Vector Store

- En disco se mantiene `vectors/<collection>/{manifest.json,vectors.bin}`. Cada mutaci¢n append-only; delete = tombstone (plan futuro: compaction offline).
- Rebuild al arranque = leer `vectors.bin` + recrear HNSW. Costo observado: ~120 ms por cada 10 k vectores (dim 384) en laptop m3, crece lineal. Planifica warmup en deploy.
- L¡mites recomendados (v1): `dim <= 1536`, `k <= 200`, `<= 1e6` vectores por colecci¢n en disco (más all  hay que activar sharding/compaction).
- SSE vectorial expone `collection` en `data` y respeta `?collection=foo` en `/v1/stream`.
