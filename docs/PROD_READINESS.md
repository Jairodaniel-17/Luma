# Prod Readiness (KISS)

## Requisitos mínimos

- **Durabilidad**: define `DATA_DIR` en producción (habilita WAL segmentado + snapshots).
- **Bind seguro**: sin flags el binario sólo escucha en `127.0.0.1`. Usa `--bind 0.0.0.0` o `--unsafe-bind` sólo si lo pones detrás de un proxy.
- **Auth**: exporta `RUSTKISS_API_KEY`/`API_KEY` para exigir `Authorization: Bearer …`. Si no lo haces, las rutas quedan abiertas (útil para laboratorio, no prod).

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

- En disco se mantiene `vectors/<collection>/{manifest.json,vectors.bin}`. Cada mutaci¢n es append-only; borra = tombstone. Usa `rust-kiss-vdb vacuum --collection <name>` para compactar sin reiniciar.
- Rebuild al arranque = leer `vectors.bin` + recrear segmentos/HNSW. Costo observado: ~120 ms por cada 10k vectores (dim 384) en laptop m3.
- Límites recomendados (v1): `dim <= 1536`, `k <= 200`, `<= 1e6` vectores por colección (más allá considera sharding o instancias extra).
- SSE vectorial expone `collection` en `data` y respeta `?collection=foo` en `/v1/stream`. También se envía `event: vector_*` para auditar ingestas.
- Los filtros por metadata usan un índice exact-match; dimensiona la RAM según tu cardinalidad.

## DocStore / SQL

- DocStore vive sobre KV (`doc:{collection}:{id}` + `docidx:*`). Ideal para dashboards/configuraciones ligeras.
- SQLite embebido (`SQLITE_ENABLED=1`) comparte proceso pero NO WAL; respáldalo como parte del backup del `DATA_DIR`.
- Ambos módulos reutilizan el middleware de auth/API key; si no los necesitas mantenlos desactivados.
