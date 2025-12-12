# Configuración

Variables de entorno:

- `PORT` (default `8080`)
- `API_KEY` (default `dev`)
- `DATA_DIR` (opcional; si existe habilita WAL + snapshots)
- `SNAPSHOT_INTERVAL_SECS` (default `30`)
- `EVENT_BUFFER_SIZE` (default `10000`)
- `LIVE_BROADCAST_CAPACITY` (default `4096`)
- `WAL_SEGMENT_MAX_BYTES` (default `67108864`)
- `WAL_RETENTION_SEGMENTS` (default `8`)
- `REQUEST_TIMEOUT_SECS` (default `30`)
- `MAX_BODY_BYTES` (default `1048576`)
- `MAX_JSON_BYTES` (default `65536`)
- `MAX_KEY_LEN` (default `512`)
- `MAX_COLLECTION_LEN` (default `64`)
- `MAX_ID_LEN` (default `128`)
- `MAX_VECTOR_DIM` (default `4096`)
- `MAX_K` (default `256`)
- `CORS_ALLOWED_ORIGINS` (opcional; lista separada por comas)

Ejemplo:
```bash
set PORT=8080
set API_KEY=dev
set DATA_DIR=.\data
set SNAPSHOT_INTERVAL_SECS=10
set WAL_SEGMENT_MAX_BYTES=16777216
set WAL_RETENTION_SEGMENTS=16
```
