# Configuración

Variables de entorno:

- `PORT_RUST_KISS_VDB` (default `9917`; CLI `--port` tiene prioridad)
- `BIND_ADDR` (default `127.0.0.1`; requiere `--bind 0.0.0.0` o `--unsafe-bind` para exponer)
- `RUSTKISS_API_KEY` / `API_KEY` (token Bearer; si no se define no hay auth)
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
- `MAX_STATE_BATCH` (default `256`; límite de operaciones por batch state)
- `MAX_VECTOR_BATCH` (default `256`; límite por batch vector)
- `MAX_DOC_FIND` (default `100`; límite por `doc.find`)
- `CORS_ALLOWED_ORIGINS` (opcional; lista separada por comas)
- `SQLITE_ENABLED` (`1`/`true` activa `/v1/sql/*`)
- `SQLITE_DB_PATH` (ruta custom; default `DATA_DIR/sqlite/rustkiss.db`)

Flags de arranque:

- `--logs info|warning|error|critical` (default `info`; controla el nivel de logging sin tocar `RUST_LOG`)
- `--bind <ip>` (override puntual de `BIND_ADDR`)
- `--unsafe-bind` (alias directo a `0.0.0.0`; imprime warning)

Ejemplo:
```bash
set PORT_RUST_KISS_VDB=12000
set DATA_DIR=.\data
set SNAPSHOT_INTERVAL_SECS=10
set WAL_SEGMENT_MAX_BYTES=16777216
set WAL_RETENTION_SEGMENTS=16
set RUSTKISS_API_KEY=super-secret
cargo run --bin rust-kiss-vdb -- --logs warning
```

Override puntual via CLI:
```bash
cargo run --bin rust-kiss-vdb -- --port 13000 --bind 0.0.0.0
```
