# Configuración

> **Empieza por [`MANUAL_USUARIO.md`](../../MANUAL_USUARIO.md)** si lo que buscas
> es *cómo configurar Luma correctamente*. Esta página es el listado de claves;
> el manual es el que te dice cuáles importan y por qué.

Precedencia, de mayor a menor:

```
argumentos de CLI   >   variables de entorno   >   luma.toml   >   defaults
```

`luma.toml` se genera con los defaults si no existe.

## Obligatorias en producción

Luma **se niega a arrancar** si faltan, salvo que pongas `LUMA_ALLOW_INSECURE=1`
(solo desarrollo).

| Variable | Regla |
|---|---|
| `LUMA_API_KEY` / `API_KEY` | Rechazada si es `dev`, si está vacía o si tiene menos de 16 caracteres |
| `LUMA_MASTER_KEY` | Rechazada si no está puesta. Cifrado en reposo con ChaCha20-Poly1305 |

## Servidor

| Variable | Default | Nota |
|---|---|---|
| `PORT_LUMA_VDB` | `1234` | `--port` tiene prioridad |
| `BIND_ADDR` | `127.0.0.1` | Para exponer, `--bind <ip>` o `--unsafe-bind` |
| `TLS_CERT_PATH` / `TLS_KEY_PATH` | — | |
| `CORS_ALLOWED_ORIGINS` | — | Lista separada por comas |
| `RATE_LIMIT_RPS` | `100` | `0` desactiva el límite |
| `RATE_LIMIT_BURST` | `0` | `0` = 10× `rps` |
| `REQUEST_TIMEOUT_SECS` | `30` | `--request-timeout-secs` |
| `MAX_BODY_MB` | `100` | En MB. `--max-body-mb` |
| `MAX_JSON_MB` | `100` | En MB. `--max-json-mb` |

## Almacenamiento y durabilidad

| Variable | Default | Nota |
|---|---|---|
| `DATA_DIR` | `data` | `--data` / `--data-dir`. **Ponlo en SSD**: en disco mecánico la escritura cae 7,3× |
| `WAL_SYNC_MODE` | `per_write` | `group` ya no afecta al camino KV/state — ver el manual |
| `WAL_SEGMENT_MAX_BYTES` | `67108864` | 64 MiB |
| `WAL_RETENTION_SEGMENTS` | `8` | `--wal-retention` |
| `WAL_FLUSH_INTERVAL_MS` | `10` | Solo modo `group` |
| `WAL_BATCH_SIZE` | `64` | Solo modo `group` |
| `SNAPSHOT_INTERVAL_SECS` | `30` | |
| `EVENT_BUFFER_SIZE` | `10000` | |
| `LIVE_BROADCAST_CAPACITY` | `4096` | |
| `SQLITE_ENABLED` | `true` | `0`/`false`/`off`/`no` lo apagan |
| `SQLITE_DB_PATH` | `DATA_DIR/sqlite/rustkiss.db` | |

## Límites

| Variable | Default |
|---|---|
| `MAX_K` | `256` |
| `MAX_VECTOR_DIM` | `4096` |
| `MAX_KEY_LEN` | `512` |
| `MAX_COLLECTION_LEN` | `64` |
| `MAX_ID_LEN` | `128` |
| `MAX_STATE_BATCH` | `256` |
| `MAX_VECTOR_BATCH` | `256` |
| `MAX_DOC_FIND` | `100` |
| `MAX_COLLECTION_VECTORS` | `0` (sin límite) |

## Vectorial

| Variable | Default |
|---|---|
| `INDEX_KIND` | `IVF_FLAT_Q8` (`HNSW` \| `IVF_FLAT_Q8` \| `DiskANN`) |
| `HNSW_M` | `16` |
| `HNSW_EF_CONSTRUCTION` | `200` |
| `HNSW_SEARCH_EF` | `128` |
| `IVF_CLUSTERS` | `4096` |
| `IVF_NPROBE` | `16` |
| `IVF_MIN_TRAIN_VECTORS` | `1024` |
| `IVF_TRAINING_SAMPLE` | `200000` |
| `IVF_RETRAIN_MIN_DELTAS` | `50000` |
| `Q8_REFINE_TOPK` | `512` |
| `DISKANN_MAX_DEGREE` | `48` |
| `DISKANN_SEARCH_LIST_SIZE` | `64` |
| `DISKANN_AUTO_BUILD_MIN_VECTORS` | `10000` |
| `DISKANN_REBUILD_MIN_DELTAS` | `100000` |
| `PRE_FILTER_THRESHOLD` | `10000` |
| `SEARCH_THREADS` | `0` (automático) |
| `SIMD_ENABLED` | `true` |

## Embeddings

| Variable | Default |
|---|---|
| `EMBEDDING_PROVIDER` | `none` (`ollama` \| `openai` \| `azure` \| `cohere` \| `huggingface` \| `mock`) |
| `EMBEDDING_MODEL` / `EMBEDDING_URL` / `EMBEDDING_API_KEY` | — |
| `EMBEDDING_DIM` | `384` |
| `EMBEDDING_CACHE_SIZE` | `10000` |
| `EMBEDDING_MAX_INFLIGHT_REQUESTS` | `16` |
| `EMBEDDING_RETRY_ATTEMPTS` | `3` |
| `EMBEDDING_RETRY_INITIAL_MS` | `200` |
| `EMBEDDING_AZURE_API_BASE` / `_DEPLOYMENT` | — |
| `EMBEDDING_AZURE_API_VERSION` | `2024-02-01` |
| `EMBEDDING_COHERE_INPUT_TYPE` | `search_document` |

## RESP (compatibilidad Redis)

| Variable | Default |
|---|---|
| `RESP_PORT` | `0` (apagado) |
| `RESP_MAX_CLIENTS` | `10000` |
| `RESP_IDLE_TIMEOUT_SECS` | `300` |
| `RESP_MAX_BUFFER_BYTES` | `67108864` (64 MiB) |
| `RESP_PUBSUB_INBOX` | `1024` |
| `RESP_ALLOW_FLUSH` | `false` |
| `RESP_TLS_ENABLED` | `false` |
| `RESP_TLS_CERT_PATH` / `RESP_TLS_KEY_PATH` | — |

## S3

| Variable | Default |
|---|---|
| `S3_PORT` | `0` (apagado) |

## Memoria de agentes (NS-Mem)

| Variable | Default |
|---|---|
| `MEMORY_WORKING_TTL_SECS` | `3600` |
| `MEMORY_DEFAULT_LIMIT` | `10` |
| `MEMORY_MAX_EVIDENCE` | `10` |
| `MEMORY_PROCEDURAL_MAX_NODES` | `128` |
| `MEMORY_FACT_PROMOTION_THRESHOLD` | `0.85` |
| `MEMORY_WALK_MAX_HOPS` | `2` |
| `MEMORY_WALK_MIN_SIMILARITY` | `0.65` |
| `MEMORY_WALK_MAX_NODES` | `40` |
| `MEMORY_CENTRALITY_UPDATE_INTERVAL_SECS` | `300` |
| `MEMORY_DECAY_HALF_LIFE_DAYS` | `30.0` |
| `MEMORY_DECAY_ARCHIVE_THRESHOLD` | `0.1` |
| `MEMORY_DECAY_INTERVAL_SECS` | `3600` |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_URL` / `LLM_API_KEY` | `none` |

## Respaldos y réplica

| Variable | Default |
|---|---|
| `BACKUP_DIR` | `backups` |
| `BACKUP_INTERVAL_SECS` / `BACKUP_RETENTION` | — |
| `BACKUP_REMOTE_URL` / `_ENDPOINT` / `_REGION` | — |
| `BACKUP_REMOTE_ALLOW_HTTP` | `false` |
| `ROLE` | primario |
| `REPLICA_POLL_INTERVAL_SECS` | `10` |
| `WAL_SHIP_INTERVAL_SECS` | `0` (apagado) |

## Observabilidad

| Variable | Default |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — |
| `SLOW_QUERY_THRESHOLD_MS` | `0` |

## Flags de arranque

- `--port <n>`, `--bind <ip>` (o `--host`), `--unsafe-bind`
- `--data <ruta>` / `--data-dir <ruta>`
- `--sqlite` / `--sqlite-enabled`, `--no-sqlite`
- `--max-body-mb`, `--max-json-mb`, `--max-vector-dim`, `--max-k`,
  `--max-key-len`, `--max-collection-len`, `--wal-retention`,
  `--request-timeout-secs`
- De subcomandos: `--verify` (backup), `--collection` (vacuum), `--once` y
  `--no-backfill` (connect), `--max-degree` / `--search-list` /
  `--build-threads` (diskann)

**No hay flag de nivel de log.** El nivel sale de `RUST_LOG` (default `info`):
`RUST_LOG=warn`, `RUST_LOG=luma=debug`, etc. Una versión anterior de esta página
documentaba un `--logs` que el binario nunca aceptó.

## Ejemplo

```bash
export LUMA_API_KEY="$(openssl rand -hex 24)"
export LUMA_MASTER_KEY="$(openssl rand -hex 32)"
export DATA_DIR=/var/lib/luma          # en SSD
export PORT_LUMA_VDB=1234
export RESP_PORT=6379                  # compatibilidad Redis
export S3_PORT=9000                    # API S3
RUST_LOG=warn luma serve --bind 0.0.0.0
```
