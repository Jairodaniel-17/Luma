# RustKissVDB

Base de datos en Rust con enfoque KISS que combina **State Store + Event Store + SSE + Vector Store** en un solo binario listo para usar.

> Filosofia: menos capas, mas visibilidad en vivo y persistencia opcional via WAL+snapshots.

## Que incluye

- **HTTP API (axum + tokio)**: endpoints `state`, `vector`, `stream`, `health`, `metrics`.
- **SSE Event Stream**: `GET /v1/stream` con replay por offset, filtros por tipo/prefijo/coleccion y manejo de gaps.
- **State Engine**: KV in-memory con TTL, revision incremental y soporte CAS (`if_revision`).
- **Event Store**: offset global u64, WAL segmentado y snapshot opcional cuando apuntas `DATA_DIR`.
- **Vector Store (HNSW)**: colecciones `{dim, metric}`, operaciones completas `add/upsert/update/delete/get/search` y filtros exactos sobre `meta`.

## Arquitectura en 30s

1. **Control plane HTTP** recibe comandos REST.
2. **Engine State+Events** valida, versiona y publica sobre un `EventBus`.
3. **SSE Layer** expone el log vivo y replay desde WAL/buffer.
4. **Vector Store** reconstruye el indice desde `vectors/<collection>/{manifest.json,vectors.bin}` en cada arranque.
5. **Persistencia** se habilita cuando defines `DATA_DIR`, guardando WAL segmentado y snapshots periodicos (`SNAPSHOT_INTERVAL_SECS`).

Detalles completos en `docs/ARCHITECTURE.md`.

## Requisitos previos

- Rust estable (cargo 1.77+).
- (Opcional) `DATA_DIR` apuntando a un directorio escribible para habilitar durabilidad.

## Quickstart

```powershell
set DATA_DIR=.\data            # omite si solo quieres in-memory
cargo run --bin rust-kiss-vdb -- --logs info
```

1. **SSE**  
   `curl -N "http://localhost:9917/v1/stream?since=0"`
2. **Docs vivas**  
   - Swagger UI: <http://localhost:9917/docs>  
   - OpenAPI: <http://localhost:9917/openapi.yaml>
3. **Demo end-to-end**  
   `scripts\demo.ps1` (crea coleccion vectorial, publica estado y muestra eventos).

> Tip: usa `--logs warning` para reducir ruido o cambia en caliente via flag sin tocar `RUST_LOG`.

## Config rapida

Variables de entorno criticas:

| Variable | Default | Nota |
| --- | --- | --- |
| `PORT_RUST_KISS_VDB` | `9917` | Puerto HTTP unico (CLI `--port` manda si se define). |
| `DATA_DIR` | vacio | Activa WAL segmentado + snapshot + storage vectorial en disco. |
| `SNAPSHOT_INTERVAL_SECS` | `30` | Bloquea WAL breve y rota segmentos. |
| `EVENT_BUFFER_SIZE` | `10000` | Buffer de replay in-memory. |
| `LIVE_BROADCAST_CAPACITY` | `4096` | Fijar mas alto si hay bursts SSE. |
| `MAX_*` | ver `docs/CONFIG.md` | Limites anti-DoS (body/json/key/id/dim/k). |

Flags CLI: `--logs info|warning|error|critical`, `--port <u16>` (prioridad: CLI `--port` -> `PORT_RUST_KISS_VDB` -> `9917`).

Ejemplos avanzados en `docs/CONFIG.md`.

## API Surface

- **State**: `PUT/GET/DELETE /v1/state/{key}`, `GET /v1/state?prefix=&limit=` (ver ejemplos en `docs/API.md`).
- **Vector**: `POST /v1/vector/{collection}` + `add/upsert/update/delete/get/search`.
- **Events**: `GET /v1/stream?since=<offset>&types=...&key_prefix=...&collection=...`.
- **Ops**: `GET /v1/health`, `GET /v1/metrics`.

Sin autenticacion en v1; protege detras de tu API Gateway hasta que se reintroduzca `API_KEY`. Consulta `docs/API.md` para `curl` listos.

## Docs de apoyo

| Archivo | Contenido |
| --- | --- |
| `docs/ARCHITECTURE.md` | Decisiones de diseno de engine, SSE, WAL y vector store. |
| `docs/API.md` | Cheatsheet de endpoints y ejemplos `curl`. |
| `docs/CONFIG.md` | Todas las variables/limites y ejemplo de arranque. |
| `docs/DEMO.md` | Pasos del demo para ver SSE + vector en vivo. |
| `docs/BENCH.md` | Como correr el micro bench (`cargo run --release --bin bench`). |
| `docs/PROD_READINESS.md` | Checklist para llevarlo a produccion (durability, CORS, limites, tuning SSE). |

## Demo y Bench

- **Demo**: sigue `docs/DEMO.md` para levantar SSE, correr `scripts/demo.ps1` y ver eventos `state_*` y `vector_*`.
- **Bench**: `cargo run --release --bin bench` imprime p50/p95 microseg para `state.put/get` y `vector.search`.

## Prod readiness express

- Define `DATA_DIR` siempre en ambiente real para no perder eventos.
- Ajusta `LIVE_BROADCAST_CAPACITY` si esperas picos SSE; clientes lentos veran `event: gap` pero pueden reconectar con `since`.
- Configura `MAX_BODY_BYTES`, `MAX_VECTOR_DIM`, `MAX_K`, etc. para tu carga.
- CORS abierto en dev; en prod define `CORS_ALLOWED_ORIGINS`.
- Logs: `--logs warning` recomendado y exporta stdout/stderr a tu stack.

Checklist completo en `docs/PROD_READINESS.md`.

## Estado del proyecto

- v1 listo para un solo nodo, sin autenticacion incorporada.
- Persistencia, vector store y SSE probados via tests + demo.
- Roadmap inmediato: compaction de vectores y controles de acceso (ver `codex/AGENTS.md` para contexto del agente).

Con esto deberias poder levantar RustKissVDB, explorar la API y operar el stack con el material dentro de `docs/`.
