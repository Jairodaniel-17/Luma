<!-- RustKissVDB: el motor de datos convergente en Rust que impulsa Luma (búsqueda vectorial + KV + SQL + eventos en un solo binario). -->
# Luma: La Plataforma de Datos Convergente

**Luma** (crate Cargo `luma`, versión **4.0.0**, *powered by RustKissVDB*) no es solo una base de datos vectorial. Es un **Motor de Datos Convergente** escrito en **Rust** que unifica, en un **único binario** (`luma`), las primitivas que necesita una aplicación de IA moderna:

- **Búsqueda vectorial** (ANN) con índices conmutables HNSW / IVF-FLAT-Q8 / DiskANN.
- **Estado clave-valor** (KV) con TTL, compare-and-swap e índices.
- **SQL relacional** embebido (SQLite en modo WAL), con opción de backend remoto libSQL/Turso.
- **Document store** JSON y **object storage** binario tipo R2.
- **Colas** durables e **imágenes** transformadas on-the-fly.
- **Bus de eventos** pub/sub con streaming SSE.
- **NS-Mem**: una capa de memoria para agentes autónomos (episódica, semántica, procedural y de trabajo).
- **Capa empresarial**: cuentas/organizaciones, roles, login con Argon2id, auditoría, respaldos, cifrado en reposo y un **panel de administración React incrustado en el propio binario**.

Todo corre en un mismo proceso, eliminando la latencia de red entre subsistemas y simplificando el despliegue a un solo ejecutable.

> Ver el estado real de cada pieza en la sección **Estado actual del proyecto** al final de este documento.

---

## 🚀 ¿Por qué Luma?

La premisa es simple: **la IA necesita más que vectores.** Mientras la arquitectura tradicional fragmenta el stack (PostgreSQL para datos, Redis para caché/colas, un servicio aparte para vectores), Luma converge esas primitivas en un binario Rust, con seguridad de memoria, concurrencia sobre Tokio y latencia interna cero entre motores.

---

## 📊 Rendimiento medido (Luma vs Qdrant vs Milvus)

Benchmark reproducible, misma máquina, mismo dataset y misma métrica para los tres motores. Sin cifras vagas: todas las columnas están medidas.

### Máquina de prueba

| | |
|---|---|
| **CPU** | Intel Core i7-1355U (12 hilos, laptop; escalado de frecuencia activo) |
| **RAM** | 15 GiB |
| **Disco** | NVMe SSD |
| **SO** | Ubuntu 24.04.4 LTS · kernel 6.17 |

> Es una laptop con throttling térmico: los valores absolutos suben en servidor, pero la **comparación relativa entre motores es válida** porque los tres corrieron en el mismo equipo, uno a la vez.

### Qué se midió y cómo

- **Dataset:** 50.000 vectores de 768 dimensiones + 200 consultas, distribución aleatoria uniforme (`base.npy` / `queries.npy`, `float32`).
- **Métrica:** cosine, `k = 10`.
- **Ground-truth:** top-10 exacto por fuerza bruta sobre el mismo dataset → recall@10 real, no estimado.
- **Configuración:** valores **por defecto** de cada motor. Qdrant y Milvus con HNSW `M=16, ef_construct=200, ef=64`.
- **RAM:** memoria anónima real del proceso tras cargar (no caché de página de disco).
- **Protocolo:** ingesta por lotes vía API HTTP → espera de indexación → 200 consultas secuenciales → recall contra ground-truth.

### Resultados

| Motor / modo | Ingesta (vec/s) | Consulta (qps) | Latencia media | Recall@10 | RAM |
|---|---:|---:|---:|---:|---:|
| **Qdrant** (HNSW, ef=64) | **1.859** | 237 | 4.22 ms | 0.416 | 320 MB |
| **Milvus** (HNSW, ef=64) | 1.284 | 476 | 2.09 ms | 0.086 | 846 MB |
| **Luma — DiskANN** (low-RAM) | 807 | **515** | **1.94 ms** | 0.056 | **133 MB** |
| **Luma — HNSW** (equilibrio, ef=128 default) | 301 | 99 | 10.08 ms | 0.853 | 411 MB |
| **Luma — HNSW** (velocidad, ef=32) | 295 | 199 | 5.02 ms | 0.474 | 406 MB |

### Discusión

Con vectores **aleatorios uniformes** (sin estructura de clusters) el ANN es un caso adversarial: incluso Qdrant solo alcanza 0.42 de recall. Por eso la columna de recall refleja sobre todo **el punto de equilibrio velocidad↔precisión de cada motor**, más que la calidad absoluta del índice. En embeddings reales (que sí forman clusters) todos los recalls suben.

- **Luma DiskANN** ocupa el extremo *rápido y ligero*: la **consulta más veloz del grupo (515 qps, 1.94 ms) con la menor RAM (133 MB — 2,4× menos que Qdrant, 6,4× menos que Milvus)**, a costa de recall.
- **Luma HNSW** es el modo *precisión*, ahora **calibrado** (ver abajo): a su punto de velocidad (`ef=32`) iguala la latencia de Qdrant (5.0 vs 4.2 ms) **con mejor recall (0.474 vs 0.416)**; subiendo `ef` escala hasta recall 0.95.
- **Qdrant** es un punto medio sólido de fábrica; **Milvus** iguala en ingesta pero pesa 846 MB y su recall a igual `ef` es el más bajo.

### Dónde gana Luma

- 🏆 **Consumo de RAM** — DiskANN corre 50k en **133 MB**; ningún competidor baja de 320 MB. Es el objetivo de diseño y se cumple medido.
- 🏆 **Latencia y throughput de consulta** — **1.94 ms / 515 qps** en DiskANN, el más rápido del grupo.
- 🏆 **Precisión a igual velocidad** — a latencia equivalente a Qdrant, Luma HNSW da **más recall** (0.474 vs 0.416); y llega hasta 0.95 subiendo `ef`.

### 🎯 Punto de equilibrio (calibración HNSW)

El modo HNSW tenía un problema: el bucle de expansión de candidatos perseguía una estimación de recall inalcanzable en datos difíciles y terminaba escaneando casi todo (recall 0.98 pero **348 ms/consulta**, inservible). Se corrigió con dos cambios (`src/vector/mod.rs`, `src/config.rs`):

1. **`ef` de búsqueda configurable** (`HNSW_SEARCH_EF`, default 128) que acota la expansión a un punto fijo, como el `hnsw_ef` de Qdrant.
2. **Búsqueda única** al techo `ef` en vez de rampar 16→32→…→N. La rampa lanzaba varias búsquedas HNSW desechables por consulta: eliminarla dio **~3× más throughput al mismo recall**.

Curva medida tras la calibración (mismo dataset, `HNSW_SEARCH_EF` variando):

| ef | qps | latencia | recall@10 | RAM |
|---:|---:|---:|---:|---:|
| 32 | 199 | 5.02 ms | 0.474 | 406 MB |
| 64 | 140 | 7.13 ms | 0.675 | 406 MB |
| 96 | 115 | 8.65 ms | 0.792 | 408 MB |
| **128 (default)** | **99** | **10.08 ms** | **0.853** | **411 MB** |
| 192 | 82 | 12.19 ms | 0.921 | 409 MB |
| 256 | 74 | 13.45 ms | 0.947 | 407 MB |

Antes vs después, mismo `ef=192`: **26 qps → 82 qps** (3,15×) con recall idéntico (0.92). El usuario elige el punto: `ef=32` para máxima velocidad, `ef≥192` para máximo recall; el default 128 es el balance.

### Lo que aún va por detrás

- **Ingesta con indexación viva** (295–807 vec/s) sigue por debajo del build multihilo de Qdrant/Milvus (1.284–1.859); el build paralelo ya existe para carga masiva, falta llevarlo al upsert vivo.
- **Throughput de consulta HNSW** a igual recall: `hnsw_rs` es algo más lento que el HNSW propio de Qdrant. La ventaja de Luma sigue siendo **RAM (DiskANN)** y **precisión a igual latencia**.

> Scripts y datos del benchmark: `bench/` + `sweep_ef.sh`. Reejecutable en cualquier equipo.

---

## 🏛️ Arquitectura real por módulos

El servidor (`src/server.rs`) valida la configuración, inicializa los subsistemas y arranca el router HTTP (`src/api/mod.rs`) sobre **axum 0.7 / hyper 1**.

### Core Engine — `src/engine/`
Corazón nativo de alto rendimiento en Rust. Coordina los subsistemas, reproduce el WAL al arrancar, expira TTLs y publica cada mutación como evento con offset monotónico (event sourcing).
- **Estado (KV):** `state.rs` / `state_db.rs`. Store en memoria de valores JSON con TTL por clave y compare-and-swap vía `if_revision`; persistencia opcional respaldada por **redb**.
- **Bus de eventos:** `events.rs`. Pub/Sub sobre `tokio::sync::broadcast`; los clientes SSE reciben el flujo en vivo y una señal de "gap" si quedan por detrás del buffer.
- **Persistencia:** `persist.rs`. WAL segmentado (`events-XXXXXX.log`, JSON lines) con snapshots periódicos (`snapshot.json`); el snapshot dispara rotación y limpieza del WAL.
- **Embeddings:** `embeddings.rs`. Cliente HTTP con proveedores conmutables (`none`/`mock`/`ollama`/`openai`/`azure`/`cohere`/`huggingface`), caché LRU, semáforo de concurrencia y reintentos con backoff exponencial + jitter.
- **Parseo y chunking:** `parser.rs` (PDF/DOCX/imágenes vía `pdf-extract`, `docx-rs`, `quick-xml`, `zip`, `image`) y `chunking.rs` para trocear texto antes de embeber.
- **Hub (`hub.rs`):** el orquestador `LumaDatabase` (ver Nivel 2).

### Motor vectorial — `src/vector/`
CRUD de vectores y búsqueda k-NN con tres estrategias de índice conmutables por config (`index_kind`):
- **HNSW** — ANN aproximado en memoria (`hnsw_rs`).
- **IVF_FLAT_Q8** *(por defecto)* — índice invertido con refinamiento por cuantización de 8 bits (`ivf.rs`, `q8.rs`).
- **DiskANN** — grafo Vamana en disco para colecciones masivas (`diskann/`).

Las colecciones se dividen en segmentos (~8 192 vectores); el segmento activo recibe upserts y los congelados son de solo lectura. Los vectores se persisten como binario (`vectors.bin`) con soporte **mmap zero-copy** (`mmap.rs`) y optimizaciones SIMD (`simd.rs`). El filtrado tipado compuesto vive en `filter.rs`.

### Servicio SQL — `src/sqlite/`
**SQLite** embebido (`rusqlite` *bundled*) en modo **WAL**, accedido mediante un **patrón actor** (canal MPSC de Tokio) para consultas async sin bloquear (`actor.rs`, `pool.rs`). Si se define `LIBSQL_URL`, el SQL se enruta a un backend **libSQL/Turso remoto** por Hrana sobre HTTPS (`hrana.rs`). Es la base del pre-filtro del hub, de NS-Mem, de la autenticación y de la capa empresarial.

### Motor de búsqueda de texto — `src/search/`
`SearchEngine` con almacenamiento propio (`storage.rs`), agrupación (`grouping.rs`) y motor de scoring (`engine.rs`), expuesto en los endpoints `/search` y `/search/ingest`.

### Capa de orquestación HTTP — `src/api/`
Router **axum** con autenticación (Bearer API key, claves estáticas y tokens de sesión), CORS configurable, timeouts, límite de tamaño de body, rate limiting opcional (`tower_governor`) y TLS opcional (`rustls`). Las rutas se dividen por dominio en `routes_*.rs`. Documentación interactiva **Scalar** servida en `/docs` desde `docs/openapi.yaml` incrustado.

### Memoria de agentes (NS-Mem) — `src/memory/`
Capa de memoria para agentes autónomos; ver el **Nivel 3** en la sección de API.

---

## 🏢 Capa Empresarial: Multi-Tenancy, Panel de Administración y Seguridad

Capa "enterprise" **aditiva** montada sobre las primitivas del core. Todo vive en el mismo binario: no necesitas Node, ni el código fuente del panel, ni servicios externos en runtime. El `AccountsService` y las tablas `sys_*` se crean *lazily* la primera vez que se usan, siempre que SQLite esté habilitado.

### Cuentas, sesiones y roles (`src/api/accounts.rs`, `routes_accounts.rs`)
- **Organizaciones y usuarios** en SQLite (`sys_orgs`, `sys_users`), más `sys_sessions` para tokens de sesión y `sys_collections` para propiedad de recursos.
- **Login por email + contraseña**: las contraseñas se hashean con **Argon2id** (`src/crypto.rs`). El login emite un **token de sesión opaco** (`lums_…`) del que solo se guarda su hash SHA-256; TTL de 7 días.
- **Roles**: `owner` > `admin` > `member` > `viewer`, integrados con el RBAC existente (`rbac.rs`, niveles viewer=10, member=20, admin=30, owner=40). Un middleware exige rol mínimo por ruta.
- Endpoints: `POST /v1/auth/register` · `login` · `logout` · `refresh`; gestión admin en `/v1/admin/orgs`, `/v1/admin/users` (alta/baja/roles), `/v1/admin/stats` y `/v1/admin/audit-events`.

### Aislamiento de datos por organización (`tenant_isolation_middleware`)
Cada colección/documento/blob queda asociado a la organización que la creó (*first-touch* en `sys_collections`). Otra organización que intente acceder a ese nombre recibe `404` — la existencia queda oculta entre tenants. El hub (`/v1/db`) y NS-Mem (`/v1/memory`) **comparten namespace a propósito** y ya aíslan internamente por el `tenant_id` del token, por lo que no se les impone propiedad exclusiva.

### Panel de administración (React + Vite + TypeScript)
- El código fuente vive en `admin-ui/` (`App.tsx`, `api.ts`, `main.tsx`); se compila (`npm ci && npm run build`) a `ui/dist/` (bundle real JS + CSS bajo `ui/dist/assets/`) y se **incrusta en el binario** con `rust-embed` (`routes_ui.rs`). Axum lo sirve en `/` con *fallback* SPA para las rutas del cliente.
- Cubre login/registro, dashboard de uso (`/v1/admin/stats`), gestión de usuarios y organizaciones, API keys, registro de auditoría y estado de salud. Usa rutas relativas `/v1/*` (sin hosts hardcodeados).
- React escapa el contenido por defecto y las respuestas de la API son siempre JSON, mitigando XSS reflejado/almacenado.

### Respaldos (`src/backup.rs`)
- Copia **consistente** del SQLite (`VACUUM INTO`) + `snapshot.json` + segmentos del WAL a `backups/<timestamp>/`, con **retención configurable**.
- CLI: `luma backup` y `luma restore <ruta>`. Tarea de fondo opcional (`backup_enabled`) con `backup_interval_secs`.

### Auditoría y cifrado
- **Auditoría de acceso** (`src/api/audit.rs`): middleware que registra `ts, api_key_id, ip, method, path, status, latency_ms` en SQLite; consultable en `GET /v1/admin/audit`. La auditoría "semántica" de negocio (login, altas/bajas) se guarda en `sys_audit_events` y se consulta en `/v1/admin/audit-events`.
- **Cifrado en reposo** de campos sensibles con **ChaCha20-Poly1305** (AEAD), clave maestra derivada de `LUMA_MASTER_KEY`. Ciphertext auto-descriptivo `enc:v1:<b64(nonce||ct)>`.
- **Cabeceras de seguridad** en todas las respuestas: `Content-Security-Policy` estricta (sin `unsafe-inline` para scripts; jsdelivr permitido para la doc Scalar), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy` y **HSTS**.

---

## 🛠️ Compilar y correr

Requisitos: **Rust 1.88+** (edition 2021). SQLite va *bundled* (no requiere instalación externa).

```bash
# Compilar
cargo build --release

# (Opcional) recompilar el panel de administración e incrustarlo en ui/dist
cd admin-ui && npm ci && npm run build && cd ..

# Arrancar el servidor (sirve API + panel en http://127.0.0.1:1234/)
LUMA_MASTER_KEY="una-clave-secreta-fuerte" ./target/release/luma serve
```

El binario `luma` acepta los siguientes subcomandos (`src/cli.rs`, `src/main.rs`):

| Subcomando | Descripción |
| :--- | :--- |
| `luma serve` | Arranca el servidor HTTP (comando por defecto si no se pasa ninguno). |
| `luma vacuum --collection <nombre>` | Compacta una colección vectorial. |
| `luma diskann build …` / `tune …` / `status` | Construye, ajusta o consulta el estado de un grafo DiskANN (`src/diskann.rs`). |
| `luma backup` | Genera un respaldo consistente (SQLite + snapshot + WAL). |
| `luma restore <ruta>` | Restaura desde un directorio de respaldo. |

### Panel y primer acceso

```bash
# Crear tu organización y entrar
curl -X POST localhost:1234/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"org_name":"Acme","email":"owner@acme.com","password":"un-password-fuerte"}'
```

Luego abre `http://127.0.0.1:1234/` en el navegador para el panel React, o `http://127.0.0.1:1234/docs` para la documentación interactiva (Scalar).

> **Producción:** define siempre `LUMA_MASTER_KEY` (cifrado) y `LUMA_API_KEY` (bootstrap). Sin `LUMA_MASTER_KEY` se usa una clave de desarrollo conocida y el servidor lo advierte en los logs. El puerto por defecto es **1234** con bind a `127.0.0.1` (`luma.toml`); las variables de entorno sobrescriben el TOML.

### Configuración

`luma.toml` en la raíz (auto-generado si falta). Secciones clave: servidor (`port`, `bind_addr`, `api_key`), almacenamiento (`data_dir`, `snapshot_interval_secs`, `wal_segment_max_bytes`), vector (`index_kind` = `HNSW`|`IVF_FLAT_Q8`|`DiskANN`, `max_vector_dim`), IVF/DiskANN, embeddings, búsqueda (`pre_filter_threshold`), NS-Mem (`memory_*`), grafo/decay y respaldos (`backup_*`). Se puede leer/actualizar en caliente vía `GET`/`PUT /v1/config`. Fuente: `src/config.rs`.

---

## 🧭 Niveles de API

El router (`src/api/mod.rs`) monta las siguientes rutas. Todas requieren `Authorization: Bearer <api_key|token>` salvo `register`/`login`, `/v1/health`, `/docs` y los assets del panel.

### Nivel 1: Endpoints primitivos

Cada motor funciona de forma aislada, para máxima velocidad y mínimo overhead.

- **Vectorial** — `/v1/vector/...`: listar/crear colecciones; `add`, `upsert`, `upsert_batch`, `update`, `delete`, `delete_batch`, `get`; `search`, `search_batch` (hasta 100 queries en paralelo con `rayon`), `scroll` (paginación por cursor), `rerank` (reordenamiento por coseno), `aggregate` (conteos por campo); `diskann/build`, `diskann/tune`, `diskann/status`.
- **Documentos JSON** — `/v1/doc/{collection}/{id}` (`PUT`/`GET`/`DELETE`) y `/v1/doc/{collection}/find`.
- **Clave-Valor** — `/v1/state/...`: `GET`/`PUT`/`DELETE` por clave, `batch_put`, índices (`indexes`, `index/{field}/{value}`), listado y TTL/CAS.
- **Object storage (R2-like)** — `/v1/blob/{bucket}/{key}` (`PUT`/`GET`/`DELETE`) y listado por bucket. Escritura atómica, endurecido contra path-traversal.
- **Colas** — `/v1/queue/{queue}` (encolar y stats), `/receive` (entrega *at-least-once* con *visibility timeout*), `DELETE /{id}` (ack).
- **Imágenes** — `GET /v1/image/{bucket}/{key}?w=&h=&format=&quality=`: resize (Lanczos3) + convert (`png`/`jpeg`) sobre objetos ya guardados en el blob store.
- **Eventos** — `GET /v1/events` y `GET /v1/stream` (SSE en vivo con señal de gap).
- **Búsqueda de texto** — `POST /search` y `POST /search/ingest` (motor `src/search/`).

> El SQL relacional se usa **internamente** (pre-filtro del hub, NS-Mem, auth, auditoría) y opcionalmente contra un backend libSQL/Turso remoto; no se expone una ruta de query SQL cruda en el router actual.

### Nivel 2: LumaDatabase Hub (RAG híbrido) — `/v1/db/{namespace}`

El orquestador `LumaDatabase` (`src/engine/hub.rs`) fusiona los motores: segmenta documentos grandes (chunking), se conecta al modelo de embeddings configurado, crea la colección si no existe, guarda vectores y persiste metadatos en SQLite de forma transaccional (con *rollback* si falla I/O).

- **`POST /v1/db/{namespace}/ingest`** — ingesta de `{ id, text, metadata }`: chunking → embedding → upsert.
- **`POST /v1/db/{namespace}/search`** — búsqueda híbrida: **pre-filtro SQL** estricto (100% de precisión) antes de la fase vectorial, luego colapsa chunks y devuelve el documento padre hidratado.

```json
{
  "query": "cláusula sobre el precio del alquiler",
  "limit": 5,
  "sql_filter": "json_extract(metadata, '$.tipo') = 'alquiler' AND json_extract(metadata, '$.year') = 2024"
}
```

### Nivel 3: NS-Mem — Memoria de agentes — `/v1/memory/{namespace}`

Capa de memoria completa para agentes autónomos (`src/memory/`), construida sobre el stack convergente.

| Tipo | Almacenamiento | Descripción |
| :--- | :--- | :--- |
| **episodic** | Vector + SQLite | Eventos e interacciones concretas indexadas para recall semántico |
| **semantic** | Vector + SQLite | Hechos y preferencias estables, promovidos desde episodic vía LLM |
| **procedural** | SQLite (DAG) | Flujos con nodos, aristas tipadas y evaluación de constraints |
| **working** | KV + TTL | Contexto efímero de sesión, expira automáticamente |

**Pipeline de consolidación**: `ingest_event` → extracción de facts (LLM o heurística local) → `semantic` (`active` si confianza ≥ 0.85, si no `draft`), creando una arista `TriggeredBy` (episodic → semantic).

**Recall (semantic walk)**: seeds K-NN → expansión BFS por aristas tipadas → ranking por `coseno × edge_factor × (1 + centralidad PageRank)` → filtra archivados → top-k.

**Endpoints**:
- `POST ingest_event` · `POST upsert_fact` · `POST upsert_procedure`
- `POST query` (recall híbrido) · `POST next_step` (siguiente nodo válido del DAG)
- `GET timeline/{entity_id}`
- `POST edges` · `GET edges/{memory_id}` · `POST edges/{edge_id}/delete`
- `GET beliefs/{fact_key}/history` · `POST graph/centrality`

Además: deduplicación de facts (cosine ≥ 0.95), decay exponencial opt-in (`memory_decay_enabled`) y detección de contradicciones (arista `Contradicts` si la similitud viejo↔nuevo < 0.55). Proveedores LLM: `none`, `mock`, `openai`, `ollama`. Ver `docs/NS_MEM.md`.

### Administración y salud

- `GET /v1/health`, `GET /v1/metrics` (percentiles p50/p95/p99).
- `POST /v1/admin/backup` (dispara snapshot), `GET /v1/admin/audit` (log de acceso filtrable) — requieren rol `admin`.
- API keys y RBAC: `/v1/auth/keys`, `/v1/auth/roles`.
- `GET`/`PUT /v1/config`.

---

## 🔌 Embeddings (BYOM — Bring Your Own Model)

Para no engordar el binario con librerías pesadas de C++, Luma usa un cliente HTTP ligero con reintentos automáticos. Soporta 6 proveedores además de `none` (sin embedding server-side, valor por defecto):

| Provider | Variables | Notas |
| :--- | :--- | :--- |
| `ollama` | `EMBEDDING_URL`, `EMBEDDING_MODEL` | Local, sin API key |
| `openai` | `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | Batching ≤ 96 |
| `azure` | `EMBEDDING_AZURE_API_BASE`, `EMBEDDING_AZURE_DEPLOYMENT` | `api-key` header |
| `cohere` | `EMBEDDING_API_KEY`, `EMBEDDING_COHERE_INPUT_TYPE` | `search_document` / `search_query` |
| `huggingface` | `EMBEDDING_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | Inference API |
| `mock` | `EMBEDDING_DIM` | Tests/CI sin red |

Retry con backoff exponencial + jitter: `EMBEDDING_RETRY_ATTEMPTS` (default 3), `EMBEDDING_RETRY_INITIAL_MS` (default 200).

---

## 🧰 Tecnologías clave

| Componente | Crate | Rol |
| :--- | :--- | :--- |
| Runtime async / HTTP | `tokio`, `axum` 0.7, `hyper` 1 | I/O no bloqueante y router |
| Core KV | `redb` | Persistencia ACID en Rust puro |
| Relacional | `rusqlite` (SQLite bundled) | SQL embebido en modo WAL |
| Vectores | `hnsw_rs` + IVF/DiskANN a medida | Índices ANN conmutables |
| Panel embebido | `rust-embed` + React/Vite | SPA servida desde `ui/dist` |
| Seguridad | `argon2`, `chacha20poly1305`, `rustls` | Hashing, cifrado, TLS |
| Parseo | `pdf-extract`, `docx-rs`, `image` | Ingesta de formatos ricos |

---

## 🗂️ Layout en disco

```
data/
├── events-000001.log          # WAL segmentado (JSON lines)
├── snapshot.json              # Último snapshot de estado
├── vectors/<collection>/       # manifest.json, vectors.bin (mmap), diskann/
└── sqlite/rustkiss.db          # Relacional + auth + docstore + tablas NS-Mem y sys_*
backups/<timestamp>/           # Respaldos (VACUUM INTO + snapshot + WAL)
```

---

## ✅ Estado actual del proyecto

Implementado y verificable en el código de hoy (crate `luma` v4.10.0):

- **Núcleo convergente**: motor vectorial (HNSW / IVF-FLAT-Q8 / DiskANN), KV con TTL/CAS, WAL segmentado + snapshots, SQLite embebido vía actor, bus de eventos SSE, hub RAG híbrido y motor de búsqueda de texto. Todo montado en el router y cubierto por `tests/`.
- **Object storage, colas e imágenes**: primitivas tipo R2 + Queues + Images ya montadas en `/v1/blob`, `/v1/queue`, `/v1/image`.
- **NS-Mem**: memoria de agentes con grafo tipado, semantic walk BFS, PageRank, versionado de beliefs, deduplicación y detección de contradicciones. Decay opt-in.
- **Capa empresarial**: cuentas/orgs/usuarios, roles owner/admin/member/viewer, login Argon2id + tokens de sesión, aislamiento multi-tenant por organización, auditoría, cifrado en reposo, respaldos (CLI + tarea de fondo) y **panel de administración React realmente compilado e incrustado** (`admin-ui/` → `ui/dist`).
- **Operación**: TLS opcional, rate limiting opt-in, CORS configurable, timeouts, cabeceras de seguridad y documentación Scalar en `/docs`.

Notas de honestidad:
- Varias capacidades pesadas vienen **deshabilitadas por defecto** (`embedding_provider = "none"`, consolidación/decay/centralidad de memoria, `rate_limit_rps = 0`) — es un perfil de desarrollo; en producción se activan por configuración/entorno.
- El backend remoto **libSQL/Turso** solo se activa si `LIBSQL_URL` está definido; de lo contrario se usa el SQLite local.
- La durabilidad depende de montar un volumen persistente para `data_dir` cuando se corre en contenedor (`FROM scratch`).

---

## 🏁 Conclusión

Luma redefine el backend para IA mediante la **convergencia**: orquesta motores de primer nivel (índices vectoriales, SQLite, redb) más una capa empresarial completa en una sola plataforma cohesionada y un solo binario.

> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**

Proyecto de prueba interno
Estado verificado el 15 de julio
Ultima revision automatica
