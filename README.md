<!-- RustKissVDB: el motor de datos convergente en Rust que impulsa Luma (búsqueda vectorial + KV + SQL + eventos en un solo binario). -->
# Luma: La Plataforma de Datos Convergente

[![CI](https://github.com/Jairodaniel-17/rust-kiss-vdb/actions/workflows/ci.yml/badge.svg)](https://github.com/Jairodaniel-17/rust-kiss-vdb/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/Jairodaniel-17/rust-kiss-vdb?label=release&sort=semver)](https://github.com/Jairodaniel-17/rust-kiss-vdb/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Python SDK](https://img.shields.io/badge/python-luma--vdb-3775A9.svg)](sdk/)
[![TS SDK](https://img.shields.io/badge/npm-luma--vdb-CB3837.svg)](sdk/typescript/)

**Luma** (crate Cargo `luma`, versión **4.24.0**, *powered by RustKissVDB*) no es solo una base de datos vectorial. Es un **Motor de Datos Convergente** escrito en **Rust** que unifica, en un **único binario** (`luma`), las primitivas que necesita una aplicación de IA moderna:

- **Búsqueda vectorial** (ANN) con índices conmutables HNSW / IVF-FLAT-Q8 / DiskANN.
- **Estado clave-valor** (KV) con TTL, compare-and-swap e índices.
- **SQL relacional** embebido (SQLite en modo WAL), con opción de backend remoto libSQL/Turso.
- **Document store** JSON y **object storage** binario tipo R2.
- **Colas** durables e **imágenes** transformadas on-the-fly.
- **Bus de eventos** pub/sub con streaming SSE.
- **NS-Mem**: una capa de memoria para agentes autónomos (episódica, semántica, procedural y de trabajo).
- **Capa empresarial**: cuentas/organizaciones, roles, login con Argon2id, auditoría, respaldos, cifrado en reposo y un **panel de administración React incrustado en el propio binario**.
- **SDKs oficiales**: Python (async + sync), TypeScript/JS e integración **LangChain**.

Todo corre en un mismo proceso, eliminando la latencia de red entre subsistemas y simplificando el despliegue a un solo ejecutable.

*Read this in [English](README.en.md).*

---

## Índice

- [Instalación](#instalacion)
- [Quickstart en 60 segundos](#quickstart)
- [SDKs oficiales](#sdks)
- [¿Por qué Luma?](#por-que-luma)
- [Superficie de plataforma](#superficie)
- [Rendimiento medido](#rendimiento)
- [Arquitectura por módulos](#arquitectura)
- [Capa empresarial](#enterprise)
- [Niveles de API](#api)
- [Embeddings (BYOM)](#embeddings)
- [Configuración](#configuracion)
- [Tecnologías clave](#tecnologias)
- [Layout en disco](#layout)
- [Estado actual del proyecto](#estado)
- [Documentación](#documentacion)
- [Licencia](#licencia)

---

<a id="instalacion"></a>

## 📦 Instalación

### Binario precompilado (vía más rápida)

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.sh | bash

# Versión concreta o destino alternativo
curl -fsSL .../install.sh | bash -s -- --version v4.24.0 --dest ~/.local/bin
```

```powershell
# Windows
irm https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.ps1 | iex
```

Los scripts descargan el binario del release de GitHub para tu plataforma (Linux, Windows, macOS) y lo colocan en el `PATH`.

### Docker

```bash
# docker-compose (incluye volumen persistente en ./data_storage)
LUMA_API_KEY=mi-clave docker compose up -d

# o imagen directa
docker build -t luma:latest .
docker run -p 1234:1234 -v $PWD/data_storage:/data \
  -e DATA_DIR=/data -e LUMA_API_KEY=mi-clave -e LUMA_MASTER_KEY=clave-fuerte \
  luma:latest
```

Hay dos Dockerfiles: `Dockerfile` (glibc) y `Dockerfile.musl` (estático, imagen mínima). El `docker-compose.yml` ya expone las variables de rate limiting, TLS, embeddings y respaldos.

> ⚠️ **La durabilidad exige montar un volumen persistente en `DATA_DIR`.** Sin él, el WAL y los vectores viven en la capa efímera del contenedor.

### Desde el código fuente

Requisitos: **Rust 1.88+** (edition 2021). SQLite va *bundled*, no requiere instalación externa.

```bash
cargo build --release

# (Opcional) recompilar el panel de administración e incrustarlo en ui/dist
cd admin-ui && npm ci && npm run build && cd ..
```

---

<a id="quickstart"></a>

## ⚡ Quickstart en 60 segundos

```bash
# 1. Arrancar (sirve API + panel en http://127.0.0.1:1234/)
LUMA_MASTER_KEY="clave-secreta-fuerte" LUMA_API_KEY="mi-api-key" \
  ./target/release/luma serve
```

```bash
# 2. Crear una colección vectorial (dimensión + métrica)
curl -X POST localhost:1234/v1/vector/docs \
  -H 'authorization: Bearer mi-api-key' -H 'content-type: application/json' \
  -d '{"dim":4,"metric":"cosine"}'

# 3. Insertar un vector con metadatos
curl -X POST localhost:1234/v1/vector/docs/upsert \
  -H 'authorization: Bearer mi-api-key' -H 'content-type: application/json' \
  -d '{"id":"doc-1","vector":[0.1,0.2,0.3,0.4],"meta":{"tipo":"contrato","year":2024}}'

# 4. Buscar los k más cercanos
curl -X POST localhost:1234/v1/vector/docs/search \
  -H 'authorization: Bearer mi-api-key' -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4],"k":5}'

# 5. Buscar con filtro tipado por metadatos
curl -X POST localhost:1234/v1/vector/docs/search \
  -H 'authorization: Bearer mi-api-key' -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3,0.4],"k":5,
       "options":{"filter":{"eq":{"field":"tipo","value":"contrato"}}}}'
```

### Panel de administración

```bash
# Crear tu organización y entrar
curl -X POST localhost:1234/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"org_name":"Acme","email":"owner@acme.com","password":"un-password-fuerte"}'
```

Abre `http://127.0.0.1:1234/` para el panel React, o `http://127.0.0.1:1234/docs` para la documentación interactiva (Scalar).

> **Producción:** define siempre `LUMA_MASTER_KEY` (cifrado) y `LUMA_API_KEY` (bootstrap). Sin `LUMA_MASTER_KEY` se usa una clave de desarrollo conocida y el servidor lo advierte en los logs. El puerto por defecto es **1234** con bind a `127.0.0.1`; las variables de entorno sobrescriben `luma.toml`.

### Subcomandos del binario

| Subcomando | Descripción |
| :--- | :--- |
| `luma serve` | Arranca el servidor HTTP (comando por defecto si no se pasa ninguno). |
| `luma vacuum --collection <nombre>` | Compacta una colección vectorial. |
| `luma diskann build …` / `tune …` / `status <colección>` | Construye, ajusta o consulta el estado de un grafo DiskANN. |
| `luma backup` | Genera un respaldo consistente (SQLite + snapshot + WAL). |
| `luma restore <ruta>` | Restaura desde un directorio de respaldo. |

---

<a id="sdks"></a>

## 🐍 SDKs oficiales

Tres clientes mantenidos en este repo, todos contra la misma API HTTP.

### Python — `sdk/` (paquete `luma-vdb`)

Async y sync, con `py.typed`. Sub-clientes: `vector`, `state`, `doc`, `admin`, `auth`, `stream`, `config`, `hub(ns)`, `memory(ns)`, `meta(c)`, `diskann(c)`.

```python
from luma import Luma          # async
from luma import SyncLuma      # sync

luma = Luma("http://localhost:1234", api_key="mi-api-key")
await luma.vector.acreate("embeddings", dim=1536)
await luma.vector.aupsert("embeddings", "doc-1", [0.1] * 1536, meta={"tipo": "contrato"})
hits = await luma.vector.asearch("embeddings", [0.1] * 1536, k=5)

# Memoria de agentes (NS-Mem)
mem = luma.memory("mi-agente")
await mem.aingest_event(text="El usuario prefiere respuestas cortas")
recall = await mem.aquery(text="¿cómo le gusta que le responda?")
```

```python
with SyncLuma("http://localhost:1234", api_key="mi-api-key") as db:
    db.vector.create("embeddings", dim=1536)
    hits = db.vector.search("embeddings", [0.1] * 1536, k=5)
```

### TypeScript / JavaScript — `sdk/typescript/` (paquete `luma-vdb`)

Node 18+ (usa `fetch` nativo). Funciona también en navegador y edge runtimes (Cloudflare Workers, Deno, Bun).

```bash
npm install luma-vdb
```

```typescript
import { LumaClient } from 'luma-vdb';

const client = new LumaClient({ baseUrl: 'http://localhost:1234', apiKey: 'mi-api-key' });

await client.vector.createCollection('docs', 384, 'cosine');
await client.vector.upsert('docs', 'item-1', vector, { category: 'tech' });
await client.vector.upsertBatch('docs', [
  { id: 'a', vector: [0.1, 0.2], meta: { tag: 'ai' } },
]);
```

### LangChain — `sdk/langchain_luma/`

`LumaVectorStore` implementa la interfaz `VectorStore` de `langchain_core`, incluyendo **MMR** (maximal marginal relevance). Crea la colección sola si no existe.

```python
from langchain_luma import LumaVectorStore

store = LumaVectorStore(
    url="http://localhost:1234", api_key="mi-api-key",
    collection="rag", embedding=mi_embedding, dim=1536,
)
store.add_texts(["...", "..."])
docs = store.max_marginal_relevance_search("mi pregunta", k=4)
```

Guía completa del SDK Python: [`docs/empezar/SDK_PYTHON.md`](docs/empezar/SDK_PYTHON.md).

---

<a id="por-que-luma"></a>

## 🚀 ¿Por qué Luma?

La premisa es simple: **la IA necesita más que vectores.** Mientras la arquitectura tradicional fragmenta el stack (PostgreSQL para datos, Redis para caché/colas, un servicio aparte para vectores), Luma converge esas primitivas en un binario Rust, con seguridad de memoria, concurrencia sobre Tokio y latencia interna cero entre motores.

---

<a id="superficie"></a>

## 🧰 Superficie de plataforma: qué reemplaza cada primitiva

Luma no es solo un motor vectorial — es una capa de servicios de plataforma. Cada primitiva está montada hoy en el router y cubierta por tests:

| Primitiva | Endpoint | Reemplaza a | Semántica clave |
|---|---|---|---|
| **Object storage** | `/v1/blob/:bucket/:key` | S3 / R2 | buckets y objetos binarios |
| **KV** | `/v1/state` | Redis (datos) / DynamoDB | TTL, compare-and-swap por revisión, índices secundarios, 16 shards |
| **Colas durables** | `/v1/queue/:queue` | SQS / Cloudflare Queues | at-least-once, visibility timeout, intentos, respaldadas en disco |
| **Bus de eventos** | `/v1/stream` (SSE) | SNS / EventBridge | pub/sub con offsets del WAL, replay con `since=` |
| **Imágenes** | `/v1/image/:bucket/:key` | CloudFront + Lambda de imágenes | transformación on-the-fly |
| **Document store** | `/v1/doc/:collection` | MongoDB / DocumentDB | JSON con find por filtros |
| **Vectores + RAG híbrido** | `/v1/vector`, `/v1/db` | Qdrant / pgvector / OpenSearch | HNSW · IVF-FLAT-Q8 · DiskANN, rerank, búsqueda de texto |
| **Memoria de agentes** | `/v1/memory` (NS-Mem) | — (no existe equivalente gestionado) | grafo tipado, beliefs versionados, contradicciones |
| **SQL** | embebido | SQLite WAL (o libSQL/Turso remoto) | relacional ligero, no pretende reemplazar PostgreSQL |
| **Cuentas y control de acceso** | `/v1/auth`, `/v1/admin` | Cognito (parcial) | orgs, roles, Argon2id, auditoría, aislamiento por organización |

Todas las primitivas comparten WAL segmentado con checksums, snapshots, respaldos, cifrado en reposo y el aislamiento multi-tenant por organización.

> **Hoja de ruta de producto:** el plan maestro de endurecimiento — durabilidad verificada, réplica/WAL shipping, API S3-compatible, conector PostgreSQL por CDC, operabilidad y criterio de GA — está en [`docs/SPEC-producto.md`](docs/SPEC-producto.md), con el plan de ejecución en [`docs/PLAN-MAESTRO.md`](docs/PLAN-MAESTRO.md). El frente de **compatibilidad con el protocolo de Redis (RESP)** — que Celery, arq, redis-py o ioredis apunten a Luma **sin cambiar código** (`REDIS_URL=redis://luma:6379`) — está **implementado y experimental**: ver [`docs/integrar/RESP.md`](docs/integrar/RESP.md) para la matriz de comandos y las divergencias. Cómo operarlo: [`docs/operar/RUNBOOKS.md`](docs/operar/RUNBOOKS.md). Qué protegemos y de quién: [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md).

---

<a id="rendimiento"></a>

## 📊 Rendimiento medido

Comparativa contra **Qdrant** y **Milvus**: misma máquina, mismo dataset (50k × 768, cosine, k=10), ground-truth exacto por fuerza bruta, los tres motores con su configuración de fábrica.

| Motor / modo | Consulta (qps) | Latencia | Recall@10 | RAM |
|---|---:|---:|---:|---:|
| Qdrant (HNSW, ef=64) | 237 | 4.22 ms | 0.416 | 320 MB |
| Milvus (HNSW, ef=64) | 476 | 2.09 ms | 0.086 | 846 MB |
| **Luma — DiskANN** | **515** | **1.94 ms** | 0.056 | **133 MB** |
| **Luma — HNSW** (ef=32) | 199 | 5.02 ms | **0.474** | 406 MB |
| **Luma — HNSW** (ef=128, default) | 99 | 10.08 ms | 0.853 | 464 MB |

Tres titulares:

- 🏆 **RAM** — DiskANN corre 50k vectores en **133 MB**; ningún competidor baja de 320 MB. Es el objetivo de diseño y se cumple medido.
- 🏆 **Latencia de consulta** — **1.94 ms / 515 qps** en DiskANN, el más rápido del grupo.
- 🏆 **Precisión a igual velocidad** — a la latencia de Qdrant (`ef=32`), Luma HNSW da **más recall** (0.474 vs 0.416), y escala hasta 0.947 subiendo `ef`.

> **Cómo leer la columna de recall:** el dataset son vectores aleatorios uniformes, sin estructura de clusters — un caso adversarial para cualquier ANN (incluso Qdrant se queda en 0.42). Esa columna mide el **punto de equilibrio velocidad↔precisión que cada motor elige de fábrica**, no la calidad absoluta del índice. Con embeddings reales todos los recalls suben. Comparar recalls entre motores solo tiene sentido a latencia equivalente.

Metodología completa, curva de `ef`, la calibración de HNSW (3× throughput al mismo recall), la paralelización de ingesta (293 → 926 vec/s) y lo que aún va por detrás: **[`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md)**.

---

<a id="arquitectura"></a>

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

<a id="enterprise"></a>

## 🏢 Capa Empresarial: Multi-Tenancy, Panel y Seguridad

Capa "enterprise" **aditiva** montada sobre las primitivas del core. Todo vive en el mismo binario: no necesitas Node, ni el código fuente del panel, ni servicios externos en runtime. El `AccountsService` y las tablas `sys_*` se crean *lazily* la primera vez que se usan, siempre que SQLite esté habilitado.

### Cuentas, sesiones y roles (`src/api/accounts.rs`, `routes_accounts.rs`)
- **Organizaciones y usuarios** en SQLite (`sys_orgs`, `sys_users`), más `sys_sessions` para tokens de sesión y `sys_collections` para propiedad de recursos.
- **Login por email + contraseña**: las contraseñas se hashean con **Argon2id** (`src/crypto.rs`). El login emite un **token de sesión opaco** (`lums_…`) del que solo se guarda su hash SHA-256; TTL de 7 días.
- **Roles**: `owner` > `admin` > `member` > `viewer`, integrados con el RBAC existente (`rbac.rs`, niveles viewer=10, member=20, admin=30, owner=40). Un middleware exige rol mínimo por ruta.
- **Multi-organización**: un usuario puede pertenecer a varias orgs (`/v1/admin/users/:id/orgs`), con invitaciones (`/v1/admin/orgs/:id/invite`) y gestión de miembros por organización.

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

Política de reporte de vulnerabilidades: [`SECURITY.md`](SECURITY.md). Modelo de amenazas: [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md). Inventario de `unsafe` (16 sitios, todos en el motor vectorial; el resto del crate lo prohíbe en tiempo de compilación): [`docs/operar/SECURITY.md`](docs/operar/SECURITY.md).

---

<a id="api"></a>

## 🧭 Niveles de API

El router (`src/api/mod.rs`) monta las siguientes rutas. Todas requieren `Authorization: Bearer <api_key|token>` salvo `register`/`login`, `/v1/health`, `/docs` y los assets del panel.

### Nivel 1: Endpoints primitivos

Cada motor funciona de forma aislada, para máxima velocidad y mínimo overhead.

- **Vectorial** — `/v1/vector/...`: listar/crear/borrar colecciones y ver detalle; `add`, `upsert`, `upsert_batch`, `update`, `delete`, `delete_batch`, `get`; `search`, `search_batch` (hasta 100 queries en paralelo con `rayon`), `scroll` (paginación por cursor), `rerank` (reordenamiento por coseno), `aggregate` (conteos por campo); `diskann/build`, `diskann/tune`, `diskann/status`.
- **Documentos JSON** — `/v1/doc/{collection}/{id}` (`PUT`/`GET`/`DELETE`) y `/v1/doc/{collection}/find`.
- **Metadatos de colección** — `POST /v1/meta/{collection}/execute`: consultas sobre los metadatos de una colección (`routes_meta.rs`).
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

Además: deduplicación de facts (cosine ≥ 0.95), decay exponencial opt-in (`memory_decay_enabled`) y detección de contradicciones (arista `Contradicts` si la similitud viejo↔nuevo < 0.55). Proveedores LLM: `none`, `mock`, `openai`, `ollama`. Ver [`docs/integrar/NS_MEM.md`](docs/integrar/NS_MEM.md).

### Administración, auth y salud

- **Salud y métricas**: `GET /v1/health`, `GET /v1/metrics` (percentiles p50/p95/p99).
- **Cuentas**: `POST /v1/auth/register` · `login` · `logout` · `refresh`; `GET /v1/auth/sessions` (sesiones activas).
- **Orgs**: `/v1/admin/orgs` (listar/crear), `/v1/admin/orgs/:id` (detalle/actualizar), `/members`, `/invite`, `/members/:user_id`.
- **Usuarios**: `/v1/admin/users` (listar/crear), `/v1/admin/users/:id`, `/:id/role`, `/:id/orgs`.
- **API keys**: `/v1/auth/keys` (listar/crear), `/keys/:id` (revocar), `/keys/:id/role`.
- **RBAC**: `/v1/auth/roles` (listar/crear), `/roles/:id`, `/roles/:id/permissions`, `POST /roles/check`.
- **Operación**: `POST /v1/admin/backup` (dispara snapshot), `GET /v1/admin/audit` (log de acceso filtrable), `GET /v1/admin/stats`, `GET /v1/admin/audit-events` — requieren rol `admin`.
- **Configuración en caliente**: `GET`/`PUT /v1/config`, `POST /v1/config/embedding/probe` (verifica que el proveedor de embeddings responde).

Referencia completa: [`docs/integrar/API.md`](docs/integrar/API.md) y la spec OpenAPI en [`docs/openapi.yaml`](docs/openapi.yaml) (servida en `/docs`).

**Toda la documentación, ordenada por lo que estés intentando hacer —
empezar, operar, integrar, referencia — está en
[`docs/README.md`](docs/README.md).**

---

<a id="embeddings"></a>

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

Retry con backoff exponencial + jitter: `EMBEDDING_RETRY_ATTEMPTS` (default 3), `EMBEDDING_RETRY_INITIAL_MS` (default 200). Puedes validar la conexión con `POST /v1/config/embedding/probe`.

---

<a id="configuracion"></a>

## ⚙️ Configuración

`luma.toml` en la raíz (auto-generado si falta). Las variables de entorno **sobrescriben** el TOML. Fuente: `src/config.rs`.

| Sección | Claves |
| :--- | :--- |
| Servidor | `port` (1234), `bind_addr` (127.0.0.1), `api_key` |
| Almacenamiento | `data_dir`, `snapshot_interval_secs`, `wal_segment_max_bytes` |
| Vector | `index_kind` (**`IVF_FLAT_Q8` por defecto** · `HNSW` · `DiskANN`), `max_vector_dim`, `simd_enabled`, `HNSW_SEARCH_EF` (128) |
| IVF | `ivf_clusters`, `ivf_nprobe`, `q8_refine_topk` |
| DiskANN | `diskann_max_degree`, `diskann_build_threads` |
| Embeddings | `embedding_provider`, `embedding_model`, `embedding_dim`, `embedding_retry_*` |
| Búsqueda | `pre_filter_threshold` (10 000) |
| NS-Mem | `memory_consolidation_enabled`, `memory_working_ttl_secs`, `memory_fact_promotion_threshold` (0.85), `llm_provider` |
| Grafo | `memory_walk_max_hops` (2), `memory_walk_min_similarity` (0.65), `memory_centrality_enabled` |
| Decay | `memory_decay_enabled` (false), `memory_decay_half_life_days` (30), `memory_decay_archive_threshold` (0.1) |
| Respaldos | `backup_enabled`, `backup_dir`, `backup_interval_secs`, `backup_retention` |
| Operación | `rate_limit_rps` (0 = off), `rate_limit_burst`, `TLS_CERT_PATH`, `TLS_KEY_PATH`, `LIBSQL_URL` |

Se puede leer y actualizar en caliente vía `GET`/`PUT /v1/config`. Detalle completo: [`docs/operar/CONFIG.md`](docs/operar/CONFIG.md).

---

<a id="tecnologias"></a>

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

<a id="layout"></a>

## 🗂️ Layout en disco

```
data/
├── events-000001.log          # WAL segmentado (JSON lines)
├── snapshot.json              # Último snapshot de estado
├── vectors/<collection>/      # manifest.json, vectors.bin (mmap), diskann/
└── sqlite/rustkiss.db         # Relacional + auth + docstore + tablas NS-Mem y sys_*
backups/<timestamp>/           # Respaldos (VACUUM INTO + snapshot + WAL)
```

---

<a id="estado"></a>

## ✅ Estado actual del proyecto

Implementado y verificable en el código de hoy (crate `luma` v4.24.0):

- **Núcleo convergente**: motor vectorial (HNSW / IVF-FLAT-Q8 / DiskANN), KV con TTL/CAS, WAL segmentado + snapshots, SQLite embebido vía actor, bus de eventos SSE, hub RAG híbrido y motor de búsqueda de texto. Todo montado en el router y cubierto por `tests/`.
- **Object storage, colas e imágenes**: primitivas tipo R2 + Queues + Images ya montadas en `/v1/blob`, `/v1/queue`, `/v1/image`.
- **NS-Mem**: memoria de agentes con grafo tipado, semantic walk BFS, PageRank, versionado de beliefs, deduplicación y detección de contradicciones. Decay opt-in.
- **Capa empresarial**: cuentas/orgs/usuarios, roles owner/admin/member/viewer, login Argon2id + tokens de sesión, aislamiento multi-tenant por organización, auditoría, cifrado en reposo, respaldos (CLI + tarea de fondo) y **panel de administración React realmente compilado e incrustado** (`admin-ui/` → `ui/dist`).
- **SDKs**: cliente Python (async + sync, `py.typed`), cliente TypeScript y `LumaVectorStore` para LangChain, todos en `sdk/`.
- **Operación**: TLS opcional, rate limiting opt-in, CORS configurable, timeouts, cabeceras de seguridad, instaladores para Linux/macOS/Windows, dos Dockerfiles + compose, y documentación Scalar en `/docs`.

Notas de honestidad:

- Varias capacidades pesadas vienen **deshabilitadas por defecto** (`embedding_provider = "none"`, consolidación/decay/centralidad de memoria, `rate_limit_rps = 0`) — es un perfil de desarrollo; en producción se activan por configuración/entorno.
- El backend remoto **libSQL/Turso** solo se activa si `LIBSQL_URL` está definido; de lo contrario se usa el SQLite local.
- La durabilidad depende de montar un volumen persistente para `data_dir` cuando se corre en contenedor.
- La **comparativa contra Qdrant/Milvus** se hizo con scripts ad-hoc que **no están versionados** en el repo: las cifras son las observadas, pero hoy no se reejecutan con un comando del repositorio. Los benchmarks *internos* (`src/bin/bench.rs`) sí son reproducibles.
- La **compatibilidad con el protocolo Redis (RESP)** está implementada y marcada **experimental**: 57/57 comandos de las fases 2 y 3 del SPEC, verificados byte a byte contra un Redis 7 real (330 comandos, 0 divergencias) y con Celery, kombu, arq y redis-py reales — incluido un worker Celery que consume, ejecuta y devuelve el resultado. Sigue siendo experimental hasta que el nightly esté verde 7 días seguidos, que es el criterio del plan. Divergencias conocidas y qué **no** está: [`docs/integrar/RESP.md`](docs/integrar/RESP.md).
- **Escrituras por el listener RESP: 22.989 `SET`/s** (25.685 `GET`/s) en SSD NVMe, medidas con `redis-benchmark` a 256 clientes y **sin bajar ninguna garantía de durabilidad** (`wal_sync_mode = "per_write"`): son 29× el punto de partida de 785/s. Por el **mismo camino de red** y con el mismo cliente, Redis 7 da 28.517 `SET`/s y 27.298 `GET`/s — o sea Luma al 81% de su escritura y al 94% de su lectura, **haciendo fsync de cada escritura confirmada, que Redis por defecto no hace**. Dos advertencias que importan más que los totales: el control de PING (26.178/s Luma, 28.763/s Redis) dice que en esta ruta los dos van contra el techo del transporte, y en **disco mecánico** el `SET` de Luma cae a 3.142/s. Desglose capa por capa y los dos diseños que se midieron y se descartaron: [`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md#camino-de-escritura-kv--resp).

---

<a id="documentacion"></a>

## 📚 Documentación

| Documento | Contenido |
| :--- | :--- |
| [`docs/integrar/API.md`](docs/integrar/API.md) | Referencia de endpoints |
| [`docs/openapi.yaml`](docs/openapi.yaml) | Spec OpenAPI (servida en `/docs`) |
| [`docs/referencia/ARCHITECTURE.md`](docs/referencia/ARCHITECTURE.md) | Arquitectura interna |
| [`docs/operar/CONFIG.md`](docs/operar/CONFIG.md) | Todas las claves de configuración |
| [`docs/operar/CLI.md`](docs/operar/CLI.md) | Subcomandos del binario |
| [`docs/referencia/DATA_MODELS.md`](docs/referencia/DATA_MODELS.md) | Modelos de datos y esquemas |
| [`docs/referencia/VECTOR_STORAGE.md`](docs/referencia/VECTOR_STORAGE.md) | Segmentos, mmap, cuantización |
| [`docs/integrar/RESP.md`](docs/integrar/RESP.md) | Compatibilidad con el protocolo de Redis: comandos, divergencias y cómo validarla |
| [`docs/integrar/NS_MEM.md`](docs/integrar/NS_MEM.md) | Memoria de agentes (API completa) |
| [`docs/empezar/SDK_PYTHON.md`](docs/empezar/SDK_PYTHON.md) | Guía del SDK Python |
| [`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md) | Comparativa vs Qdrant/Milvus, camino de escritura KV/RESP (el 29×, capa por capa, y el contraste con Redis 7) y benchmarks internos |
| [`docs/referencia/BENCH.md`](docs/referencia/BENCH.md) | Cómo correr el binario de bench |
| [`docs/empezar/FEATURES.md`](docs/empezar/FEATURES.md) | Inventario de funcionalidades |
| [`docs/operar/SECURITY.md`](docs/operar/SECURITY.md) · [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md) | Seguridad y modelo de amenazas |
| [`docs/operar/PROD_READINESS.md`](docs/operar/PROD_READINESS.md) | Checklist de producción |
| [`docs/PLAN-MAESTRO.md`](docs/PLAN-MAESTRO.md) | **Plan de ejecución unificado** (bloques, orden, estado) |
| [`docs/SPEC-producto.md`](docs/SPEC-producto.md) · [`docs/SPEC-resp.md`](docs/SPEC-resp.md) · [`docs/SPEC-roadmap.md`](docs/SPEC-roadmap.md) | SPEC de origen (detalle y criterios de aceptación) |
| [`docs/referencia/CHANGELOG.md`](docs/referencia/CHANGELOG.md) | Historial de versiones |
| [`docs/empezar/DEMO.md`](docs/empezar/DEMO.md) | Guion de demo |

---

<a id="licencia"></a>

## 📄 Licencia

[MIT](LICENSE) © Luma contributors.

Reporte de vulnerabilidades: ver [`SECURITY.md`](SECURITY.md).

---

## 🏁 Conclusión

Luma redefine el backend para IA mediante la **convergencia**: orquesta motores de primer nivel (índices vectoriales, SQLite, redb) más una capa empresarial completa en una sola plataforma cohesionada y un solo binario.

> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**
