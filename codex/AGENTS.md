# Proyecto: Rust SSE Real-Time DB + Vector Index (KISS)

Eres un agente senior de ingeniería (arquitectura + implementación) encargado de construir una **base de datos de tiempo real** en **Rust**, diseñada para ser **simple**, **rápida**, **fácil de escalar**, y usable “en vivo” (observabilidad y streaming). El sistema combina:

1. **State Store** (KV con TTL/versiones)
2. **Event Store** (log append-only de cambios)
3. **SSE Layer** (stream de eventos: “la voz del servidor”)
4. **Vector Store (módulo)** con operaciones completas (**add/update/delete/upsert/search**) y soporte de **dimensiones** y **métricas**.

Tienes disponibles **todos los frameworks de Rust** (tokio, axum, actix-web, tonic gRPC, tracing, etc.) y puedes elegir los que maximicen simplicidad y estabilidad.

## Filosofía y Principios (KISS > SOLID)

- **KISS primero**: menos capas, menos abstracciones, menos “pattern soup”.
- Solo introducir complejidad si existe una necesidad clara (medible).
- Evita “arquitectura astronauta”: construye un **MVP robusto** y extensible con cambios pequeños.
- **No** uses sobre-ingeniería por defecto (DI excesiva, factories, 20 traits por carpeta).
- Acepta que algunas decisiones serán “buenas y simples” antes que “perfectas y complejas”.

## Objetivo del Producto

Construir una “DB para todo” enfocada en:

- Estado en tiempo real (jobs, colas, sesiones, progreso, cache)
- Eventos en vivo vía SSE
- Búsqueda semántica vía embeddings (vector index)
- Operación simple: correr en una máquina y escalar con sharding/replicas después

## No-Objetivos (para evitar scope creep)

- No construir un motor SQL.
- No construir transacciones ACID complejas al inicio.
- No implementar cluster consensus tipo Raft en la v1.
- No reinventar Redis ni Mongo; esto es **State+Events+SSE + Vector module**.

---

# Arquitectura requerida (v1)

## Componentes

### 1) HTTP API (Control Plane)

Endpoints simples para comandos: `put/get/delete/list`, operaciones vectoriales, administración, health.

### 2) SSE (Data Plane)

Un endpoint SSE principal para suscripciones a eventos por:

- key exacta
- prefijo (namespace)
- tipo de evento
- job_id / tags (vía metadata)

### 3) State Engine (KV)

- Almacena `key -> value` (value tipo bytes/json)
- TTL opcional por clave
- Versionado monotónico (`revision`/`etag`)
- Operaciones: `GET`, `PUT`, `UPSERT`, `DELETE`, `CAS` (compare-and-set) opcional
- Prefijos/Namespaces: `job:*`, `user:*`, `cache:*`

### 4) Event Store (append-only)

- Cada mutación del estado genera un evento
- Persistencia mínima: log rotativo + snapshots opcionales
- Permite replay básico para recuperación

### 5) Vector Module (Index + Storage)

Debe soportar:

- `vector.add(id, vector, meta)`
- `vector.upsert(id, vector, meta)`
- `vector.update(id, vector|meta)`
- `vector.delete(id)`
- `vector.get(id)`
- `vector.search(query_vector, k, filters?, namespace?)`
- Validación estricta de **dimensión** por índice/colección
- Métricas: `cosine` y `dot` como mínimo (L2 opcional)
- Namespaces/collections: `docs`, `code`, `customers`, etc.
- Metadata: JSON pequeño (tags, source, job_id, timestamps)
- Estrategia KISS:
  - v1: índice **HNSW** (si es simple de integrar) o **brute-force** (si no)
  - Debe existir modo fallback: “small-index brute force”
  - Mantener interfaz estable para swap de implementación futura

---

# API Especificación (mínima, clara)

## Autenticación

- La v1 actual expone los endpoints sin autenticación interna. Documenta cómo protegerlos detrás de un proxy o deja hooks claros para reactivar `Authorization: Bearer <key>` cuando sea necesario.
- Rate limiting opcional simple (token bucket) sigue siendo deseable cuando se reintroduzca auth.

## HTTP Endpoints sugeridos

### State

- `PUT /v1/state/{key}` body: `{ value, ttl_ms?, if_revision? }`
- `GET /v1/state/{key}`
- `DELETE /v1/state/{key}`
- `GET /v1/state?prefix=job:&limit=100`

### Events (SSE)

- `GET /v1/events?prefix=job:&types=state_updated,progress&since=...`
  - Debe soportar reconexión: `Last-Event-ID` y/o `since` (timestamp/revision)

### Vector

- `POST /v1/vector/{collection}/add`
- `POST /v1/vector/{collection}/upsert`
- `POST /v1/vector/{collection}/update`
- `POST /v1/vector/{collection}/delete`
- `GET  /v1/vector/{collection}/get?id=...`
- `POST /v1/vector/{collection}/search` body: `{ vector, k, filters?, include_meta? }`
- `POST /v1/vector/{collection}` create collection: `{ dim, metric }`

## Formato de eventos SSE (estándar)

Cada evento debe incluir:

- `event:` tipo (`state_updated`, `state_deleted`, `vector_added`, `vector_updated`, `vector_deleted`, `vector_search`, `progress`, `log`)
- `id:` incremental (u64) global o por stream
- `data:` JSON

Ejemplo `data`:

```json
{
  "ts": 1733950000,
  "type": "state_updated",
  "key": "job:123",
  "revision": 18,
  "patch": { "progress": 42 }
}
```

---

# Requisitos de Escalabilidad (KISS-friendly)

Implementa primero **single-node** excelente.
Deja ganchos claros para escalar después sin reescribir todo:

- Separar “API stateless” vs “Engine stateful” (en módulos)
- Namespaces para sharding futuro (por prefijo/collection)
- Event log para replicación eventual (no consensus en v1)
- Configuración por archivo/env sencilla

---

# Persistencia (v1 pragmática)

- Estado en memoria (rápido) + snapshot periódico opcional
- WAL/event log append-only para recuperar
- Política simple:
  - cada N segundos snapshot
  - log rotativo por tamaño

- Si se complica, mantener persistencia opcional (feature flag)

---

# Observabilidad y Operación en Vivo

- Logging con `tracing`
- Métricas simples (requests, latency, connected SSE clients)
- Endpoint: `GET /v1/health` y `GET /v1/metrics`
- Modo “dev” con dashboard textual simple opcional

---

# Reglas de Implementación (para evitar complejidad)

- Máximo 1–2 niveles de carpetas.
- Evitar “framework wars”: elige **uno** (recomendado: `axum + tokio`).
- Pocas abstracciones:
  - Un trait `VectorIndex` si y solo si es necesario
  - Tipos concretos primero, traits después

- Código legible > genérico.

---

# Entregables del Agente (obligatorios)

1. **Diseño**: `ARCHITECTURE.md` con decisiones y tradeoffs (corto y claro).
2. **API**: `openapi.yaml` o `API.md` con ejemplos curl.
3. **Repo** listo:
   - `src/main.rs`
   - `src/api/` (routes)
   - `src/engine/` (state + events)
   - `src/vector/` (collections, index, storage)

4. **Tests**:
   - unit tests (state + vector)
   - integration tests (HTTP + SSE)

5. **Demo**:
   - script que crea colección vectorial, agrega embeddings, hace search, y muestra SSE en vivo.

6. **Bench básico**:
   - latencia p50/p95 de put/get y vector search (aunque sea simple).

---

# Criterios de Aceptación (Definition of Done)

- SSE funciona con múltiples clientes, reconecta bien, no se cuelga.
- `vector.search` valida dimensión y métrica; devuelve top-k correcto.
- `add/upsert/update/delete` actualizan index y emiten eventos SSE.
- Manejo de errores claro (HTTP codes + JSON error).
- Config simple: `PORT_RUST_KISS_VDB`, `SNAPSHOT_INTERVAL`, `MAX_LOG_MB`, etc. (`API_KEY` queda opcional mientras auth siga desactivada).
- Documentación suficiente para que alguien lo use sin preguntarte nada.

---

# Plan de Ejecución (el agente debe seguirlo)

1. MVP State + SSE (sin vector) + auth + eventos
2. Event store + id incremental + reconexión
3. Vector collections (create, dim, metric)
4. Vector add/upsert/update/delete + storage + SSE events
5. Vector search (brute force o HNSW) + filtros básicos
6. Persistencia opcional (snapshot/log)
7. Tests + demo + docs + bench

---

# Nota final (importante)

Mantén el espíritu:

- **“Simple pero sólido”**
- **“Menos magia, más sistema usable”**
- **“Lo que no se usa, no se construye”**

Tu salida debe ser un repositorio funcional con documentación y demos, no solo teoría.
