# Features de Luma

Resumen de capacidades actuales del proyecto, organizado por subsistema.

## Core Engine

- KV store en Rust con TTL, revisión monotónica y operaciones CAS simples
- event bus con offsets globales y SSE
- WAL segmentado con replay y snapshot
- recuperación tras restart
- métrica y observabilidad básica

## Vector Engine

- colecciones vectoriales con métrica `cosine` y `dot`
- `upsert`, `delete`, `search`, `get`
- soporte HNSW, IVF y DiskANN según configuración
- persistencia en disco y reconstrucción al arranque
- métricas de búsqueda y compaction
- filtrado por `allowed_ids`

## SQLite Relacional

- ejecución de `SELECT` y `EXEC` por API
- modo WAL
- actor dedicado para serializar acceso
- base para metadata, filtros híbridos y ahora `ns-mem`

## Hub Híbrido (`/v1/db`)

- ingesta de documentos con chunking
- embeddings por API
- escritura coordinada KV + vector + SQLite
- planner `sql_first` o `vector_first`
- hydration del documento original
- `include_plan` e `include_diagnostics`

## DocStore y State

- almacenamiento JSON por key
- índices secundarios simples sobre state
- docstore ligero sobre KV
- endpoints para `put/get/delete/find`

## NS-Mem

Capacidades actuales:

- memoria `episodic`
- memoria `semantic`
- memoria `procedural`
- working memory con TTL
- query `auto`
- timeline por entidad
- `next_step` determinista
- consolidación `episodic -> semantic`
- extracción de facts con `mock`, `openai`, `ollama` o heurísticas locales

Endpoints:

- `POST /v1/memory/{namespace}/ingest_event`
- `POST /v1/memory/{namespace}/upsert_fact`
- `POST /v1/memory/{namespace}/upsert_procedure`
- `POST /v1/memory/{namespace}/query`
- `POST /v1/memory/{namespace}/next_step`
- `GET /v1/memory/{namespace}/timeline/{entity_id}`

## Seguridad

- autenticación por API key
- scoping multitenant en rutas del hub y memory
- límites de tamaño en request body, JSON, ids y colecciones

## Integración con modelos

- embeddings con `none`, `mock`, `openai`, `ollama`
- extracción de facts para `ns-mem` con `none`, `mock`, `openai`, `ollama`
- desacople entre retrieval vectorial y razonamiento procedural

## Testing

- unit tests del core
- tests HTTP de endpoints
- tests de persistencia
- tests de hub híbrido
- tests multitenant
- tests de `ns-mem`

## Estado del roadmap técnico

Listo hoy:

- base convergente single-binary
- retrieval híbrido
- memoria procedural simple
- consolidación semántica inicial

Próximo salto útil:

- aprobación humana de facts `draft`
- endpoint de inspección de facts extraídos
- planner más rico para respuestas compuestas
- trazabilidad expandida de memoria y consolidación
