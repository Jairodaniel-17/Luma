# NS-Mem en Luma

Esta guía describe cómo usar la capa `NS-Mem` de Luma para manejar memoria episódica, semántica, procedural y working memory sobre el stack convergente del proyecto.

## Qué resuelve

`NS-Mem` añade una capa de memoria para agentes encima de Luma:

- `episodic`: eventos e interacciones concretas
- `semantic`: hechos estables o preferencias consolidadas
- `procedural`: procedimientos, DAGs y reglas de transición
- `working`: contexto efímero por sesión, guardado en KV con TTL

La implementación actual usa:

- vector store para `episodic` y `semantic`
- SQLite para `memory_records`, `procedures`, nodos, edges y constraints
- KV para working memory
- un consolidator opcional `episodic -> semantic`

## Configuración

Parámetros principales:

- `MEMORY_CONSOLIDATION_ENABLED=true|false`
- `MEMORY_WORKING_TTL_SECS=3600`
- `MEMORY_DEFAULT_LIMIT=10`
- `MEMORY_MAX_EVIDENCE=10`
- `MEMORY_PROCEDURAL_MAX_NODES=128`
- `MEMORY_FACT_PROMOTION_THRESHOLD=0.85`
- `LLM_PROVIDER=none|mock|openai|ollama`
- `LLM_MODEL=...`
- `LLM_URL=...`
- `LLM_API_KEY=...`

Ejemplo mínimo para desarrollo local:

```bash
export SQLITE_ENABLED=true
export MEMORY_CONSOLIDATION_ENABLED=true
export LLM_PROVIDER=mock
cargo run
```

## Endpoints

### 1. Ingestar un evento episódico

- `POST /v1/memory/{namespace}/ingest_event`

```bash
curl -X POST http://localhost:1234/v1/memory/agents/ingest_event \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-1",
    "text": "El usuario pidió activar alertas por correo",
    "metadata": {
      "channel": "chat"
    },
    "session_id": "sess-1"
  }'
```

Qué hace:

- persiste el evento en `memory_records` como `episodic`
- genera embedding y lo indexa en vector store
- guarda working memory si llega `session_id`
- si la consolidación está activa, intenta promover facts semánticos

### 2. Crear o actualizar un fact semántico

- `POST /v1/memory/{namespace}/upsert_fact`

```bash
curl -X POST http://localhost:1234/v1/memory/agents/upsert_fact \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user-1",
    "fact_key": "notification_preference",
    "content": "Prefiere alertas por correo",
    "metadata": {
      "category": "preferences"
    }
  }'
```

### 3. Consultar memoria híbrida

- `POST /v1/memory/{namespace}/query`

Modo `auto`:

```bash
curl -X POST http://localhost:1234/v1/memory/agents/query \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué recuerda del usuario sobre notificaciones?",
    "entity_id": "user-1",
    "include_evidence": true,
    "include_plan": true,
    "include_diagnostics": true
  }'
```

Comportamiento actual:

- preguntas de recuerdo: `recall`
- preguntas de historial: `timeline`
- preguntas de siguiente paso: `next_step`

### 4. Obtener timeline episódico

- `GET /v1/memory/{namespace}/timeline/{entity_id}`

```bash
curl http://localhost:1234/v1/memory/agents/timeline/user-1 \
  -H "Authorization: Bearer dev"
```

### 5. Registrar un procedimiento

- `POST /v1/memory/{namespace}/upsert_procedure`

```bash
curl -X POST http://localhost:1234/v1/memory/ops/upsert_procedure \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "procedure_id": "approve_refund",
    "name": "Approve refund",
    "nodes": [
      { "node_id": "start", "kind": "start", "label": "Start", "payload": {} },
      { "node_id": "validate", "kind": "action", "label": "Validate request", "payload": {} },
      { "node_id": "approve", "kind": "goal", "label": "Approve", "payload": {} }
    ],
    "edges": [
      { "from_node_id": "start", "to_node_id": "validate", "priority": 10, "condition": null },
      {
        "from_node_id": "validate",
        "to_node_id": "approve",
        "priority": 10,
        "condition": { "field": "request.amount", "op": "lte", "value": 500 }
      }
    ],
    "constraints": [
      {
        "constraint_id": "role-check",
        "target_node_id": "approve",
        "condition": { "field": "actor.role", "op": "eq", "value": "manager" },
        "message": "manager role required"
      }
    ]
  }'
```

### 6. Resolver el siguiente paso válido

- `POST /v1/memory/{namespace}/next_step`

```bash
curl -X POST http://localhost:1234/v1/memory/ops/next_step \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "procedure_id": "approve_refund",
    "current_node_id": "validate",
    "context": {
      "request": { "amount": 200 },
      "actor": { "role": "manager" }
    }
  }'
```

## Consolidación episódica a semántica

Si `MEMORY_CONSOLIDATION_ENABLED=true`, cada evento episódico con `entity_id` pasa por el consolidator.

Flujo actual:

1. el evento entra como `episodic`
2. `llm.rs` intenta extraer `FactCandidate`
3. si no hay LLM remoto o falla, se usan heurísticas locales
4. cada fact se guarda como `semantic`
5. si `confidence >= MEMORY_FACT_PROMOTION_THRESHOLD`, queda `active`
6. si no, queda `draft`

Ejemplo:

- evento: `El usuario pidió activar alertas por correo`
- fact derivado: `notification_preference = Prefiere alertas por correo`

## Proveedores LLM

Estado actual:

- `none`: desactiva extracción remota; solo heurísticas locales
- `mock`: útil para tests y demos
- `openai`: espera un endpoint compatible con `chat/completions`
- `ollama`: espera un endpoint compatible con `generate`

La capa LLM solo extrae facts. No decide el flujo procedural ni evalúa restricciones.

## Reglas y constraints

Los constraints de procedimientos se evalúan de forma determinista en Rust.

Operadores soportados:

- `eq`
- `neq`
- `gt`
- `gte`
- `lt`
- `lte`
- `contains`
- `in`

Ejemplo de contexto:

```json
{
  "request": { "amount": 200 },
  "actor": { "role": "manager" }
}
```

Ejemplo de condición:

```json
{
  "field": "actor.role",
  "op": "eq",
  "value": "manager"
}
```

## Estado actual de la feature

Implementado:

- ingest episódico
- facts semánticos
- recall híbrido
- timeline por entidad
- procedimientos con DAG
- `next_step`
- consolidación `episodic -> semantic`
- soporte `mock/openai/ollama` para extracción de facts

Pendiente o básico todavía:

- endpoint explícito para inspeccionar facts antes de persistir
- versionado rico de facts
- feedback loop humano para aprobar `drafts`
- planner más avanzado para mezclar procedural + recall en una sola respuesta
