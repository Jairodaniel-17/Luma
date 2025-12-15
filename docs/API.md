# API (v1)

> Si defines `RUSTKISS_API_KEY` (o `API_KEY`) debes enviar `Authorization: Bearer <token>`. Ejemplos omitidos para mantenerlos legibles.

Formato de error uniforme:
```json
{ "error": "not_found", "message": "document not found" }
```

## Health / Metrics

- `GET /v1/health`
- `GET /v1/metrics`

## State

### PUT
`PUT /v1/state/{key}`

Body:
```json
{ "value": { "progress": 42 }, "ttl_ms": 60000, "if_revision": 3 }
```

Ejemplo:
```bash
curl -X PUT "http://localhost:9917/v1/state/job:123" ^
  -H "Content-Type: application/json" ^
  -d "{\"value\":{\"progress\":42},\"ttl_ms\":60000}"
```

### GET
`GET /v1/state/{key}`

```bash
curl "http://localhost:9917/v1/state/job:123"
```

### DELETE
`DELETE /v1/state/{key}`

### LIST
`GET /v1/state?prefix=job:&limit=100`

### BATCH PUT
`POST /v1/state/batch_put`

```bash
curl -X POST "http://localhost:9917/v1/state/batch_put" ^
  -H "Content-Type: application/json" ^
  -d "{\"operations\":[{\"key\":\"bulk:1\",\"value\":{\"x\":1}},{\"key\":\"bulk:2\",\"value\":{\"x\":2}}]}"
```

Respuesta:
```json
{
  "results": [
    { "status": "ok", "key": "bulk:1", "revision": 1 },
    { "status": "error", "key": "bulk:2", "error": { "error": "invalid_argument", "message": "value too large" } }
  ]
}
```

## Events (SSE)

`GET /v1/stream?since=<offset>&types=...&key_prefix=...&collection=...`

```bash
curl -N "http://localhost:9917/v1/stream?since=0&types=state_updated&key_prefix=job:" ^
```

Reconexión:
- `since` y `Last-Event-ID` usan **offset/event_id (u64)**.

Backpressure:
- Si el servidor detecta que el cliente se quedó atrás, emite `event: gap` con `{from_offset,to_offset,dropped}`.

## Vector

### Crear colección
`POST /v1/vector/{collection}`

```bash
curl -X POST "http://localhost:9917/v1/vector/docs" ^
  -H "Content-Type: application/json" ^
  -d "{\"dim\":3,\"metric\":\"cosine\"}"
```

### Add / Upsert
`POST /v1/vector/{collection}/add`
`POST /v1/vector/{collection}/upsert`

```bash
curl -X POST "http://localhost:9917/v1/vector/docs/upsert" ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"a\",\"vector\":[1,0,0],\"meta\":{\"tag\":\"x\"}}"
```

### Search
`POST /v1/vector/{collection}/search`

```bash
curl -X POST "http://localhost:9917/v1/vector/docs/search" ^
  -H "Content-Type: application/json" ^
  -d "{\"vector\":[0.9,0.1,0],\"k\":3,\"filters\":{\"tag\":\"x\"},\"include_meta\":true}"
```

### Batch

- `POST /v1/vector/{collection}/upsert_batch`
- `POST /v1/vector/{collection}/delete_batch`

```bash
curl -X POST "http://localhost:9917/v1/vector/docs/upsert_batch" ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"id\":\"v1\",\"vector\":[0.1,0.2],\"meta\":{\"tag\":\"a\"}}]}"
```

Respuesta: `{"results":[{"status":"upserted","id":"v1"}]}`

## DocStore

- `PUT /v1/doc/{collection}/{id}`
- `GET /v1/doc/{collection}/{id}`
- `DELETE /v1/doc/{collection}/{id}`
- `POST /v1/doc/{collection}/find`

```bash
curl -X PUT "http://localhost:9917/v1/doc/tickets/tk_1" ^
  -H "Content-Type: application/json" ^
  -d "{\"title\":\"Bug 1\",\"severity\":\"high\"}"

curl -X POST "http://localhost:9917/v1/doc/tickets/find" ^
  -H "Content-Type: application/json" ^
  -d "{\"filter\":{\"severity\":\"high\"},\"limit\":20}"
```

Respuesta:
```json
{
  "documents": [
    { "id": "tk_1", "revision": 2, "doc": { "title": "Bug 1", "severity": "high" } }
  ]
}
```

## SQL (opcional)

- `POST /v1/sql/query`
- `POST /v1/sql/exec`

```bash
curl -X POST "http://localhost:9917/v1/sql/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"sql\":\"SELECT name FROM sqlite_master WHERE type='table'\",\"params\":[]}"
```

`/query` solo acepta `SELECT`, devuelve `{"rows":[{...}]}`.  
`/exec` devuelve `{"rows_affected": N}` y sirve para DDL/DML.
