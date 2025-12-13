# API (v1)

> No hay autenticación en esta versión; todos los ejemplos omiten `Authorization`.

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
