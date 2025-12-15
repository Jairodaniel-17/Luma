# CHANGELOG.md

## v0.2.0 (2025-12-14)

- **P0**
  - Bind seguro (`127.0.0.1` default, flags `--bind`/`--unsafe-bind` para exponer).
  - Errores consistentes (`ErrorBody { error, message }` en todas las rutas).
  - Batch APIs: `/v1/state/batch_put`, `/v1/vector/*_batch`.
  - SDK Python formal (`rustkissvdb.Client`) + ejemplos actualizados.
- **P1**
  - Segmentación de vector store + índice por metadata (keyword exact-match).
  - Comando `vacuum` para compactar colecciones vectoriales (WAL limpio).
- **P2**
  - DocStore sobre KV (`/v1/doc/*` + find por metadata).
  - Módulo SQLite opcional (`/v1/sql/query`, `/v1/sql/exec`).
- **Docs**
  - Nuevos documentos: `SECURITY.md`, `VECTOR_STORAGE.md`, `ROADMAP.md`, `SDK_PYTHON.md`, `DATA_MODELS.md`, `CHANGELOG.md`.
  - OpenAPI / API.md actualizados con las rutas nuevas.

> Nota: la numeración sigue la guía del agente (P0/P1/P2). Cada release debe incluir código + tests + docs sincronizados.
