# DATA_MODELS.md

## 1. State Store (KV)

- Claves arbitrarias (`max_key_len`, TTL opcional, CAS via `if_revision`).
- Se usa como “capa base” para features nuevas:
  - `doc:{collection}:{id}` → documento JSON.
  - `docidx:{collection}:{field}:{value}` → índice exacto (array de IDs).
  - `docs:{collection}:manifest` → metadata de ingesta RAG.

## 2. DocStore (sobre KV)

| Ruta                         | Acción                                   |
|------------------------------|------------------------------------------|
| `PUT /v1/doc/{collection}/{id}` | upsert JSON                              |
| `GET /v1/doc/{collection}/{id}` | obtener documento                        |
| `DELETE /v1/doc/{collection}/{id}` | borrar + limpiar índices                |
| `POST /v1/doc/{collection}/find`  | búsqueda simple `{field: "value"}`       |

Notas:

- Sólo indexamos strings top-level (exact-match). Otros tipos se filtran en memoria.
- Respuesta incluye `id`, `doc`, `revision`.
- La paginación es best-effort (`limit` con fallback a `MAX_DOC_FIND`).

## 3. Vector Store

- Cada colección = dir `data_dir/vectors/<collection>`.
- Manifest + `vectors.bin` WAL + segmentos (ver `VECTOR_STORAGE.md`).
- Metadata filters reutilizan el mismo mecanismo que DocStore (hash exacto).

## 4. SQLite embebido

- Activación vía `SQLITE_ENABLED=1` (+ `DATA_DIR` o `SQLITE_DB_PATH`).
- DB única por instancia (`rustkiss.db`).
- Endpoints:
  - `POST /v1/sql/query` → sólo `SELECT` (rows JSON).
  - `POST /v1/sql/exec` → `INSERT/UPDATE/DDL` (rows_affected).
- Configuración inicial: `PRAGMA journal_mode=WAL`, `busy_timeout=5s`.
- Útil para quick prototyping / dashboards internos sin montar otro servicio.

## 5. Relaciones entre modelos

- State y DocStore comparten el mismo WAL, por lo que TTL/compaction afecta a ambos.
- DocStore y VectorStore pueden sincronizarse mediante eventos (`state_updated -> vector_upsert`), pero siguen siendo módulos independientes.
- SQLite NO comparte WAL con RustKissVDB; es un engine aparte con su propio locking.
