# Guía de la API de la Base de Datos Vectorial

Esta guía proporciona una descripción detallada de la API de `rust-kiss-vdb`, con ejemplos de `curl` para las operaciones más comunes.

## Autenticación

Todas las solicitudes a la API deben incluir una clave de API a través del encabezado `Authorization` como un token "Bearer".

```bash
-H "Authorization: Bearer TU_API_KEY"
```

Si no se proporciona una clave, el servidor usará `dev` por defecto.

## Endpoints de la API Vectorial

La API principal para la gestión de vectores se encuentra bajo el prefijo `/v1/vector`.

### 1. Crear una Colección

Antes de poder añadir vectores, necesitas crear una "colección" que los contenga. Cada colección tiene una dimensión de vector y una métrica de distancia fijas.

-   **Endpoint:** `POST /v1/vector/{nombre_coleccion}`
-   **Métricas Soportadas:** `cosine` (coseno), `dot` (producto punto).

**Ejemplo:** Crear una colección llamada `mis_embeddings` para vectores de 384 dimensiones con métrica de coseno.

```bash
curl -X POST http://localhost:9917/v1/vector/mis_embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d 
  {
    "dim": 384,
    "metric": "cosine"
  }
```

**Respuesta Exitosa (200 OK):**

```json
{
  "collection": "mis_embeddings",
  "dim": 384,
  "metric": "cosine"
}
```

### 2. Listar Colecciones

Puedes obtener una lista de todas las colecciones existentes y sus propiedades.

-   **Endpoint:** `GET /v1/vector`

**Ejemplo:**

```bash
curl http://localhost:9917/v1/vector \
  -H "Authorization: Bearer dev"
```

**Respuesta Exitosa (200 OK):**

```json
{
  "collections": [
    {
      "collection": "mis_embeddings",
      "dim": 384,
      "metric": "cosine",
      "live_count": 0,
      "total_records": 0,
      "upsert_count": 0,
      "file_len": 1024,
      "applied_offset": 1,
      "created_at_ms": 1678886400000,
      "updated_at_ms": 1678886400000
    }
  ]
}
```

### 3. Añadir o Actualizar Vectores (Upsert)

La operación "upsert" añade un vector si su `id` no existe, o lo actualiza si ya existe. Los vectores deben tener un `id` único (string), el propio `vector` (array de floats), y opcionalmente un campo `meta` para almacenar metadatos en formato JSON.

-   **Endpoint:** `POST /v1/vector/{nombre_coleccion}/upsert`

**Ejemplo:** Añadir un vector a la colección `mis_embeddings`.

```bash
curl -X POST http://localhost:9917/v1/vector/mis_embeddings/upsert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d 
  {
    "id": "doc_123",
    "vector": [0.1, 0.2, ..., 0.9],
    "meta": {
      "autor": "Jairo",
      "año": 2024,
      "publicado": true
    }
  }
```

También puedes usar el endpoint `/upsert_batch` para añadir múltiples vectores en una sola solicitud, lo cual es mucho más eficiente.

### 4. Búsqueda de Vectores

La búsqueda de similitud es la operación central de una base de datos vectorial. Proporcionas un vector de consulta y la API devuelve los `k` vectores más similares de la colección.

-   **Endpoint:** `POST /v1/vector/{nombre_coleccion}/search`

**Parámetros del Cuerpo:**

-   `vector`: El vector de consulta.
-   `k`: El número de vecinos más cercanos a devolver.
-   `filters` (opcional): Un objeto JSON para filtrar vectores basado en sus metadatos antes de la búsqueda.
-   `include_meta` (opcional): Si es `true`, la respuesta incluirá los metadatos de los vectores encontrados.

**Ejemplo:** Buscar los 5 vectores más similares en `mis_embeddings`.

```bash
curl -X POST http://localhost:9917/v1/vector/mis_embeddings/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d 
  {
    "vector": [0.11, 0.22, ..., 0.99],
    "k": 5,
    "include_meta": true
  }
```

**Respuesta Exitosa (200 OK):**

```json
{
  "hits": [
    {
      "id": "doc_123",
      "score": 0.987,
      "meta": {
        "autor": "Jairo",
        "año": 2024,
        "publicado": true
      }
    },
    {
      "id": "doc_456",
      "score": 0.954,
      "meta": { ... }
    }
  ]
}
```

#### Filtrado en la Búsqueda

Puedes restringir la búsqueda a solo los vectores que cumplan ciertas condiciones en sus metadatos. El filtro es un objeto JSON donde las claves coinciden con las claves del campo `meta`.

**Ejemplo:** Buscar los 5 vectores más similares que además tengan `publicado: true` y `autor: "Jairo"` en sus metadatos.

```bash
curl -X POST http://localhost:9917/v1/vector/mis_embeddings/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d 
  {
    "vector": [0.11, 0.22, ..., 0.99],
    "k": 5,
    "filters": {
      "publicado": true,
      "autor": "Jairo"
    },
    "include_meta": true
  }
```
> **Nota:** El motor de búsqueda actual solo soporta filtros de coincidencia exacta (clave-valor). Operadores más complejos como rangos (`$gt`, `$lt`) no están implementados en la capa de la API genérica.

### 5. Obtener un Vector por ID

-   **Endpoint:** `GET /v1/vector/{nombre_coleccion}/get?id={id_vector}`

**Ejemplo:**

```bash
curl "http://localhost:9917/v1/vector/mis_embeddings/get?id=doc_123" \
  -H "Authorization: Bearer dev"
```

### 6. Eliminar un Vector por ID

-   **Endpoint:** `POST /v1/vector/{nombre_coleccion}/delete`

**Ejemplo:**

```bash
curl -X POST http://localhost:9917/v1/vector/mis_embeddings/delete \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d '{
    "id": "doc_123"
  }'
```

## Suscripción a Eventos en Tiempo Real (SSE)

`rust-kiss-vdb` permite suscribirse a cambios en la base de datos en tiempo real mediante Server-Sent Events (SSE). Esto es útil para mantener cachés sincronizadas o reaccionar a inserciones de vectores.

### Endpoint Principal: `/v1/stream`

Este es el endpoint recomendado para consumir el flujo de eventos.

-   **Método:** `GET`
-   **Parámetros de Consulta:**
    -   `since` (opcional, u64): ID del evento (offset) desde el cual empezar a recibir. Por defecto es 0 (inicio).
    -   `types` (opcional, string): Lista separada por comas de los tipos de eventos a filtrar (ej. `vector_upserted,state_changed`).
    -   `key_prefix` (opcional, string): Filtra eventos cuya clave comience con este prefijo.
    -   `collection` (opcional, string): Filtra eventos que pertenezcan a una colección específica.

**Ejemplo:** Suscribirse a inserciones en la colección `mis_embeddings`.

```bash
curl "http://localhost:9917/v1/stream?collection=mis_embeddings&types=vector_upserted" \
  -H "Authorization: Bearer dev"
```

### Endpoint Deprecado: `/v1/events`

El endpoint `/v1/events` se mantiene por compatibilidad con versiones anteriores pero **está deprecado**. Funciona como un alias de `/v1/stream` con la siguiente diferencia en los parámetros:

-   El parámetro `prefix` en `/v1/events` se mapea internamente a `key_prefix`.
-   No soporta el filtrado explícito por `collection` (aunque puedes usar filtros de clave si tus claves tienen prefijos de colección).

**Se recomienda encarecidamente migrar a `/v1/stream` para nuevas implementaciones.**

---
*Para una descripción completa de todos los endpoints, incluidos los de gestión de estado (`/state`), documentos (`/doc`) y SQL (`/sql`), consulta la especificación OpenAPI en `docs/openapi.yaml`.*