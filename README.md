# Luma (rust-kiss-vdb): La Plataforma de Datos Convergente

Luma es un **Motor de Datos Convergente** diseñado para la era de la Inteligencia Artificial Generativa y los Agentes Autónomos. Unifica bases de datos vectoriales, búsqueda full-text, almacenamiento Key-Value y colas de eventos en un solo binario de alto rendimiento escrito en Rust.

## 🚀 Características Principales

*   **Motor Vectorial:** Implementación de DiskANN (Vamana) e IVF para búsqueda semántica escalable en disco. Soporta cuantización Q8 y optimizaciones SIMD.
*   **Búsqueda Full-Text:** Motor de búsqueda integrado para recuperación léxica.
*   **Almacenamiento Key-Value (KV):** Persistencia ACID basada en `redb` con soporte para TTL (Time-To-Live).
*   **Bus de Eventos (SSE):** Sistema Pub/Sub en tiempo real via Server-Sent Events (SSE) para sincronización de estado.
*   **SQL Embebido:** Integración con SQLite (vía `rusqlite`) para consultas relacionales y metadatos estructurados.
*   **API Unificada:** API REST simple para acceder a todas las funcionalidades.

## 🛠️ Requisitos

*   Rust (stable)
*   `build-essential` (o equivalente en tu OS) para compilar dependencias C (SQLite).

## 🏃 Cómo Ejecutar

### Iniciar el Servidor

```bash
# Ejecutar en modo desarrollo
cargo run

# Ejecutar con release (optimizado)
cargo run --release
```

El servidor iniciará por defecto en `http://127.0.0.1:8080` (o el puerto configurado).

### Variables de Entorno

Puedes configurar Luma mediante variables de entorno o archivo `.env`:

*   `PORT_LUMA_VDB`: Puerto de escucha (default: 8080).
*   `API_KEY`: Clave de autenticación (default: "dev").
*   `DATA_DIR`: Directorio de datos (default: "./data").
*   `SQLITE_ENABLED`: Habilitar motor SQL (default: true).

## 🧪 Cómo Correr Tests

Para ejecutar la suite de pruebas completa:

```bash
cargo test
```

Para correr tests específicos (ej. integración SSE):

```bash
cargo test --test auth_ttl_sse_gap
```

## 🗺️ Arquitectura

Luma orquesta múltiples motores:
1.  **Core Engine:** Maneja vectores, KV y eventos.
2.  **Search Engine:** Maneja índices invertidos para texto.
3.  **SQLite Service:** Maneja datos relacionales.

Todos los datos persisten en el directorio configurado en `DATA_DIR`.

## ⚠️ Limitaciones y Roadmap

*   **Estado:** Alpha/Beta. APIs pueden cambiar.
*   **Cluster:** Actualmente opera como nodo único (single-node).
*   **Autenticación:** Básica por API Key estática o gestión de llaves simple.

**Roadmap:**
*   Replicación y Sharding.
*   Soporte avanzado de filtros híbridos (Vector + SQL + Texto).
*   SDKs para Python y JS.

---
> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**
