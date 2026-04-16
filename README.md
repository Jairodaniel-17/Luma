# Luma (rust-kiss-vdb): La Plataforma de Datos Convergente

Luma no es solo una base de datos vectorial. Es un **Motor de Datos Convergente** diseñado para la era de la Inteligencia Artificial Generativa y los Agentes Autónomos.

Mientras que la arquitectura tradicional fragmenta tu stack tecnológico (PostgreSQL para datos, Redis para caché/colas, Pinecone para vectores), Luma unifica estas primitivas en un **único binario escrito en Rust**, eliminando la latencia de red, simplificando el despliegue y garantizando un rendimiento extremo.

## 🚀 ¿Por qué Luma?

La premisa es simple: **La IA necesita más que vectores.**

Un agente de IA moderno necesita:
1.  **Memoria Semántica:** Búsqueda vectorial para recuperar información relevante.
2.  **Memoria Estructurada:** Metadatos relacionales (SQL) para filtrado preciso y datos de negocio.
3.  **Estado Efímero:** Almacenamiento Key-Value (KV) de alta velocidad para sesiones y contexto.
4.  **Sistema Nervioso:** Un bus de eventos (Pub/Sub) para comunicación en tiempo real entre agentes y usuarios.

Luma ofrece todo esto "out-of-the-box" mediante una arquitectura orquestada.

---

## 🏛️ Arquitectura Multi-Motor

Luma no es un monolito, sino un **Orquestador de Alto Rendimiento** que gestiona y sincroniza múltiples motores especializados dentro de un mismo proceso. Al iniciar, el servidor (`src/server.rs`) levanta y conecta estos componentes:

### 1. El Core Engine (`src/engine/`)
Este es el corazón nativo de alto rendimiento, escrito puramente en Rust. Gestiona los datos que requieren latencia crítica y estructuras no relacionales.
*   **Motor Vectorial:** Implementación de **DiskANN (Vamana)** y **IVF**. Maneja índices masivos en disco con optimizaciones SIMD y cuantización (Q8).
*   **Motor de Estado (KV):** Impulsado por **redb**, ofrece almacenamiento ACID para documentos JSON y sesiones, con soporte nativo para TTL (expiración automática).
*   **Bus de Eventos:** Un sistema de Pub/Sub (`tokio::sync::broadcast`) que actúa como el sistema nervioso, permitiendo streaming de datos y reactividad en tiempo real (SSE).
*   **Unified WAL:** Un Write-Ahead Log personalizado garantiza la durabilidad y consistencia de estos componentes.

### 2. El Servicio SQL Relacional (`src/sqlite/`)
Para cuando se necesita la robustez del modelo relacional estándar.
*   Luma integra **SQLite** embebido (vía `rusqlite`), configurado en modo **WAL (Write-Ahead Logging)** para máxima concurrencia.
*   Funciona como un motor paralelo al Core, permitiendo JOINS complejos, transacciones ACID estrictas y filtrado avanzado de metadatos.
*   El servidor expone endpoints que permiten "cruzar" información entre el mundo vectorial y el relacional.

### 3. La Capa de Orquestación (`src/server.rs` & `src/api/`)
El "pegamento" que une los mundos.
*   Expone una **API HTTP Unificada** (`src/api/`) que enruta las peticiones al motor correspondiente.
*   Maneja la autenticación y la seguridad de forma centralizada.
*   Permite que un solo binario sirva como la infraestructura completa para una aplicación de IA.

---

## 🛠️ Tecnologías Clave

Luma está construido sobre el ecosistema de **Rust**, priorizando la seguridad de memoria, la concurrencia y la eficiencia.

| Componente | Tecnología / Crate | Rol en la Arquitectura |
| :--- | :--- | :--- |
| **Orquestación** | `tokio` | Runtime asíncrono para manejar I/O no bloqueante y miles de conexiones concurrentes. |
| **Core KV** | `redb` | Persistencia ACID pura en Rust para el Core Engine, sin dependencias externas. |
| **Relacional** | `rusqlite` (SQLite) | Motor SQL embebido, gestionado como un servicio interno independiente. |
| **Vectores** | Custom `DiskANN` | Algoritmos de grafos en disco desarrollados a medida para búsqueda semántica. |
| **Serialización** | `serde` + `serde_json` | Lingua franca para el intercambio de datos entre motores y API. |

### Mapa del Código Fuente

*   **`src/server.rs`**: El punto de entrada. Inicializa la configuración, levanta el Core Engine y el Servicio SQL, y arranca el servidor HTTP.
*   **`src/engine/`**: Implementación del **Core Engine**. Agrupa los módulos de vectores, estado (KV) y eventos bajo una misma gestión de ciclo de vida.
    *   `luma::engine::inner`: Contiene la lógica de sincronización y el bus de eventos.
*   **`src/sqlite/`**: Contiene `SqliteService`, la abstracción que maneja el pool de conexiones y las consultas al motor SQL embebido.
*   **`src/vector/`**: Lógica matemática pura y estructuras de datos para la indexación vectorial (DiskANN, IVF).
*   **`src/api/`**: Controladores HTTP que exponen las capacidades de todos los motores al usuario final.
*   **`src/memory/`**: NS-Mem — capa de memoria de agentes. Incluye ingesta episódica, facts semánticos, motor procedural DAG, consolidación LLM y retrieval híbrido.

---


---

## 🧭 Paradigma de Uso: Tres Niveles

Luma está diseñado bajo una arquitectura "A la Carta", lo que significa que puedes usarlo como una base de datos simple de bajo nivel, o como un poderoso motor orquestador para flujos RAG completos.

### Nivel 1: Endpoints Primitivos (Modo "A la Carta")
Ideales para máxima velocidad y bajo overhead. Cada motor funciona de forma aislada para que tú gestiones la lógica desde tu backend.
*   **Vectorial:** `/v1/vector/...` (Búsquedas KNN/ANN. Solo arrays de floats).
*   **Documentos:** `/v1/doc/...` (Almacenamiento de JSON, como un MongoDB ligero).
*   **Clave-Valor:** `/v1/state/...` (Para locks, coordinación, configuración).
*   **Relacional:** `/v1/sql/...` (Ejecución cruda de queries SQLite).

### Nivel 3: NS-Mem — Memoria de Agentes

Una capa de memoria completa para agentes autónomos, construida encima del stack convergente de Luma:

| Tipo | Almacenamiento | Descripción |
| :--- | :--- | :--- |
| **episodic** | Vector + SQLite | Eventos e interacciones concretas indexadas para recall semántico |
| **semantic** | Vector + SQLite | Hechos y preferencias estables, promovidos desde episodic vía LLM |
| **procedural** | SQLite (DAG) | Flujos de trabajo con nodos, edges tipados y evaluación de constraints |
| **working** | KV + TTL | Contexto efímero de sesión, expira automáticamente |

**Pipeline de consolidación**: `ingest_event` → extracción de facts (LLM o heurística local) → `semantic` (`active` si confianza ≥ 0.85, else `draft`).

**Endpoints** (`/v1/memory/{namespace}/`):
- `POST ingest_event` — ingesta episódica con embedding + working memory opcional
- `POST upsert_fact` — crea o actualiza un fact semántico
- `POST query` — recall híbrido (episodic + semantic + procedural)
- `GET timeline/{entity_id}` — historial cronológico por entidad
- `POST upsert_procedure` — registra/actualiza un DAG procedural
- `POST next_step` — resuelve el siguiente nodo válido según contexto y constraints

Proveedores LLM soportados: `none`, `mock`, `openai`, `ollama`. Ver `docs/NS_MEM.md`.

---

### Nivel 2: LumaDatabase Hub (Modo Híbrido RAG)
El orquestador interno (`LumaDatabase`) fusiona el poder de todos los motores para hacer el trabajo pesado por ti. Segmenta documentos grandes (Chunking), se conecta a modelos de Embeddings (Ollama/OpenAI) de forma automática, y hace **Pre-filtrado SQL estricto** a la velocidad de la luz antes de realizar la búsqueda vectorial, evitando los problemas clásicos del "Post-filtrado" vectorial.

#### 1. Ingesta Automática (`POST /v1/db/{namespace}/ingest`)
Luma procesa el texto, se conecta al modelo de IA configurado, crea la colección si no existe, divide el texto si es muy largo, guarda los vectores, y almacena tus metadatos en SQLite de forma transaccional (con *Rollback* automático si falla I/O).

```json
{
  "id": "contrato_juan_perez", 
  "text": "El arrendatario se compromete a pagar $500 mensuales...",
  "metadata": {
    "cliente": "Juan Perez",
    "year": 2024,
    "tipo": "alquiler",
    "activo": true
  }
}
```

#### 2. Búsqueda Híbrida con Pre-filtrado Relacional (`POST /v1/db/{namespace}/search`)
Busca semánticamente usando un modelo de embeddings y filtra rígidamente (Pre-filtro) usando SQLite para garantizar un 100% de precisión. Luma "colapsa" internamente los fragmentos (chunks) y te devuelve el documento padre hidratado con sus mejores *snippets*.

```json
{
  "query": "cláusula sobre el precio del alquiler",
  "limit": 5,
  "sql_filter": "json_extract(metadata, '$.tipo') = 'alquiler' AND json_extract(metadata, '$.year') = 2024"
}
```

*Incluso si existe un contrato de compra-venta de 2023 muy parecido semánticamente a la pregunta, Luma lo descartará instantáneamente en la fase SQLite (usando el canal de concurrencia MPSC en memoria) antes de desperdiciar CPU en el motor vectorial.*

### 🔌 Configuración de Embeddings (BYOM - Bring Your Own Model)
Para no engordar el binario con librerías pesadas de C++, Luma usa un cliente ligero HTTP integrado. Simplemente configúralo en tu código o entorno:

```rust
// Ejemplo para Ollama local
luma::engine::embeddings::EmbeddingProvider::Ollama {
    api_url: "http://localhost:11434".to_string(),
    model: "granite-embedding:30m".to_string(),
}
```

## 💡 Flujos de Trabajo Híbridos

Gracias a esta arquitectura orquestada, puedes construir flujos imposibles con bases de datos aisladas:

### RAG con Contexto de Negocio
1.  **Vector (Core):** Encuentra los 10 documentos más parecidos semánticamente a la pregunta del usuario.
2.  **SQL (Relacional):** Filtra esos documentos verificando en la tabla `usuarios_y_permisos` si el usuario actual tiene acceso nivel 'admin'.
3.  **Eventos (Core):** Publica un evento `search_audit` que otros microservicios pueden escuchar en tiempo real.

Todo esto ocurre dentro de una sola llamada al servidor Luma, con latencia de red interna cero.

## 🚀 Novedades en v1.4.0 (NS-Mem — Agent Memory Layer)

*   **NS-Mem**: Capa completa de memoria para agentes (`src/memory/`). Tipos: `episodic`, `semantic`, `procedural`, `working`. Pipeline de consolidación automática episodic → semantic vía LLM.
*   **Motor procedural DAG**: Flujos de trabajo persistidos en SQLite con evaluación determinista de constraints en Rust (8 operadores).
*   **LLM providers**: Extracción de facts vía `openai`, `ollama`, `mock` o heurísticas locales (`none`).
*   **KV sharding**: Store fragmentado en 16 buckets independientes (menor contención).
*   **WAL group commit**: Reducción de fsyncs en ingesta masiva.
*   **Métricas con histogramas de latencia**: `/v1/metrics` con percentiles p50/p95/p99.

## 🚀 Novedades en v1.3.2 (Mmap & Zero-Copy Architecture)

La versión 1.3.2 marca un hito en la eficiencia de recursos, permitiendo a Luma escalar a millones de vectores con un consumo de RAM mínimo:
*   **Motor de Almacenamiento Zero-Copy:** Implementación de `VectorMmap` usando memoria mapeada (`memmap2`). Los vectores ya no residen obligatoriamente en el heap de Rust, sino que se acceden directamente desde el disco a través del Page Cache del SO.
*   **Latencia de Ingesta Ultra-Baja:** Reducción de la latencia de escritura a ~2.4 microsegundos por vector (dim 1536), eliminando pases de serialización intermedios.
*   **Arranque Instantáneo:** El mapeo de archivos permite que colecciones masivas estén listas para buscar en milisegundos, delegando la carga de datos al Kernel bajo demanda.
*   **Mapeo ID-to-Offset O(1):** Integración de offsets binarios en `VectorItem` para saltar directamente a la posición física del vector en disco durante la fase de refinamiento.
*   **Migración Automática Silenciosa:** Los datos existentes en el formato WAL heredado se migran automáticamente al nuevo motor binario al primer arranque.

## 🚀 Novedades en v1.3.1 (Mejoras de Arquitectura Nivel 2)

La versión 1.3.1 introduce cambios radicales para eliminar cuellos de botella y maximizar el rendimiento en escenarios de alta concurrencia:
*   **Push-Down Filtering Verdadero:** Integración directa de filtrado relacional con DiskANN/IVF. Los nodos descartados por SQLite nunca son evaluados, reduciendo la complejidad a `O(M)` sobre el subgrafo válido.
*   **Ingesta Concurrente Controlada:** El orquestador usa `tokio::sync::Semaphore` para limitar inteligentemente peticiones HTTP masivas a modelos de embeddings remotos, manteniendo el throughput máximo sin ahogar la red.
*   **Locking Granular:** Se eliminó el `commit_lock` global en el motor principal por un `DashMap`, permitiendo ingestas multi-hilo completamente paralelas y sin bloqueos entre diferentes colecciones.
*   **Hydration Nativo:** Sustituimos `serde_json::Value` por `structs` nativos en el hot path, eliminando la presión en el GC y mejorando drásticamente los tiempos de latencia y uso de CPU al buscar.
*   **Auto-Schema en Background:** La auto-creación de índices (`CREATE INDEX`) de SQLite ahora se gestiona de forma asíncrona mediante una cola MPSC dedicada y de un solo worker, resolviendo definitivamente el overhead de locks (tipo `SQLITE_BUSY`).
*   **Modelo de Consistencia Explícito:** Patrón Eventual Consistency por compensación de *rollback* (Garantiza revertir estados relacionales o en `redb` si falla el vector store).

---

## 🏁 Conclusión

Luma (rust-kiss-vdb) redefine el backend para IA mediante la **convergencia**. No es un simple wrapper; es un sistema de ingeniería cuidadosa que orquesta los mejores motores de su clase (DiskANN para vectores, SQLite para relaciones, redb para KV) en una sola plataforma cohesionada.

> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**
