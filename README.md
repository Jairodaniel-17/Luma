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
*   **`src/api/`**: Controladores HTTP que exponen las capacidades de ambos motores al usuario final.

---

## 💡 Flujos de Trabajo Híbridos

Gracias a esta arquitectura orquestada, puedes construir flujos imposibles con bases de datos aisladas:

### RAG con Contexto de Negocio
1.  **Vector (Core):** Encuentra los 10 documentos más parecidos semánticamente a la pregunta del usuario.
2.  **SQL (Relacional):** Filtra esos documentos verificando en la tabla `usuarios_y_permisos` si el usuario actual tiene acceso nivel 'admin'.
3.  **Eventos (Core):** Publica un evento `search_audit` que otros microservicios pueden escuchar en tiempo real.

Todo esto ocurre dentro de una sola llamada al servidor Luma, con latencia de red interna cero.

---

## 🏁 Conclusión

Luma (rust-kiss-vdb) redefine el backend para IA mediante la **convergencia**. No es un simple wrapper; es un sistema de ingeniería cuidadosa que orquesta los mejores motores de su clase (DiskANN para vectores, SQLite para relaciones, redb para KV) en una sola plataforma cohesionada.

> **Keep It Simple, Stupid (KISS). Keep It Fast, Rust.**
