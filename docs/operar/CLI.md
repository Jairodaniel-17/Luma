# Configuración de la Línea de Comandos y Variables de Entorno

`rust-kiss-vdb` se puede configurar a través de argumentos de línea de comandos y variables de entorno. La siguiente es una lista completa de las opciones disponibles.

## Argumentos de Línea de Comandos

Estos argumentos proporcionan una forma directa de configurar las opciones básicas del servidor al momento de ejecutarlo.

| Argumento                 | Descripción                                                                                                | Por Defecto                                | Variable de Entorno Equivalente |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------- |
| `--port <PORT>`           | Especifica el puerto en el que escuchará el servidor.                                                      | `9917`                                     | `PORT_RUST_KISS_VDB`            |
| `--bind <IP>` / `--host <IP>` | Define la dirección IP a la que se vinculará el servidor.                                                  | `127.0.0.1`                                | `BIND_ADDR`                     |
| `--unsafe-bind`           | Un atajo para `--bind 0.0.0.0`, que expone el servidor a la red. Úsalo con precaución.                      | -                                          | -                               |
| `--data <PATH>` / `--data-dir <PATH>` | La ruta al directorio donde se almacenarán los datos, snapshots y el WAL (Write-Ahead Log).              | No establecido (se ejecuta en modo en memoria) | `DATA_DIR`                      |

## Subcomandos

Además del modo `serve` por defecto, la CLI soporta los siguientes subcomandos:

| Comando                               | Descripción                                                                                                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `serve`                               | (Por defecto) Inicia el servidor de la base de datos vectorial.                                                                          |
| `vacuum --collection <NOMBRE>`        | Ejecuta un proceso de compactación en una colección específica para optimizar el almacenamiento y eliminar datos marcados como borrados.    |
| `diskann ...`                         | Subcomandos para construir, ajustar y verificar el estado de los índices DiskAnn. Consulta `diskann --help` para más detalles.            |

## Variables de Entorno

Para una configuración más detallada y granular, puedes usar las siguientes variables de entorno. Son especialmente útiles para ajustar el rendimiento del motor y los límites de recursos.

### Configuración Principal

| Variable                    | Descripción                                                                     | Por Defecto        |
| --------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| `RUSTKISS_API_KEY` / `API_KEY` | La clave de API requerida para autenticar las solicitudes.                      | `dev`              |
| `DATA_DIR`                  | Directorio para el almacenamiento de datos persistentes.                        | -                  |
| `PORT_RUST_KISS_VDB`        | El puerto del servidor.                                                         | `9917`             |
| `BIND_ADDR`                 | La dirección IP a la que se vincula el servidor.                                | `127.0.0.1`        |
| `CORS_ALLOWED_ORIGINS`      | Orígenes permitidos para CORS, separados por comas.                             | -                  |
| `SQLITE_ENABLED`            | Habilita la API de SQL (`true`/`false`).                                        | `false`            |
| `SQLITE_PATH`               | Ruta al archivo de la base de datos SQLite.                                     | -                  |

### Rendimiento y Motor

| Variable                           | Descripción                                                                          | Por Defecto     |
| ---------------------------------- | ------------------------------------------------------------------------------------ | --------------- |
| `SEARCH_THREADS`                   | Número de hilos para la búsqueda. `0` usa el número de núcleos de CPU disponibles.   | `0`             |
| `PARALLEL_PROBE`                   | Habilita el sondeo en paralelo de segmentos durante la búsqueda.                     | `true`          |
| `SIMD_ENABLED`                     | Habilita optimizaciones SIMD para cálculos de distancia.                             | `true`          |
| `SNAPSHOT_INTERVAL_SECS`           | Intervalo en segundos para crear snapshots de los datos en disco.                    | `30`            |
| `WAL_SEGMENT_MAX_BYTES`            | Tamaño máximo en bytes por archivo de segmento del WAL.                              | `67108864` (64MB) |
| `COMPACTION_TRIGGER_TOMBSTONE_RATIO` | Proporción de registros borrados que dispara una compactación automática.            | `0.2`           |
| `REQUEST_TIMEOUT_SECS`             | Tiempo máximo de espera en segundos para las solicitudes.                            | `30`            |

### Índices Vectoriales (IVF y DiskAnn)

| Variable                    | Descripción                                                                                  | Por Defecto   |
| --------------------------- | -------------------------------------------------------------------------------------------- | ------------- |
| `INDEX_KIND`                | Tipo de índice a utilizar (`IVF_FLAT_Q8`, `DISKANN_Q8`, etc.).                                 | `IVF_FLAT_Q8` |
| `IVF_CLUSTERS`              | Número de clústeres a usar en el índice IVF.                                                 | `4096`        |
| `IVF_NPROBE`                | Número de clústeres a explorar durante una búsqueda IVF.                                     | `16`          |
| `IVF_MIN_TRAIN_VECTORS`     | Número mínimo de vectores necesarios para entrenar el índice IVF.                            | `1024`        |
| `DISKANN_MAX_DEGREE`        | Grado máximo del grafo en un índice DiskAnn.                                                 | `48`          |
| `DISKANN_SEARCH_LIST_SIZE`  | Tamaño de la lista de búsqueda (beam search) para DiskAnn.                                   | `64`          |

### Límites y Recursos

| Variable             | Descripción                                       | Por Defecto     |
| -------------------- | ------------------------------------------------- | --------------- |
| `MAX_BODY_BYTES`     | Tamaño máximo del cuerpo de la solicitud en bytes.  | `1048576` (1MB) |
| `MAX_VECTOR_DIM`     | Dimensión máxima permitida para un vector.        | `4096`          |
| `MAX_K`              | Valor máximo de `k` para una búsqueda top-k.      | `256`           |
| `MAX_VECTOR_BATCH`   | Tamaño máximo de lote para operaciones de vectores. | `256`           |
| `MAX_DOC_FIND`       | Límite máximo para consultas de documentos.       | `100`           |

---

*Para obtener una lista exhaustiva y los valores por defecto más actualizados, consulta el archivo `src/config.rs`.*
