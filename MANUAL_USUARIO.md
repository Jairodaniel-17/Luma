# Manual de usuario de Luma

Para la versión **v4.26.0**. Todo lo que hay aquí está verificado contra el
código de esa versión, no contra la memoria de cómo funcionaba antes.

Este manual es el que se lee **antes** de poner Luma en producción: qué es, qué
puertas abre, y cómo se configura para que no te muerda. La referencia exhaustiva
de cada endpoint y cada clave vive en `docs/`, y este documento enlaza a ella en
vez de repetirla.

---

## 1. Qué es Luma

Un **binario único de 31 MB**, sin runtime, sin libc, sin servicios auxiliares.
Dentro trae cuatro cosas que normalmente son cuatro productos:

| | Qué es | Cómo se habla con ella |
|---|---|---|
| **Motor de datos** | Vectorial + clave-valor + SQL + documentos + colas + blobs + eventos | HTTP en `/v1/...` |
| **Compatible con Redis** | 117 comandos RESP2 sobre el mismo motor | Puerto RESP, `redis://` |
| **Compatible con S3** | Los mismos blobs, con SigV4, presignadas y multipart | Puerto S3 propio |
| **Conector de Postgres** | Replicación lógica → colección buscable | `luma connect postgres` |

La idea de producto en una frase: **Luma no reemplaza tu Postgres, se conecta a
él**; y no te pide cambiar de cliente para adoptarlo, porque habla los protocolos
que ya usas.

### Los tres niveles del API HTTP

- **Nivel 1 — primitivas.** `/v1/vector`, `/v1/state`, `/v1/doc`, `/v1/queue`,
  `/v1/blob`, `/v1/events`. Cada subsistema por separado, sin magia.
- **Nivel 2 — el hub.** `/v1/db/{ns}/ingest` y `/v1/db/{ns}/search`. Le das un
  documento y él trocea, embebe, indexa y guarda; le das una consulta y él
  combina filtro SQL + búsqueda vectorial + hidratación.
- **Nivel 3 — memoria de agentes (NS-Mem).** `/v1/memory/{ns}/...`: episódica,
  semántica, procedimental y de trabajo, con grafo de aristas tipadas,
  versionado de creencias y centralidad.

### Qué está estable y qué no

| Superficie | Estado |
|---|---|
| Motor HTTP (`/v1/...`), vectorial, KV, SQL, colas, blobs | **Estable** |
| NS-Mem (`/v1/memory`) | **Estable** |
| Multi-tenancy, RBAC, auditoría, panel admin | **Estable** |
| **RESP / compatibilidad Redis** | **Experimental por calendario**, no por funcionalidad — §1.1 |
| **API S3** | **Experimental por cobertura de pruebas** — §1.1 |
| **Conector Postgres (CDC)** | **Experimental por kilometraje operativo** — §1.1 |

### 1.1 Por qué esas tres son experimentales

No significa lo mismo en las tres, y la diferencia es justo la que decide si
puedes usarla hoy.

#### RESP — le falta calendario, no código

El set de comandos está **completo** (57/57 de las fases 2 y 3 del SPEC) y
verificado byte a byte contra un Redis 7 real: **327 comandos idénticos, 0
divergencias**, más Celery, kombu, arq y redis-py de verdad —incluido un worker
de Celery que consume, ejecuta y devuelve el resultado.

Lo único que falta es el criterio que el propio plan se puso: **el job nightly
verde 7 noches seguidas**. Va por **1 de 7**; la primera corrida programada fue
esta madrugada, las anteriores se lanzaron a mano. No es una duda sobre si
funciona, es que nadie ha visto todavía siete noches seguidas sin sorpresas.

Lo que sí conviene saber antes de adoptarlo:

- **No hay Lua de verdad.** `EVAL`/`EVALSHA` reconocen los tres scripts del
  `Lock` de redis-py —los que usa kombu— y los ejecutan de forma nativa, sin
  intérprete. **Cualquier otro script se rechaza** con un error que lo dice. Si
  tu código usa Lua propio, no va.
- **Fuera de alcance por decisión**: cluster (`MOVED`/`ASK`), `REPLICAOF`,
  Streams (`XADD`), keyspace notifications.
- Las estructuras viven en RAM: no es para colas de 10 M de mensajes residentes.

#### S3 — le falta superficie de pruebas

Verificada contra **boto3 real** (14 comprobaciones: presignadas que caducan,
multipart subido fuera de orden, los ETag) y contra los vectores SigV4 publicados
por AWS. Lo que **no** se ha probado, tal como lo dice
[`docs/integrar/S3.md`](docs/integrar/S3.md):

- **La suite mint de MinIO.** La razón es defendible: prueba versionado,
  lifecycle y object lock, que aquí se rechazan **a propósito**. Correrla daría
  una lista larga de fallos esperados y un semáforo que nadie leería.
- **Cargas grandes.** Las pruebas de multipart usan partes de cien bytes. El
  ensamblado es correcto por construcción, pero nadie ha subido un gigabyte.
- **Concurrencia.** Dos clientes escribiendo la misma clave a la vez. Cada
  escritura es atómica (temporal + rename), así que verás uno de los dos cuerpos
  y nunca una mezcla — pero cuál de los dos no está definido.

Lo no soportado se rechaza con **501 NotImplemented**, no se ignora: versionado,
lifecycle, ACLs, replicación, object lock, cifrado por cabecera, tagging, CORS y
hosting web. La diferencia importa: un cliente que pone una ACL y recibe **200**
*cree que el objeto es privado* cuando no lo es.

#### Postgres CDC — le falta kilometraje operativo

7 pruebas contra un Postgres 16 real, con el protocolo `pgoutput` escrito a mano
porque la crate que hace esto no está publicada. Lo que no se ha ejercitado:

- **TLS contra un servidor real.** El camino existe (`sslmode=require`, rustls
  con las raíces de webpki), pero el Postgres del contenedor no sirve TLS: solo
  está probado que rechaza `prefer`.
- **Volumen.** Las pruebas mueven decenas de filas, no millones.
- **Reconexión a media transacción.** Reconecta por pasada, lo que cubre un corte
  de red, un failover y un reinicio con el mismo camino de código — pero nadie ha
  cortado el cable en mitad de una transacción grande.
- **Varios conectores contra la misma base.** Cada uno querría su slot; sin
  probar.
- **Solo el protocolo lógico 1.** Las versiones 2–4 hacen streaming de
  transacciones grandes antes del commit.

Y tres cosas que **no hace por decisión**, no por pendiente: no escribe de vuelta
(Postgres sigue siendo la fuente de verdad), no aplica `TRUNCATE` —lo registra y
marca el checkpoint `stale` en vez de vaciar una colección derivada— y no hace
DDL.

#### En una frase

«Experimental» aquí significa **medido y funcionando, sin el kilometraje que
justifique llamarlo GA**. RESP espera un contador de días; S3 y CDC esperan que
alguien los use en serio y cuente qué pasó.

---

## 2. Instalar y arrancar

### Binario

Descarga el de tu plataforma de
[Releases](https://github.com/Jairodaniel-17/Luma/releases) (Linux musl y gnu,
Windows, macOS ARM) y ejecuta:

```bash
./luma serve
```

Sin argumentos también sirve: `luma` sin subcomando arranca el servidor.

### Docker

La imagen es `FROM scratch` con el binario estático dentro, así que **pesa lo que
pesa el binario: 31 MB**. No hay shell dentro; no esperes poder hacer `docker
exec ... sh`.

```bash
docker run -p 1234:1234 -v luma-data:/data \
  -e DATA_DIR=/data \
  -e LUMA_API_KEY="$(openssl rand -hex 24)" \
  -e LUMA_MASTER_KEY="$(openssl rand -hex 32)" \
  ghcr.io/jairodaniel-17/luma:v4.26.0
```

**El volumen no es opcional.** Sin `-v`, el `data_dir` vive en la capa efímera
del contenedor y se pierde entero al recrearlo.

### Compilar

```bash
cargo build --release        # target/release/luma
```

MSRV 1.88.

---

## 3. La configuración mínima correcta

Si solo lees una sección, que sea esta. Son cinco decisiones, y cuatro de ellas
Luma no puede tomar por ti.

### 3.1 Los dos secretos, o no arranca

Luma **se niega a arrancar** con secretos inseguros. No es un aviso, es un exit
code:

```
Error: refusing to start with 2 insecure secret setting(s)
```

| Variable | Qué pasa si falta |
|---|---|
| `LUMA_API_KEY` | Se rechaza si es `dev`, si está vacía o si tiene **menos de 16 caracteres** |
| `LUMA_MASTER_KEY` | Se rechaza si no está puesta: el cifrado en reposo caería a una clave de desarrollo conocida |

```bash
export LUMA_API_KEY="$(openssl rand -hex 24)"
export LUMA_MASTER_KEY="$(openssl rand -hex 32)"
```

> `LUMA_ALLOW_INSECURE=1` degrada ambos rechazos a avisos. **Es para tu portátil.**
> Si aparece en un manifiesto de producción, algo se hizo mal.

> **`LUMA_MASTER_KEY` no se rota sola.** Cifra los datos en reposo con
> ChaCha20-Poly1305; si la pierdes, pierdes lo cifrado. Guárdala donde guardas
> las claves, no en el `docker-compose.yml`.

### 3.2 Dónde vive `data_dir` decide tu velocidad de escritura

Esto no es un consejo de estilo. Mismo binario, misma configuración, mismo
cliente, misma máquina:

| `data_dir` en | `SET`/s | `GET`/s |
|---|---:|---:|
| Disco mecánico (HDD SATA) | 3.142 | 24.671 |
| SSD NVMe | **22.989** | 25.685 |

**7,3× en escritura, y la lectura no se mueve.** Esa asimetría es la firma exacta
de un camino dominado por la latencia de `fsync`: las lecturas se sirven de
memoria y nunca tocan el disco. Un `fsync` en un plato que gira cuesta un cuarto
de vuelta.

**Pon `data_dir` en SSD.** Si lo montas en red (NFS, EFS), mídelo antes de
prometer números: la latencia de `fsync` es lo único que importa ahí.

### 3.3 No lo expongas sin querer

Por defecto Luma escucha en `127.0.0.1`. Para exponerlo hay que decirlo en voz
alta — `BIND_ADDR=0.0.0.0` funciona pero imprime un aviso pidiéndote que uses la
forma explícita:

```bash
luma serve --bind 0.0.0.0      # o --unsafe-bind, que es su atajo
```

Y si lo expones, pon TLS (`TLS_CERT_PATH` / `TLS_KEY_PATH`) o un proxy delante.

### 3.4 El `luma.toml` que se distribuye **desactiva el rate limit**

Un detalle que muerde: el default del código es `rate_limit_rps = 100`, pero el
`luma.toml` de ejemplo trae `rate_limit_rps = 0`, y **0 significa sin límite**.
Si copias ese fichero a producción tal cual, te quedas sin cortafuegos de
peticiones. Ponlo a un número:

```toml
rate_limit_rps = 100
rate_limit_burst = 0     # 0 = 10× rps
```

### 3.5 Precedencia: quién gana

```
argumentos de CLI   >   variables de entorno   >   luma.toml   >   defaults
```

Si `luma.toml` no existe, se genera uno con los defaults al arrancar.

---

## 4. Variables de entorno que de verdad importan

Los nombres están sacados del código. Vale la pena decirlo porque la
documentación previa no coincidía: `docs/operar/CONFIG.md` nombraba
`PORT_RUST_KISS_VDB` y `RUSTKISS_API_KEY`, que no existen, y un flag `--logs`
que el binario nunca aceptó. Está corregida; si encuentras una diferencia entre
un documento y el código, gana el código.

El listado completo de las 122 claves está en
[`docs/operar/CONFIG.md`](docs/operar/CONFIG.md). Aquí van las que decides tú.

### Servidor y seguridad

| Variable | Default | Nota |
|---|---|---|
| `PORT_LUMA_VDB` | `1234` | **No es `PORT`.** `--port` gana sobre ella |
| `BIND_ADDR` | `127.0.0.1` | Prefiere `--bind` / `--unsafe-bind` |
| `LUMA_API_KEY` / `API_KEY` | — | Obligatoria, ≥16 caracteres |
| `LUMA_MASTER_KEY` | — | Obligatoria. Cifrado en reposo |
| `LUMA_ALLOW_INSECURE` | — | Solo desarrollo |
| `TLS_CERT_PATH` / `TLS_KEY_PATH` | — | TLS del puerto HTTP |
| `CORS_ALLOWED_ORIGINS` | — | Lista separada por comas |
| `RATE_LIMIT_RPS` | `100` | `0` desactiva |
| `REQUEST_TIMEOUT_SECS` | `30` | |
| `MAX_BODY_MB` / `MAX_JSON_MB` | `100` | En **MB**, no en bytes. `--max-body-mb` / `--max-json-mb` |

### Almacenamiento y durabilidad

| Variable | Default | Nota |
|---|---|---|
| `DATA_DIR` | `data` | **No es `LUMA_DATA_DIR`.** Ponlo en SSD |
| `WAL_SYNC_MODE` | `per_write` | Ver §4.1 |
| `WAL_SEGMENT_MAX_BYTES` | `67108864` (64 MiB) | |
| `WAL_RETENTION_SEGMENTS` | `8` | |
| `SNAPSHOT_INTERVAL_SECS` | `30` | |
| `SQLITE_ENABLED` | `true` | |

#### 4.1 Sobre `wal_sync_mode`, que cambió de significado

El default es `per_write`: cada escritura confirmada está en disco antes de que
Luma responda. Lo que cambió en v4.26.0 es **cuánto cuesta eso**.

- **En el camino KV/state ya no cuesta casi nada y `group` no hace nada.** Esas
  escrituras pasan por un *group commit* que hace `fsync` una vez por lote
  **siempre**, sin mirar esta variable. Medido: `group` da 1.497/s y `per_write`
  1.399/s — un 7%, donde antes había un 2,3×.
- **En los caminos que añaden evento a evento** —vector, documentos, blobs,
  colas— `group` sigue significando lo que siempre significó: una ventana de un
  lote (64) o 10 ms en la que una caída te cuesta esas escrituras, a cambio de
  throughput.

Traducción práctica: **déjalo en `per_write`**. La razón para no hacerlo casi
desapareció.

### Vectorial

| Variable | Default | Nota |
|---|---|---|
| `INDEX_KIND` | `IVF_FLAT_Q8` | `HNSW`, `IVF_FLAT_Q8` o `DiskANN` |
| `HNSW_SEARCH_EF` | `128` | Sube = más recall, más lento |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | `16` / `200` | |
| `IVF_CLUSTERS` / `IVF_NPROBE` | `4096` / `16` | |
| `MAX_VECTOR_DIM` | `4096` | |
| `MAX_K` | `256` | |
| `PRE_FILTER_THRESHOLD` | `10000` | Bajo ese tamaño, fuerza bruta en vez de HNSW + post-filtro |

Cómo elegir índice, con números medidos, en
[`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md). El resumen:
**DiskANN** para RAM mínima (133 MB en 50k×768), **HNSW** para recall,
**IVF_FLAT_Q8** como equilibrio.

### Embeddings

| Variable | Default | Nota |
|---|---|---|
| `EMBEDDING_PROVIDER` | `none` | `ollama`, `openai`, `azure`, `cohere`, `huggingface`, `mock` |
| `EMBEDDING_MODEL` / `EMBEDDING_URL` | — | |
| `EMBEDDING_DIM` | `384` | **Tiene que coincidir con el modelo** |
| `EMBEDDING_API_KEY` | — | |
| `EMBEDDING_RETRY_ATTEMPTS` | `3` | Backoff exponencial con jitter |
| `EMBEDDING_CACHE_SIZE` | `10000` | LRU por `proveedor::modelo::dim::texto` |
| `EMBEDDING_MAX_INFLIGHT_REQUESTS` | `16` | Semáforo hacia el proveedor |

> **`embedding_provider = "none"` viene de fábrica.** Es un perfil de desarrollo:
> el nivel 2 (`/v1/db`) y NS-Mem necesitan embeddings para hacer su trabajo. En
> tests usa `mock` para no depender de un servicio externo.

Prueba la configuración sin adivinar:

```bash
curl -X POST localhost:1234/v1/config/embedding/probe -H "Authorization: Bearer $KEY"
```

---

## 5. Encender cada puerta

### 5.1 Compatibilidad con Redis (RESP)

**Apagada por defecto** (`resp_port = 0`). Para encenderla:

```toml
resp_port = 6379
```

O `RESP_PORT=6379`. A partir de ahí, tus clientes existentes apuntan a Luma sin
cambiar una línea de código:

```bash
REDIS_URL=redis://:$LUMA_API_KEY@luma:6379/0
```

La contraseña de `AUTH` es tu `api_key`, o una API key de organización — en cuyo
caso la conexión queda ligada a esa org y sus claves van prefijadas.

| Variable | Default |
|---|---|
| `RESP_MAX_CLIENTS` | `10000` |
| `RESP_IDLE_TIMEOUT_SECS` | `300` |
| `RESP_MAX_BUFFER_BYTES` | `67108864` (64 MiB) |
| `RESP_PUBSUB_INBOX` | `1024` |
| `RESP_ALLOW_FLUSH` | `false` |
| `RESP_TLS_ENABLED` + `RESP_TLS_CERT_PATH` / `RESP_TLS_KEY_PATH` | `false` |

> **`FLUSHDB`/`FLUSHALL` vienen desactivados a propósito.** Un flush accidental
> desde un cliente mal configurado no se deshace sin restore.

> **Sin TLS, el `AUTH` viaja en claro.** Luma lo avisa al arrancar. Ponlo o deja
> el puerto en una red de confianza.

Los 117 comandos, por familia: strings y contadores, hashes (`H*`), listas
(`L*`/`R*`), sets (`S*`), sorted sets (`Z*`), expiración (`EXPIRE`, `TTL`,
`PERSIST` y sus variantes `P*`/`*AT`), escaneo (`SCAN`, `HSCAN`, `SSCAN`,
`ZSCAN`, `KEYS`), pub/sub, transacciones (`MULTI`/`EXEC`/`WATCH`), scripting
(`EVAL`/`EVALSHA`/`SCRIPT`) y los bloqueantes (`BLPOP`, `BRPOP`, `BLMOVE`,
`BZPOPMIN`/`BZPOPMAX`). La lista exacta y en qué diverge de Redis:
[`docs/integrar/RESP.md`](docs/integrar/RESP.md).

**Rendimiento**, para que decidas con datos y no con fe — mismo cliente, misma
ruta de red, 256 clientes, SSD NVMe:

| | `SET`/s | `GET`/s |
|---|---:|---:|
| Redis 7 | 28.517 | 27.298 |
| **Luma** | **22.989** | **25.685** |

81% de la escritura de Redis y 94% de su lectura — **haciendo `fsync` de cada
escritura confirmada, que Redis de fábrica no hace**. Redis por defecto responde
OK antes de que el dato esté en el medio.

### 5.2 API compatible con S3

**Apagada por defecto** (`s3_port = 0`):

```toml
s3_port = 9000
```

Puerto propio, no una ruta: S3 se adueña de la raíz (`GET /` es ListBuckets), así
que compartir router con `/v1/...` haría que uno tapara al otro.

Credenciales, como admin de una organización:

```bash
curl -X POST localhost:1234/v1/admin/s3-credentials -H "Authorization: Bearer $KEY"
```

Los objetos son **los mismos bytes** que el API nativo de blobs
(`{data_dir}/blobs/{bucket}/{key}`): una cuota, un respaldo, una fuente de
verdad. Detalles y límites en [`docs/integrar/S3.md`](docs/integrar/S3.md).

### 5.3 Conector de Postgres (CDC)

Postgres sigue siendo tu fuente de verdad. Luma consume su replicación lógica y
mantiene una copia derivada con forma de índice de búsqueda; **nada escribe de
vuelta**.

```toml
# erp.toml
name = "erp"
url  = "postgres://luma:secreto@db:5432/erp?sslmode=require"
slot = "luma_cdc"
publication = "luma_cdc"
backfill = true
flush_interval_secs = 10

[[tables]]
table        = "sales.orders"
namespace    = "orders"
text_columns = ["customer", "notes"]
skip_columns = ["internal_token"]
```

```bash
luma connect postgres erp.toml              # sigue para siempre
luma connect postgres erp.toml --once       # una pasada acotada, para cron o test
luma connect postgres erp.toml --no-backfill
```

Requisitos del lado Postgres y qué hace con las columnas TOAST:
[`docs/integrar/POSTGRES-CDC.md`](docs/integrar/POSTGRES-CDC.md).

---

## 6. Operación

### Subcomandos

```bash
luma serve                          # también es lo que hace `luma` a secas
luma backup [--verify]              # --verify lo lee de vuelta contra su manifiesto
luma restore <ruta>
luma vacuum --collection <nombre>
luma diskann build|tune|status ...
luma promote                        # réplica → primario
luma demote                         # primario → réplica
luma role                           # qué es este data_dir (solo lectura)
luma connect postgres <config.toml>
luma help
```

Un subcomando mal escrito **es un error, no un servidor**: `luma backupp` no
levanta un listener en el puerto de producción.

### Respaldos

```toml
backup_enabled = true
backup_dir = "backups"
backup_interval_secs = 86400
backup_retention = 7
```

Un respaldo contiene SQLite (vía `VACUUM INTO`, así que es consistente),
`snapshot.json`, todos los segmentos del WAL, `vectors/`, `blobs/` y `queues/`.

**No contiene la proyección KV (`state.lsm`), y es a propósito.** Es una
proyección del WAL: el restore la reconstruye reproduciendo. Copiarla en caliente
falla en Windows con violación de compartición y en Linux produce una lectura
desgarrada, que es peor que no tenerla.

Un respaldo que nadie ha restaurado es una hipótesis — por eso existe
`--verify`, y por eso deberías restaurar uno de vez en cuando.

### Salud y métricas

| Endpoint | Para qué |
|---|---|
| `GET /v1/health` | Liveness |
| `GET /v1/health/primary` | Si este nodo es primario |
| `GET /v1/metrics` | Prometheus |
| `GET /v1/admin/stats` | Panel |
| `GET /v1/admin/audit` | Log de auditoría, filtrable |
| `GET /v1/admin/resp` | Conexiones RESP activas |

Dashboards y alertas listos en [`docs/observability/`](docs/observability/).

### Panel de administración

Va **dentro del binario** (React compilado e incrustado con `rust-embed`), en la
raíz del puerto HTTP. Login, orgs, usuarios, API keys, auditoría y salud. No hay
que desplegar nada aparte.

### Documentación del API en vivo

`GET /docs` sirve la spec OpenAPI navegable.

---

## 7. Multi-tenancy y permisos

- **Organizaciones y usuarios** con `/v1/auth/register`, `/v1/auth/login`
  (Argon2id), sesiones con token opaco `lums_…` de 7 días del que solo se
  persiste el hash SHA-256.
- **Roles**: `owner` (40) > `admin` (30) > `member`/`user` (20) >
  `viewer`/`readonly` (10).
- **Aislamiento por organización**: la propiedad de una colección se fija en el
  primer uso; un acceso desde otra organización devuelve **404**, no 403 — no
  confirmamos que el recurso exista.
- **Auditoría**: cada petición deja `ts`, `api_key_id`, `ip`, `method`, `path`,
  `status`, `latency_ms`.

---

## 8. Límites y qué NO hace

Lo que no está, dicho antes de que te haga falta:

- **No hay replicación multi-máquina con failover automático.** Hay
  primario/réplica con envío de WAL y `promote`/`demote` manuales.
- **No es para colas de 10 M de mensajes residentes.** El motor es memoria + WAL;
  las estructuras grandes viven en RAM.
- **La compatibilidad Redis no es Redis.** No hay cluster, ni módulos, ni
  `SETRANGE`/`BITCOUNT`/streams. La matriz completa está en
  `docs/integrar/RESP.md` y es la fuente de verdad, no este párrafo.
- **La API S3 no ha pasado la suite mint de MinIO.**
- **`data_dir` en disco mecánico** te da un tercio de la escritura. Ya lo dije en
  §3.2; lo repito porque es el error más caro y el más fácil de cometer.

### Límites configurables

`MAX_K` 256 · `MAX_VECTOR_DIM` 4096 · `MAX_STATE_BATCH` 256 · `MAX_VECTOR_BATCH`
256 · `MAX_DOC_FIND` 100 · `MAX_ID_LEN` 128 · `MAX_KEY_LEN` 512 · `MAX_COLLECTION_LEN`
64 · `MAX_BODY_MB` / `MAX_JSON_MB` 100. Todos
tienen su variable de entorno con el mismo nombre.

---

## 9. Errores típicos

| Síntoma | Causa |
|---|---|
| `refusing to start with N insecure secret setting(s)` | Falta `LUMA_API_KEY` (≥16 chars) o `LUMA_MASTER_KEY` |
| Cambio `PORT` y sigue en 1234 | La variable es **`PORT_LUMA_VDB`**, o usa `--port` |
| Cambio `LUMA_DATA_DIR` y no lo coge | La variable es **`DATA_DIR`** |
| `WRONGPASS` por RESP | La contraseña de `AUTH` es tu `api_key`, no un usuario |
| `NOAUTH Authentication required` | El cliente no manda `AUTH`; pon la contraseña en la `REDIS_URL` |
| Escrituras lentísimas (~3.000/s) | `data_dir` en disco mecánico → §3.2 |
| `/v1/db` y `/v1/memory` no encuentran nada | `embedding_provider = "none"`, que es el default |
| Sin límite de peticiones en producción | El `luma.toml` de ejemplo trae `rate_limit_rps = 0` → §3.4 |
| Datos perdidos al recrear el contenedor | Falta el volumen para `data_dir` |
| `subcomando desconocido` | Está bien: un subcomando mal escrito no arranca un servidor |

---

## 10. Dónde seguir leyendo

| Documento | Para qué |
|---|---|
| [`docs/integrar/API.md`](docs/integrar/API.md) | Referencia de endpoints |
| [`docs/integrar/RESP.md`](docs/integrar/RESP.md) | Matriz de comandos Redis y divergencias |
| [`docs/integrar/S3.md`](docs/integrar/S3.md) | API S3 |
| [`docs/integrar/POSTGRES-CDC.md`](docs/integrar/POSTGRES-CDC.md) | Conector Postgres |
| [`docs/integrar/NS_MEM.md`](docs/integrar/NS_MEM.md) | Memoria de agentes |
| [`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md) | Números medidos, y los diseños que se descartaron |
| [`docs/operar/RUNBOOKS.md`](docs/operar/RUNBOOKS.md) | Procedimientos de operación |
| [`docs/operar/PROD_READINESS.md`](docs/operar/PROD_READINESS.md) | Matriz de durabilidad |
| [`docs/operar/THREAT_MODEL.md`](docs/operar/THREAT_MODEL.md) | Qué se protege y de quién |
| [`docs/operar/CLI.md`](docs/operar/CLI.md) | Subcomandos en detalle |
