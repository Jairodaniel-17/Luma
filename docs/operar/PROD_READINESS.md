# Prod Readiness (KISS)

## Requisitos mínimos

- **Durabilidad**: define `DATA_DIR` en producción (habilita WAL segmentado + snapshots).
- **Integridad WAL**: los segmentos nuevos incluyen checksum por registro. Si el proceso cae a mitad de append, Luma recupera el prefijo válido y contabiliza corrupción detectada en métricas.
- **Bind seguro**: sin flags el binario sólo escucha en `127.0.0.1`. Usa `--bind 0.0.0.0` o `--unsafe-bind` sólo si lo pones detrás de un proxy.
- **Auth**: exporta `RUSTKISS_API_KEY`/`API_KEY` para exigir `Authorization: Bearer …`. Si no lo haces, las rutas quedan abiertas (útil para laboratorio, no prod).

## Durabilidad por primitiva (W1.2)

Qué garantiza cada primitiva **en el momento en que devuelve OK**. Auditado
extremo a extremo, no inferido de la configuración.

Leyenda de "confirmado":
- **fsync** — los datos están en el medio antes de responder.
- **fsync diferido** — se responde antes del flush; una caída puede perder una
  ventana acotada de escrituras confirmadas.
- **reconstruible** — no se sincroniza, pero el WAL es la fuente de verdad y el
  replay al arrancar lo regenera. La pérdida no es de datos, es de trabajo.

| Primitiva | Al devolver OK | Riesgo real de una caída |
|---|---|---|
| **WAL de eventos** (`events-NNNNNN.log`) — respalda todas las mutaciones | **fsync** por defecto: `wal_sync_mode = "per_write"` | Ninguna. `wal_sync_mode = "group"` sigue disponible y abre una ventana de un lote (64) o 10 ms, a cambio de ~2,3× de throughput |
| **Blob** (`/v1/blob`) | **fsync** del fichero, luego del directorio tras el rename | Ninguno en Linux. En Windows la entrada de directorio depende del journaling de NTFS |
| **Colas** (`/v1/queue`) | **fsync** del fichero del mensaje, luego del directorio | Igual que blob. Un `enqueue` confirmado no se pierde |
| **Manifest de colección vectorial** | **fsync** del temporal, rename, fsync del directorio | Igual que blob |
| **Runs de vectores** (`runs/*.run`) | `sync_data` por registro en el camino unitario; en el camino por lotes un único `sync_active_run` al cerrar el lote | En el camino por lotes, hasta un lote de vectores. El WAL sigue teniéndolos, así que el replay los recupera |
| **KV respaldado por un LSM** (`state.lsm`, fjall) | **reconstruible**: nada hace fsync en el camino de escritura | Ninguna pérdida de datos: es una proyección del WAL y el replay la reconstruye desde `applied_offset`. Las tres rutas de escritura pasan por un único `keyspace.batch()` que incluye `applied_offset`, así que el offset y los datos no pueden divergir tras una caída |
| **SQLite** (relacional, auth, docstore, NS-Mem) | **fsync**: modo WAL con `synchronous = FULL` | Ninguna. Cuesta ~9× frente a `NORMAL` (23 560 → 2 513 escrituras/s medidas), y esas tablas no están en el camino de volumen |
| **Snapshots** (`snapshot.json`) | Escritura periódica, no en el camino de la petición | No aplica: el snapshot solo acorta el replay, nunca es la única copia |

### Lo que esta tabla deja ver

Los dos puntos que esta sección declaraba como «decidir conscientemente» están
**cerrados**, y vale la pena decir por qué, porque el argumento que sostenía uno
de ellos era falso:

1. **`wal_sync_mode` ya es `per_write`.** El default era `group`, y el
   comentario que lo defendía decía que la durabilidad de una escritura
   confirmada estaba a salvo *«porque el state store (redb) y los segmentos
   vectoriales sí hacen fsync inmediato»*. redb no lo hace: `state_db.rs` pone
   `Durability::Eventual` en sus tres rutas de escritura. Ni el WAL ni su
   proyección llegaban al disco antes de que `put_state` devolviera OK.

   La matriz de crash-recovery pasaba igual, pero **por tiempo y no por
   garantía**: sus viajes HTTP tardan más que el flush de fondo de 10 ms, así
   que todo estaba en disco cuando mataba el proceso.

   Coste medido del default honesto: 1 964 → 848 escrituras/s
   (`tests/wal_sync_cost.rs`). `group` sigue disponible para quien prefiera
   throughput sabiendo lo que compra.

2. **SQLite ya corre con `synchronous = FULL`.** Y la matriz de crash-recovery
   dejó de excluirlo: corría con `SQLITE_ENABLED=false` porque su durabilidad
   era «una cuestión aparte». No lo era — era la misma cuestión con una
   respuesta incómoda.

La durabilidad de la proyección KV **sí** era correcta y sigue igual —aunque
debajo ya no está redb sino un LSM: es una proyección del
WAL, y en un crash vuelve a su último commit inmediato mientras el replay
re-aplica el resto. Ese diseño solo funciona si el WAL es durable de verdad, que
es justo lo que faltaba.

El objetivo de despliegue es **Linux** (imagen musl). Windows es plataforma de
desarrollo soportada, no de producción con garantías equivalentes: NTFS no
expone flush de directorio, así que la durabilidad de los renames descansa en
su journaling de metadatos y no en un flush que hagamos nosotros.

> La matriz de crash-recovery **demuestra** esta tabla: mata el proceso durante
> ráfagas de escritura de cada motor y comprueba que lo confirmado sobrevive.
> Corre en CI en cada push y con muchas más iteraciones en el nightly, ahora
> incluyendo SQLite.

## Respaldos: qué cubren (W1.4)

`luma backup` copia **todo** el estado persistente:

| Contenido | Cómo se copia |
|---|---|
| SQLite | `VACUUM INTO` — copia consistente sin parar el actor |
| `snapshot.json` | copia directa |
| Segmentos del WAL | todos los `events-*.log` presentes |
| `state.lsm` (proyección KV) | **no se copia** — ver abajo |
| `vectors/` | árbol completo: manifest, runs y mmaps |
| `blobs/` | árbol completo |
| `queues/` | árbol completo |
| Estructuras (listas, hashes, sets, zsets) | van dentro del WAL y el snapshot: se guardan bajo el prefijo `struct:` del KV. Hay test de round-trip completo |

> **Por qué la proyección no se copia.** Es una proyección del WAL: el restore lo
> reconstruye. Una versión anterior sí lo copiaba, para que el restore arrancara
> servido en vez de replayando, y estaba mal por partida doble: el fichero está
> abierto y mapeado por el engine en marcha, así que en Windows la copia falla
> con violación de compartición —rompiendo `luma backup` entero— y en Linux
> funciona produciendo una lectura rota. Un redb roto es peor que ninguno: el
> restore arrancaría desde estado derivado corrupto en lugar de reconstruirlo
> limpio.

Cada backup lleva un `manifest.json` con la versión que lo escribió y los
conteos de cada cosa. Sin él un restore es adivinar: no se distingue un backup
vacío de uno cuyo directorio de vectores falló al copiarse en silencio.

> **Histórico, por si aparece un backup viejo:** hasta esta versión el backup
> copiaba solo SQLite + snapshot + WAL. Vectores, blobs y colas quedaban fuera.
> Blobs y colas **no están en el WAL**, así que en un backup anterior a este
> cambio esa pérdida es definitiva; los vectores solo se recuperarían si aún se
> conservan los segmentos del WAL que los construyeron, cosa que
> `wal_retention_segments` garantiza que no.

### Verificación

`luma backup --verify` restaura el backup, corre `PRAGMA integrity_check` sobre
el SQLite y compara los conteos reales contra el manifest. La tarea de fondo
verifica **cada** backup que toma: leer de vuelta lo recién escrito es barato al
lado de producirlo, y es la diferencia entre tener backups y creer que se tienen.
Un backup que no verifica se reporta como **error**, no como un backup correcto
con una nota — quien lea los logs no puede interpretarlo como "hay backup".

### Respaldo fuera del host (W1.3)

Opt-in. Con `backup_remote_url` vacío no cambia nada; el backup local sigue igual.

| Clave | Para qué |
|---|---|
| `backup_remote_url` | Destino, `s3://bucket/prefijo`. La ruta del URL es el prefijo dentro del bucket, así que varias instancias pueden compartirlo sin chocar |
| `backup_remote_endpoint` | Endpoint propio: MinIO, R2 u otro compatible con S3. Vacío = AWS S3 real |
| `backup_remote_region` | Región |
| `backup_remote_allow_http` | Permite un endpoint en HTTP plano. Solo para un MinIO en red de confianza: los artefactos van cifrados, **las credenciales de la petición no** |

Las credenciales salen de las variables AWS estándar, así que un instance
profile, un `.env` o un secreto de Kubernetes funcionan sin fontanería propia de
Luma.

**Los artefactos se cifran antes de salir del host** con la master key, con la
misma caja ChaCha20-Poly1305 del cifrado en reposo. Importa porque un bucket de
backups suele ser la copia peor vigilada de un sistema: sobrevive a los hosts, se
comparte con quien vaya a restaurar, y es lo primero que expone una policy mal
puesta.

Dos garantías de orden que evitan los fallos silenciosos típicos:

- **El manifest se sube el último.** Una subida interrumpida deja un prefijo sin
  manifest, y la descarga rechaza ese prefijo en vez de restaurar un backup
  parcial que parece completo.
- **La poda va después de una subida correcta**, nunca antes. Perder la copia
  remota más nueva para hacer sitio a una que luego falla al subir es el peor
  orden posible.

Un fallo remoto se registra pero no es fatal: el backup local ya salió bien, y
convertir un problema de red pasajero en una ejecución fallida tiraría una copia
buena para nada.

### WAL shipping continuo (W2.1)

`wal_ship_interval_secs` (0 = desactivado, requiere `backup_remote_url`).

Un backup completo da una foto cada varias horas; todo lo escrito desde la
última vive en un solo disco. El shipping sube los segmentos sellados según se
cierran y **reenvía el que está creciendo** en cada intervalo.

> **Ese intervalo *es* el RPO.** Una máquina perdida entre ticks pierde como
> mucho un intervalo de escrituras. Es el número que va en el runbook, y por eso
> se registra al arrancar en vez de dejarlo a deducir.

Se suben los segmentos en crudo, no un formato de replicación aparte: un
snapshot más la cadena de segmentos posterior es exactamente lo que el servidor
reconstruye al arrancar, así que **la ruta de recuperación es la ruta de
arranque**, ya cubierta por la matriz de crash-recovery. Recuperar es "descargar
y arrancar", sin un paso de aplicación separado que equivocar.

El snapshot se sube **primero**: un bucket con segmentos más nuevos que su
snapshot es recuperable; uno con un snapshot más nuevo que sus segmentos no lo
es, porque el replay arrancaría después de eventos que nunca subieron.

Un segmento que no ha cambiado no se resube — se compara la longitud, que basta
porque el WAL es append-only y nunca se reescribe en sitio. Un intervalo tranquilo
cuesta un listado de directorio.

**Esto no es replicación.** Nada sigue el stream y lo aplica en vivo; eso es
W2.2 (réplica de lectura). Esto es recuperación ante desastre: el bucket guarda
lo suficiente para reconstruir la instancia en otra máquina, con una pérdida
máxima declarada.

## Observabilidad (W5.1)

`/v1/metrics` sirve formato Prometheus (`text/plain; version=0.0.4`), con
contadores, gauges e histogramas de latencia por etapa.

Stack de demostración listo para levantar:

```bash
docker compose -f docs/observability/docker-compose.yml up
# Grafana en http://localhost:3000, sin login, con el dashboard ya cargado
```

Todo se aprovisiona desde ficheros, no se clica después: el criterio de
aceptación es que arranque mostrando el dashboard **sin editar nada**.

| Fichero | Qué es |
|---|---|
| `docs/observability/dashboard.json` | Dashboard: durabilidad, throughput, latencias por etapa, embeddings, SSE, RESP y capacidad |
| `docs/observability/alerts.yml` | Reglas de alerta de Prometheus |
| `docs/observability/prometheus.yml` | Scrape config (apúntalo a tu instancia) |

### Por qué hay un test sobre esto

`tests/observability.rs` comprueba que **cada métrica que nombran el dashboard y
las alertas existe de verdad** en la salida del servidor.

Un panel que dibuja la nada parece un sistema ocioso, y una regla de alerta
sobre una métrica renombrada no salta nunca — que es indistinguible de que todo
va bien. Las dos cosas son peores que no tenerlas. El test falla justo cuando
alguien renombra una métrica sin tocar estas piezas, y está verificado que falla
de verdad, no por vacuidad.

También valida que la salida entera es texto Prometheus legal: Prometheus
rechaza el scrape completo ante una sola línea malformada, así que una métrica
rota se lleva por delante a todas las demás.

## Cuotas por organización (W5.2, parcial)

Multi-tenancy que aísla pero no limita es la mitad del trabajo: una organización
puede llenar el disco, y el resto se enteran en el mismo momento que el
operador.

Las cuotas viajan en el registro de la api key (`quotas`), que ya existía como
JSON sin tipo y **nadie leía**. Ahora tiene tipo y se aplica.

```json
{ "max_keys": 100000, "max_vectors": 1000000,
  "max_blob_bytes": 10737418240, "max_queue_messages": 50000 }
```

Todos los campos son opcionales y ausente significa **sin límite**, así que un
registro `{}` —que es lo que llevan todas las keys hoy— se comporta exactamente
igual que antes.

| Recurso | Estado |
|---|---|
| `max_keys` | **Aplicado.** El keyspace lleva prefijo de organización, así que medir el uso del llamante es un escaneo por prefijo |
| `max_vectors`, `max_blob_bytes`, `max_queue_messages` | Parsean y `check` los respeta, pero **nada los invoca todavía** — a propósito, ver abajo |

### Decisiones

- Se rechaza **la escritura que cruzaría el límite**, no las siguientes. Un
  límite de 100 claves admite la centésima y rechaza la 101, que es lo que hace
  que el número de la config sea el número sobre el que razonar.
- **Sobrescribir no consume cuota.** Si contara, los datos pasarían a ser de
  solo lectura en el instante de alcanzar el límite, y eso no es lo que
  significa un límite de almacenamiento.
- Las lecturas **nunca** se rechazan. Una organización en su límite puede sacar
  sus datos, que es justo el objetivo de decirle que limpie.
- **507 Insufficient Storage, no 429.** No se está limitando el ritmo: no queda
  sitio, y reintentar lo mismo no va a funcionar nunca. Un 429 invitaría
  precisamente al bucle de reintentos que no sirve de nada.
- Un registro de cuota ilegible se trata como **sin límite**, y se grita en los
  logs. Un error de tipeo en la config no puede convertirse en una caída.
- El error nombra uso, petición y límite. "Cuota excedida" a secas deja al
  llamante sin saber si tiene que borrar una cosa o mil.

### Por qué faltan tres

El keyspace lleva prefijo de organización; los blobs no — el layout es
`blobs/{bucket}/…` con la propiedad registrada aparte en `sys_collections`. Un
recorrido de directorio mide los bytes de **todas** las organizaciones, y cobrar
a una los bytes de otra rechazaría la escritura de B porque A llenó el disco:
exactamente el fallo que el criterio de aceptación de este ítem prohíbe, y peor
que no tener cuota.

Aplicarlos requiere consultar el índice de propiedad por bucket, que es una
consulta asíncrona que este guardián sincrónico no puede hacer. Es el siguiente
paso, no un atajo que tomar ahora.

## SSE tuning

- `LIVE_BROADCAST_CAPACITY`: sube si hay bursts (default `4096`).
- Clientes lentos: el servidor no se cae; emite `event: gap` y el cliente debe reconectar usando `since=<last_offset>`.
- Si el `since` pedido ya no está retenido por buffer o WAL, Luma emite `event: gap` con `from_offset`/`to_offset` antes del primer evento disponible.
- Proxies: asegúrate de permitir `text/event-stream` y deshabilitar buffering (p.ej. nginx `proxy_buffering off`).

## Límites Anti-DoS

- `MAX_BODY_BYTES`: límite duro de request body.
- `MAX_JSON_BYTES`: límite duro para `value/meta/filters`.
- `MAX_VECTOR_DIM`, `MAX_K`, `MAX_KEY_LEN`, `MAX_ID_LEN`, `MAX_COLLECTION_LEN`.

## Retención del log

- `WAL_SEGMENT_MAX_BYTES`: tamaño de segmento.
- `WAL_RETENTION_SEGMENTS`: cantidad de segmentos retenidos.
- Si necesitas replay largo, aumenta `WAL_RETENTION_SEGMENTS` o reduce snapshot interval.
- Observa `wal_replay_corrupt_total`, `wal_gap_total` y `wal_rotation_total` en `/v1/metrics`.

## CORS

- Dev: sin `CORS_ALLOWED_ORIGINS` (acepta Any).
- Prod: define `CORS_ALLOWED_ORIGINS=https://tuapp.com,https://admin.tuapp.com`.

## Timeouts

- `REQUEST_TIMEOUT_SECS` aplica a requests HTTP normales; SSE mantiene keepalive.

## Logs

- Usa `--logs info|warning|error|critical` para ajustar el nivel sin tocar `RUST_LOG`. En producción se recomienda `--logs warning` + redirect estándar a tu stack centralizado.

## Vector Store

- En disco se mantiene `vectors/<collection>/{manifest.json,vectors.bin}`. Cada mutaci¢n es append-only; borra = tombstone. Usa `rust-kiss-vdb vacuum --collection <name>` para compactar sin reiniciar.
- Rebuild al arranque = leer `vectors.bin` + recrear segmentos/HNSW. Costo observado: ~120 ms por cada 10k vectores (dim 384) en laptop m3.
- Límites recomendados (v1): `dim <= 1536`, `k <= 200`, `<= 1e6` vectores por colección (más allá considera sharding o instancias extra).
- SSE vectorial expone `collection` en `data` y respeta `?collection=foo` en `/v1/stream`. También se envía `event: vector_*` para auditar ingestas.
- Los filtros por metadata usan un índice exact-match; dimensiona la RAM según tu cardinalidad.
- La compactación HNSW en memoria es opt-in (`HNSW_SEGMENT_COMPACTION_ENABLED=1`) y ahora rebuilda fuera del lock de escritura, intercambiando segmentos solo si el `applied_offset` no cambió durante el rebuild.

## Estado / KV

- `/v1/state` soporta `?prefix=` para scans por prefijo y `?start=&end=` para scans lexicográficos end-exclusive.
- CAS (`if_revision`) y replay aplican revisiones de forma idempotente; una revisión más antigua no pisa una más nueva.

## DocStore / SQL

- DocStore vive sobre KV (`doc:{collection}:{id}` + `docidx:*`). Ideal para dashboards/configuraciones ligeras.
- SQLite embebido (`SQLITE_ENABLED=1`) comparte proceso pero NO WAL; respáldalo como parte del backup del `DATA_DIR`.
- Ambos módulos reutilizan el middleware de auth/API key; si no los necesitas mantenlos desactivados.
- El hub híbrido publica métricas por etapa: `hybrid_sql_prefilter_duration_ms`, `hybrid_vector_duration_ms`, `hybrid_hydration_duration_ms`, `hybrid_chunking_duration_ms`, `hybrid_vector_write_duration_ms`, `hybrid_sql_write_duration_ms`.
- También expone contadores/gauges útiles para explicar consultas: `hybrid_sql_first_total`, `hybrid_vector_first_total`, `hybrid_last_sql_candidates`, `hybrid_last_vector_candidates`, `hybrid_last_doc_candidates`, `hybrid_last_hydrated_docs`.
