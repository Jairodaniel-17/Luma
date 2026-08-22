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
| **WAL de eventos** (`events-NNNNNN.log`) — respalda todas las mutaciones | **fsync diferido** por defecto: `wal_sync_mode = "group"`, lotes de `wal_batch_size` (64) o `wal_flush_interval_ms` (10 ms) | Hasta un lote o 10 ms de mutaciones confirmadas. Poner `wal_sync_mode = "per_write"` lo elimina a costa de throughput |
| **Blob** (`/v1/blob`) | **fsync** del fichero, luego del directorio tras el rename | Ninguno en Linux. En Windows la entrada de directorio depende del journaling de NTFS |
| **Colas** (`/v1/queue`) | **fsync** del fichero del mensaje, luego del directorio | Igual que blob. Un `enqueue` confirmado no se pierde |
| **Manifest de colección vectorial** | **fsync** del temporal, rename, fsync del directorio | Igual que blob |
| **Runs de vectores** (`runs/*.run`) | `sync_data` por registro en el camino unitario; en el camino por lotes un único `sync_active_run` al cerrar el lote | En el camino por lotes, hasta un lote de vectores. El WAL sigue teniéndolos, así que el replay los recupera |
| **KV respaldado por redb** | **reconstruible**: las transacciones usan `Durability::Eventual`, sin fsync por commit | Ninguna pérdida de datos: redb es una proyección del WAL y el replay la reconstruye desde `applied_offset` |
| **SQLite** (relacional, auth, docstore, NS-Mem) | **fsync diferido**: modo WAL con `synchronous = NORMAL` | Un corte de energía puede perder commits recientes. No corrompe la base (eso exigiría `synchronous = OFF`). `FULL` lo elimina a costa de latencia de escritura |
| **Snapshots** (`snapshot.json`) | Escritura periódica, no en el camino de la petición | No aplica: el snapshot solo acorta el replay, nunca es la única copia |

### Lo que esta tabla deja ver

Dos puntos que conviene decidir de forma consciente antes de producción, no
descubrir después:

1. **El default de `wal_sync_mode` es `group`, no `per_write`.** Es la decisión
   correcta para throughput, pero significa que "no pierde datos confirmados"
   tiene una ventana de 64 escrituras o 10 ms. Quien necesite RPO cero por
   escritura tiene que cambiarlo explícitamente.
2. **SQLite corre con `synchronous = NORMAL`.** Las cuentas, la auditoría y las
   tablas de NS-Mem viven ahí. Un corte de energía no corrompe nada, pero puede
   perder los últimos commits — incluida una alta de usuario que la API ya
   confirmó.

El objetivo de despliegue es **Linux** (imagen musl). Windows es plataforma de
desarrollo soportada, no de producción con garantías equivalentes: NTFS no
expone flush de directorio, así que la durabilidad de los renames descansa en
su journaling de metadatos y no en un flush que hagamos nosotros.

> Pendiente de W1.1: la matriz de crash-recovery que **demuestra** esta tabla
> matando el proceso durante ráfagas de escritura de cada motor. Hasta que
> exista, la tabla describe el código auditado, no un comportamiento verificado
> bajo fallo.

## Respaldos: qué cubren (W1.4)

`luma backup` copia **todo** el estado persistente:

| Contenido | Cómo se copia |
|---|---|
| SQLite | `VACUUM INTO` — copia consistente sin parar el actor |
| `snapshot.json` | copia directa |
| Segmentos del WAL | todos los `events-*.log` presentes |
| `state.redb` | copia directa. Es reconstruible desde el WAL, pero copiarlo hace que un restore arranque servido en vez de replayando |
| `vectors/` | árbol completo: manifest, runs y mmaps |
| `blobs/` | árbol completo |
| `queues/` | árbol completo |

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
