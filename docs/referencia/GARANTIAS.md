# Garantías por motor

Qué promete Luma en cada motor sobre atomicidad, aislamiento, consistencia y durabilidad, con qué motor conocido se
compara y qué prueba lo sostiene. Lo que no está en esta tabla **no** se promete.

Durabilidad común a todos: con `wal_sync_mode = "per_write"` (el valor por defecto), una respuesta 200 significa que
el dato ya pasó por `fsync`. Si un `fsync` falla, Luma no reintenta (en Linux podría dar un OK mentiroso): rechaza la
escritura y queda en solo lectura hasta reiniciar. Con el disco casi lleno rechaza con 507 y sigue leyendo.

| Motor | Atomicidad | Aislamiento | Se compara con |
|---|---|---|---|
| KV (`/v1/state`) | Cada operación. `POST /v1/state/txn` es una transacción: varias claves en **un solo registro** del WAL, todo o nada. (`batch_put` no: aplica una por una y reporta cada una) | Por clave, linealizable. Una transacción bloquea sus claves y valida cada revisión antes de escribir; con `check` sobre lo que solo leyó es **serializable** | etcd `txn`; PostgreSQL `SERIALIZABLE` para transacciones optimistas |
| Documentos (`/v1/doc`) | El documento y sus entradas de índice se escriben en el mismo lote | Por documento; un `find` nunca ve un índice que no corresponde al documento | MongoDB (un documento) |
| Estructuras RESP (listas, hashes, sets, zsets) | Cada comando, también los de varias claves (`LMOVE`, `RENAME`, `MSET`) | `MULTI`/`EXEC` aislado **frente a otros clientes RESP**; no frente a escrituras por HTTP en las mismas claves | Redis 7 (diferencial de 327 comandos en CI) |
| Colas (`/v1/queue`) | Encolar y confirmar (ack) son atómicos y durables | Al menos una vez: un mensaje recibido y no confirmado vuelve a ser visible al vencer su lease | Amazon SQS estándar |
| Blobs (`/v1/blob`, S3) | Escritura a temporal + `fsync` + rename + `fsync` del directorio: se ve el objeto entero o el anterior, nunca uno a medias | Lectura tras escritura | Amazon S3 |
| Vectores (`/v1/vector`) | Cada upsert/borrado, y cada lote | Por id | Qdrant / pgvector (sin transacciones) |

## Lo que no hay, a propósito

- **Transacciones interactivas entre motores.** No existe un `BEGIN … COMMIT` que abarque una clave, un blob y un
  mensaje. Quien lo necesita (una factura, su asiento y su salida de almacén) lo hace en una base relacional; así lo
  hace el ERP de Friochap con su SQLite (`docs/SPEC-friochap.md`).
- **Transacciones interactivas con bloqueo.** Las de Luma son optimistas: se lee, se arma la transacción con las
  revisiones leídas y, si alguna cambió, responde 409 y el cliente reintenta. No hay `SELECT … FOR UPDATE`.
- **Lecturas de una instantánea de varias claves.** Cada lectura ve lo último confirmado de su clave.

## Qué lo sostiene

| Garantía | Prueba |
|---|---|
| Ninguna escritura confirmada se pierde ni se adelanta en el WAL bajo concurrencia | `tests/acid_concurrency.rs`, también en arm64 (`test-arm64` en CI) |
| CAS, lotes y índices de documentos sin carreras ni ABA | `tests/acid_concurrency.rs` |
| Transacciones todo o nada, también tras reiniciar; sin write skew con `check` (200 rondas concurrentes) | `tests/acid_concurrency.rs`, `tests/http_state_vector.rs` |
| Comandos RESP atómicos, `EXEC` aislado | tests de `src/resp/commands.rs`; diferencial contra Redis 7 |
| Corte de luz (se pierde la caché de páginas) | `scripts/apagon/` con LazyFS: 1 115 cortes sin pérdidas; 200 cada noche |
| Reinicio real sin sincronizar | `scripts/sysrq/run.sh`: `echo b > /proc/sysrq-trigger` en Debian 13 con la unidad de producción: 50 reinicios en x86 y 50 en arm64, 0 pérdidas |
| `kill -9` en cada camino de escritura | `tests/crash_recovery.rs` (nightly) |
| Disco lleno y `fsync` con EIO | `tests/disk_full.rs`, `tests/fsync_eio.rs` |
| Host perdido: lo confirmado hasta 10 s antes está en la máquina nueva | `scripts/dr_drill.sh`, cada noche contra OCI Object Storage real: 4,8–4,9 s, máquina nueva lista en 27–28 s |
| Una versión nueva lee lo que escribió la anterior | `tests/golden_data_dir.rs` |

## Contra otros motores, con el mismo corte de luz

`scripts/apagon/comparar.sh` (en GitHub: «Durability against other engines», semanal): la misma carga para todos
(clave = valor desde seis hilos; cuenta solo lo que el motor confirmó), `kill -9` en un momento al azar, LazyFS tira lo
que no llegó a disco, se reinicia y se cuentan las confirmadas que faltan. 15 cortes por motor, 27/10/2026:

| Motor | Escrituras confirmadas | Perdidas |
|---|---|---|
| **Luma** | 118 475 | **0** |
| Redis 7, `appendfsync always` | 133 882 | 0 |
| PostgreSQL, `synchronous_commit=on` | 133 239 | 0 |
| SQLite (WAL), `synchronous=FULL` | 54 739 | 0 |
| Redis 7, `appendfsync everysec` | 251 380 | 40 808 |
| PostgreSQL, `synchronous_commit=off` | 154 443 | 7 943 |
| SQLite (WAL), `synchronous=NORMAL` | 65 236 | 2 828 |

Las tres últimas son el control: no prometen durabilidad, y que pierdan muestra que el corte llega hasta lo no escrito.

**etcd** no entra en esa corrida: bbolt mapea su archivo en memoria y falla sobre el montaje FUSE de LazyFS («unexpected
fault address»). Se midió con reinicios reales: `scripts/sysrq/etcd.sh`, la misma carga y `echo b >
/proc/sysrq-trigger` en Debian 13 arm64 (AWS t4g.small, disco EBS), 27/10/2026:

| Motor | Reinicios | Escrituras confirmadas | Perdidas |
|---|---|---|---|
| etcd 3.5.17 (`fsync` por commit, lo normal) | 20 | 36 077 | 0 |
| etcd 3.5.17, `--unsafe-no-fsync` (control) | 10 | 16 447 | 8 599 |

El control también vale para los 50 reinicios de Luma en arm64 de arriba: en esa máquina, un reinicio real sí borra
lo que no se sincronizó.
