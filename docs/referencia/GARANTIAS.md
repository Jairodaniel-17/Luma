# Garantías por motor

Qué promete Luma en cada motor sobre atomicidad, aislamiento, consistencia y durabilidad, con qué motor conocido se
compara y qué prueba lo sostiene. Lo que no está en esta tabla **no** se promete.

Durabilidad común a todos: con `wal_sync_mode = "per_write"` (el valor por defecto), una respuesta 200 significa que
el dato ya pasó por `fsync`. Si un `fsync` falla, Luma no reintenta (en Linux podría dar un OK mentiroso): rechaza la
escritura y queda en solo lectura hasta reiniciar. Con el disco casi lleno rechaza con 507 y sigue leyendo.

| Motor | Atomicidad | Aislamiento | Se compara con |
|---|---|---|---|
| KV (`/v1/state`) | Cada operación; un lote (`/v1/state/batch_put`) es **un solo registro** del WAL: todo o nada | Por clave, linealizable. Un lote valida la revisión de cada clave (`if_revision`) antes de escribir nada; `if_revision: 0` = crear solo si no existe | Redis con `WATCH`/`MULTI`; etcd con `txn` de comparación |
| Documentos (`/v1/doc`) | El documento y sus entradas de índice se escriben en el mismo lote | Por documento; un `find` nunca ve un índice que no corresponde al documento | MongoDB (un documento) |
| Estructuras RESP (listas, hashes, sets, zsets) | Cada comando, también los de varias claves (`LMOVE`, `RENAME`, `MSET`) | `MULTI`/`EXEC` aislado **frente a otros clientes RESP**; no frente a escrituras por HTTP en las mismas claves | Redis 7 (diferencial de 327 comandos en CI) |
| Colas (`/v1/queue`) | Encolar y confirmar (ack) son atómicos y durables | Al menos una vez: un mensaje recibido y no confirmado vuelve a ser visible al vencer su lease | Amazon SQS estándar |
| Blobs (`/v1/blob`, S3) | Escritura a temporal + `fsync` + rename + `fsync` del directorio: se ve el objeto entero o el anterior, nunca uno a medias | Lectura tras escritura | Amazon S3 |
| Vectores (`/v1/vector`) | Cada upsert/borrado, y cada lote | Por id | Qdrant / pgvector (sin transacciones) |

## Lo que no hay, a propósito

- **Transacciones interactivas entre motores.** No existe un `BEGIN … COMMIT` que abarque una clave, un blob y un
  mensaje. Quien lo necesita (una factura, su asiento y su salida de almacén) lo hace en una base relacional; así lo
  hace el ERP de Friochap con su SQLite (`docs/SPEC-friochap.md`).
- **Aislamiento serializable sobre lecturas.** Un lote valida las claves que escribe, no las que solo leyó: dos
  lotes que leen A y B y escriben cada uno en una distinta pueden pasar los dos (write skew), como en el aislamiento
  por instantánea de PostgreSQL. Para impedirlo, el lote reescribe también la clave leída con su `if_revision`.
- **Lecturas de una instantánea de varias claves.** Cada lectura ve lo último confirmado de su clave.

## Qué lo sostiene

| Garantía | Prueba |
|---|---|
| Ninguna escritura confirmada se pierde ni se adelanta en el WAL bajo concurrencia | `tests/acid_concurrency.rs`, también en arm64 (`test-arm64` en CI) |
| CAS, lotes y índices de documentos sin carreras ni ABA | `tests/acid_concurrency.rs` |
| Comandos RESP atómicos, `EXEC` aislado | tests de `src/resp/commands.rs`; diferencial contra Redis 7 |
| Corte de luz (se pierde la caché de páginas) | `scripts/apagon/` con LazyFS: 1 115 cortes sin pérdidas; 200 cada noche |
| Reinicio real sin sincronizar, arm64 | `scripts/sysrq/run.sh`: `echo b > /proc/sysrq-trigger` en una VM Debian 13 arm64 |
| `kill -9` en cada camino de escritura | `tests/crash_recovery.rs` (nightly) |
| Disco lleno y `fsync` con EIO | `tests/disk_full.rs`, `tests/fsync_eio.rs` |
| Host perdido: lo confirmado hasta 10 s antes está en la máquina nueva | `scripts/dr_drill.sh` |
| Una versión nueva lee lo que escribió la anterior | `tests/golden_data_dir.rs` |
