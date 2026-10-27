# SPEC · Luma en producción para Friochap: no se cae y no pierde nada

Fecha: 26/10/2026. Destino: la VM de Friochap en OCI (Ampere A1, arm64, Debian 13, ext4), junto al ERP, que sale el
**1/12/2026**. Un ERP no admite «casi siempre»: este SPEC convierte las garantías de `docs/operar/PROD_READINESS.md` en
garantías **comprobadas en esa máquina y con esa carga**, y cierra lo que falta para que Luma no se caiga.

## Qué hace Luma para Friochap (y qué no)

| Uso | Primitiva | Qué pasa si se pierde un dato confirmado |
|---|---|---|
| Caché de respuestas de IA | KV con TTL | se vuelve a pagar la pregunta |
| Cola de lecturas de planos | colas + blobs | una lectura pagada no llega a la persona |
| Buscador del ERP (PLT-07 de Friochap) | hub: texto + vectores | un documento no aparece al buscar hasta reindexar |

Los datos contables y de operación (facturas, asientos, kardex, planillas) **no** van a Luma: viven en el SQLite del ERP,
porque una factura, su asiento y su salida de almacén se confirman en **una sola transacción** dentro del proceso del
ERP. Luma está en otro proceso y no puede ser parte de esa transacción (no hay transacción distribuida entre los dos).
Esa es la razón de la separación, no una desconfianza en Luma.

## Estado de partida (lo que ya está y no se rehace)

- `wal_sync_mode = "per_write"` por defecto: fsync antes de confirmar, con group commit (≈7 % de costo).
- Blobs y colas: fsync del archivo y del directorio. SQLite interno con `synchronous = FULL`.
- Matriz de crash-recovery (mata el proceso durante ráfagas de cada motor) en CI y nightly.
- Respaldo completo cifrado (`backup_*`, también a S3/R2) y envío continuo del WAL (`wal_ship_interval_secs`).
- Métricas para Prometheus y alertas (`docs/observability/`).
- Historial reciente que obliga a comprobar en vez de suponer: el 23/08 aparecieron 11 bugs bajo pruebas que miraban el
  sitio equivocado, dos de ellos con pérdida silenciosa de datos (`c184423`: tras un arranque sin escrituras, lo nuevo
  se perdía; `6b0ddfe`: segmentos de WAL vacíos que nunca se borraban).

## Estado al 26/10/2026

| Criterio | Estado | Evidencia |
|---|---|---|
| CA-1 · 1 000 cortes con LazyFS | en curso, 0 pérdidas hasta ahora | `scripts/apagon/` (100 por noche en `nightly-resilience.yml`) |
| CA-2 · 50 `sysrq` en arm64 | **falta**: necesita una VM arm64 desechable | — |
| CA-3 · disco lleno | cumplido | `tests/disk_full.rs` en ext4 de 64 MB (`scripts/disk_full.sh`): 507, lecturas 200, recuperación sola, todo sobrevive al reinicio |
| CA-4 · EIO en fsync | cumplido | `tests/fsync_eio.rs` con `dm-error` (`scripts/fsync_eio.sh`): falla la escritura, solo lectura en `/v1/health/ready` aunque el disco vuelva |
| CA-5 · watchdog | cumplido | `scripts/watchdog_check.py`: `READY` tras el bind, latidos que se cortan con el proceso congelado, `STOPPING` y salida en 1 s |
| CA-6 · RSS y `MemoryMax` | **falta la carga real de Friochap**; pico con la carga sintética anotado por `scripts/apagon/dr.sh` | — |
| CA-7 · actualizar con carga | cumplido | v4.26.0 → nueva bajo carga, 10 ciclos sin pérdidas; `tests/golden_data_dir.rs` lee un `data_dir` grabado por v4.26.0 |
| CA-8 · otra máquina desde el remoto | cumplido con MinIO; falta repetirlo contra R2 en la VM | `scripts/dr_drill.sh`: 0 fallas, todo lo confirmado hasta 2,5–4,4 s antes del corte está (promesa: 10 s); RTO 6–12 s |
| CA-9 · rodaje de 72 h | **falta** (semana del 16/11) | — |

Corregido en el camino, con pérdida o corrupción de datos de por medio:
- Escrituras confirmadas que se perdían porque deletes, lotes y vectores se adelantaban en el WAL a la cola de group
  commit. CAS no atómico, ABA en índices de documentos y `LMOVE`, INCR/APPEND/SET NX sin atomicidad (`tests/acid_concurrency.rs`).
- Los respaldos perdían toda clave anterior al segmento de WAL más viejo que quedaba: no llevaban la proyección KV. Ahora
  llevan `state.export.jsonl`, leído de un instante consistente antes de copiar el WAL.
- El cercado por epoch no cercaba: `promote` la reclamaba en un prefijo y el envío la leía en otro.
- Blobs y colas solo volvían del último respaldo completo (horas): ahora se envían cada `wal_ship_interval_secs` con el
  WAL, y `luma restore-remote` arma la máquina nueva.
- Colección de documentos con `:` chocaba con otra colección (y otra organización); el índice de documentos se
  reconstruye al arrancar sobre datos de una versión anterior.
- Lease de colas sin fsync sobre el mensaje, acks y borrados no durables, log de `/search` sin fsync.
- `limit` del buscador del hub sin tope; `POST /v1/admin/backup` cortado a los 30 s con un respaldo grande.

**Para Friochap:** el ERP habla con Luma desde 127.0.0.1 y el límite por defecto es **100 peticiones/s por IP**
(`RATE_LIMIT_RPS`); en la VM se sube o se apaga (`0`), o el ERP recibirá 429. La base SQLite de Luma (cuentas, auditoría,
tablas del hub) no se envía cada 10 s: vuelve del último respaldo completo; el buscador se reindexa desde el ERP.

## Alcance

### F1 · Corte de luz, no solo `kill -9` (crítico)
La matriz actual mata el **proceso**: el kernel sigue vivo y lo que está en la caché de páginas llega igual al disco. Un
corte de luz pierde esa caché. Se agrega una matriz que la pierde de verdad:
- **LazyFS** (sistema de archivos FUSE que guarda en memoria todo lo no sincronizado y lo descarta a pedido) bajo el
  `data_dir`, en Linux, en CI nightly: ráfagas de escrituras KV, colas, blobs y vectores → `lazyfs::clear-cache` →
  reinicio → **todo lo que recibió OK sigue ahí y nada está corrupto**.
- Una vez antes del 1/12, en una VM arm64 desechable: carga + `echo b > /proc/sysrq-trigger` (reinicio sin vaciar
  discos), 50 ciclos.

### F2 · Disco lleno y disco con errores
- Con el disco lleno (ENOSPC): Luma **rechaza** la escritura con un error claro (507) y **sigue sirviendo lecturas**; no
  corrompe el WAL ni se cae; al liberar espacio, vuelve sola a aceptar escrituras.
- Un fsync que falla (EIO) no se reintenta en silencio: la escritura se responde como fallida y Luma pasa a solo lectura
  hasta reiniciarse (el fsync fallido de Linux puede haber perdido páginas: reintentar daría un OK mentiroso).
- Prueba con un sistema de archivos pequeño (loop de 64 MB) y con `dm-error`/LazyFS para EIO.

### F3 · Supervisión: que systemd sepa si Luma está viva de verdad
- `sd_notify`: `READY=1` cuando terminó el replay y acepta peticiones (unidad `Type=notify`), y `WATCHDOG=1` desde el
  bucle que atiende peticiones, no desde un hilo aparte (si el atendedor se traba, el latido se corta).
  `WatchdogSec=30`: si se traba, systemd la reinicia.
- `GET /v1/health/ready`: 200 solo si el WAL se puede escribir, hay más de un 10 % de disco libre y el replay terminó;
  si no, 503 con el motivo. Lo consulta el vigilante externo de Friochap (INF-04).

### F4 · Memoria acotada
- Con la carga de Friochap (PLT-06 del ERP + reindexado completo del buscador), la RSS pico medida + 50 % fija el
  `MemoryMax` de la unidad. El reindexado y la ingesta van por lotes con tope, para que nunca crezcan con el tamaño de
  los datos.
- Ningún camino de una petición asigna sin tope según lo que manda el cliente (cuerpos, `limit`, lotes).

### F5 · Actualizar sin perder ni cortar
- Apagado ordenado con SIGTERM: deja de aceptar, termina lo que está en curso (tope 10 s), sella el WAL, sale.
- Una versión nueva arranca sobre los datos de la anterior (`tests/golden_data_dir.rs` extendido con un `data_dir` de la
  versión que esté en producción) y, si el formato cambia, migra o se niega a arrancar con un mensaje claro; nunca
  arranca vacía.
- Binario fijado por versión en el despliegue de Friochap; se actualiza a propósito, con `luma backup` antes.

### F6 · Respaldo y recuperación ante desastre activados
- En la VM de Friochap: `backup_remote_url` a R2 (cifrado), respaldo completo diario y `wal_ship_interval_secs = 10`
  (**RPO de 10 s**). La prueba de restauración semanal del ERP (INF-02) restaura también Luma en una carpeta temporal y
  comprueba conteos de colas, blobs y claves.
- Procedimiento escrito y probado: VM perdida → VM nueva → `luma restore` desde R2 → arranca (RTO medido).

### F7 · Rodaje de 72 horas
- Antes del 1/12, 72 h seguidas en una A1 con la carga realista del ERP más reindexados, actualizaciones de Luma en
  caliente cada 12 h y un `kill -9` cada 2 h: cero pérdidas, cero errores 5xx no esperados, memoria estable (sin
  crecimiento sostenido).

## Criterios de aceptación
- **CA-1.** 1 000 ciclos de la matriz con LazyFS sin una sola escritura confirmada perdida ni corrupción (F1).
- **CA-2.** 50 reinicios por `sysrq` en arm64 con el mismo resultado (F1).
- **CA-3.** Disco lleno: escrituras con 507, lecturas bien, recuperación sola al liberar espacio, WAL íntegro (F2).
- **CA-4.** EIO en fsync: la escritura falla, Luma queda en solo lectura y lo informa en `/v1/health/ready` (F2).
- **CA-5.** Con el atendedor trabado a propósito, systemd la reinicia en menos de 45 s (F3).
- **CA-6.** RSS pico con la carga de Friochap medida y `MemoryMax` fijado con su margen; sin muertes por memoria en F7.
- **CA-7.** Actualizar de la versión en producción a la nueva con carga encima: ninguna petición perdida del lado del
  ERP (reintenta) y ningún dato perdido (F5).
- **CA-8.** Restauración en otra máquina desde R2 con pérdida máxima de 10 s de escrituras; RTO anotado (F6).
- **CA-9.** Rodaje de 72 h cumplido (F7).

## Fuera de alcance
Alta disponibilidad con dos VM y conmutación automática (`failover.rs`, réplica de lectura W2.2): una sola VM es la
decisión de costo de Friochap. Si la VM entera cae, caen ERP y Luma juntos y se recuperan con F6 y el Litestream del ERP.
Se reabre si Friochap pide disponibilidad por encima de lo que da una VM.

## Orden
F3 y F2 primero (son cambios de código pequeños y protegen de lo más probable), luego F1, F4 y F5, y F6 con la VM de
Friochap (2–8 nov). F7 la semana del 16/11.
