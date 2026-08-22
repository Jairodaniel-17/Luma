# PLAN MAESTRO DE EJECUCIÓN — post v4.24.0

Plan único que unifica los tres SPEC existentes en **una sola ruta lineal de
bloques ejecutables**. No sustituye a los SPEC: los ordena, deduplica sus
solapes y resuelve sus conflictos de numeración.

- [`SPEC-producto.md`](SPEC-producto.md) — frentes W1–W5 (durabilidad, réplica, protocolos, CDC, operabilidad).
- [`SPEC-resp.md`](SPEC-resp.md) — fases F0–F4 (compatibilidad con el protocolo Redis).
- [`SPEC-roadmap.md`](SPEC-roadmap.md) — milestones M1–M4 (8/12 hechos; quedan 4 ítems).

Convención de estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho.

---

## Reglas de proceso (aplican a todo el plan)

1. **Commit por entrega.** Cada ítem terminado se commitea de inmediato. No se
   acumulan ítems sin commitear.
2. **Los tests se corren en lote, en el cierre de bloque.** No hay `cargo test`
   por ítem: cada bloque termina con una **puerta de verificación** única
   (`cargo fmt --all -- --check` → `cargo clippy --all-targets --all-features -D warnings`
   → `cargo test`). Los ítems dentro del bloque se commitean con la
   verificación pendiente y el cierre del bloque es lo que la resuelve.
3. **Los benchmarks están desactivados.** No se ejecuta `cargo bench` en el
   ciclo de trabajo. Los dos ítems de los SPEC que son puramente de medición
   quedan **aparcados** (ver *Ítems aparcados*). El job `bench-compile` de CI
   (`cargo bench --no-run`) se mantiene: solo compila, no mide, y protege de
   romper los benches sin darse cuenta.
4. **Compatibilidad de datos hacia adelante es no negociable.** Toda versión N
   lee lo escrito por N−1 (política completa en `SPEC-producto.md`). Registros
   nuevos siempre versionados y etiquetados; nunca reinterpretar lo viejo.

---

## Estado real del código (verificado en v4.24.0, no asumido)

Lo que cambia el tamaño de varios ítems respecto a lo que suponen los SPEC:

| Hallazgo | Consecuencia |
|---|---|
| **No existe nada de RESP** — ni `resp_port` en `config.rs`, ni módulo, ni dependencia | F0–F4 es greenfield completo. Sin deuda que desmontar tampoco. |
| El valor del KV es `serde_json::Value` puro (`state.rs:45`, `state.rs:489`) | F0.1 (`StoredVal`) es trabajo real y toca WAL + redb + snapshot + API HTTP. Es el ítem con más riesgo de compatibilidad de todo el plan. |
| No existe `src/engine/structures.rs` | F0.2 es greenfield: las 4 estructuras completas. El ítem más grande del plan. |
| **`Metrics::render_prometheus()` ya existe** (`engine/metrics.rs:63`) y `/v1/metrics` sirve `metrics_text()` | W5.1 está a medias, no en cero. Falta el `content-type` correcto, histogramas por motor y el dashboard. Baja de esfuerzo BAJO a *muy* bajo. |
| `vector::persist::Manifest` ya tiene `version: u32` y campos `#[serde(default)]` | M3.2 (modelo/dim por colección) es casi gratis: campo nuevo con `serde(default)`, sin migración. |
| `rustls` + `tokio-rustls` ya son dependencias | F4.1 (TLS en el listener RESP) reusa la config TLS existente. Esfuerzo real bajo. |
| **No existe `object_store` ni ninguna dep S3** | W1.3 y W2.1 arrastran una dependencia nueva. Pasa por `deny.toml`. |
| No hay `#![forbid(unsafe_code)]` en `lib.rs` ni `main.rs` | W5.3 necesita inventario previo de `unsafe`, no solo añadir el atributo. |
| `wal_sync_mode` existe y se usa (`sync_data` / `sync_all` en `persist.rs`) | W1.2 es auditoría y documentación, no implementación desde cero. |
| M1.2 y M2.1–M2.3 ya están en el router (`sessions/revoke-all`, `my-orgs`, `switch-org`) | Confirmado hecho. El pendiente real del roadmap viejo son 4 ítems. |

---

## Conflictos entre los SPEC, resueltos

**1. Los tres SPEC reclaman `v4.25.0`.** SPEC-roadmap lo asigna a M4.1,
SPEC-resp a la Fase 0, SPEC-producto a W1. Se renumera todo linealmente en
este plan (columna *Release* de la tabla de bloques). **La numeración de este
documento manda**; las de los SPEC quedan obsoletas.

**2. El harness de crash-recovery está especificado dos veces.** W1.1
(SPEC-producto) y el crash-test de F0.2 (SPEC-resp) son el mismo harness.
Se construye **una vez** en el Bloque 1 y F0.2 lo consume. Ahorro real: el
ítem más caro de F0 se reduce a añadir sus tipos de registro a un harness ya
existente.

**3. OpenAPI aparece dos veces con distinto alcance.** M4.1 es actualizar el
YAML a mano para los endpoints nuevos; W3.3 es generarlo desde el código con
CI que falla si divergen. No son alternativas: M4.1 tapa el agujero de hoy
(Bloque 0, barato) y W3.3 elimina la clase de problema (Bloque 9).

**4. Cuotas por tenant están duplicadas.** B.1 del SPEC-roadmap y W5.2 del
SPEC-producto son el mismo ítem. Se ejecuta una vez, como W5.2.

**5. Réplica/HA duplicada.** B.3 del SPEC-roadmap ≡ W2 del SPEC-producto.
Se ejecuta como W2.

---

## Desviaciones deliberadas del orden de los SPEC

Dos cambios de orden respecto a lo que dicen los SPEC, con motivo:

**D-1 — F4.1 (TLS y límites del listener) se adelanta de la Fase 4 al Bloque 4,
junto con el listener mismo.** SPEC-resp deja los límites de conexión,
`max_clients`, timeouts de idle y backpressure para el final. Exponer un
listener TCP nuevo sin límites de conexión ni timeouts es abrir un vector de
DoS y luego cerrarlo cuatro releases después. Los límites entran con el
listener o el listener no entra.

**D-2 — F4.3 (docs y matriz de compatibilidad) se adelanta al Bloque 6, junto
con el E2E de Celery/arq.** El criterio de éxito global del SPEC-resp es que
alguien apunte su Celery a Luma sin tocar código; eso no se cumple sin el
documento que explica cómo. La doc es parte de la entrega, no un epílogo.

---

## Mapa de bloques

Cada bloque = una release + una puerta de verificación. La ruta es **lineal a
propósito**: el riesgo declarado nº 1 del SPEC-producto es abrir los cinco
frentes a la vez y no cerrar ninguno.

| # | Bloque | Release | Contenido | Tamaño |
|---:|---|---|---|---|
| 0 | ✅ Barrido de deuda del roadmap | `v4.25.0` | M3.2, M3.4, M4.1, M3.3 + fix de `content-type` en métricas | S |
| 1 | ✅ **Durabilidad demostrada** | `v4.26.0` | W1.1 (harness compartido), W1.2, fixture dorado en CI | M |
| 2 | ✅ Cimientos del motor RESP | `v4.27.0` | F0.1 `StoredVal`, F0.2 estructuras, F0.3 notificadores | **XL** |
| 3 | ✅ RPO prometible | `v4.28.0` | W1.3 backup remoto, W1.4 verificación, W2.1 WAL shipping | L |
| 4 | ✅ Listener RESP + strings/keys | `v4.29.0` | F1.1–F1.4 + **F4.1** (ver D-1) | L |
| 5 | ✅ Estructuras por RESP | `v4.30.0` | F2.1 listas, F2.2 hashes, F2.3 sets, F2.4 zsets | L |
| 6 | **Celery y arq funcionando** | `v4.31.0` | F3.1 bloqueantes, F3.2 MULTI/WATCH, F3.3 Pub/Sub, F3.4 E2E + **F4.3** (ver D-2) | L |
| 7 | Operar con vista | `v4.32.0` | W5.1 métricas/OTel, W2.2 réplica de lectura | L |
| 8 | RESP endurecido | `v4.33.0` | F4.2 backup+panel, F4.5 nightly de resiliencia | M |
| 9 | Adopción por protocolo | `v4.34.0` | W2.3 failover asistido, W3.2 S3, W3.3 OpenAPI generado | **XL** |
| 10 | Conector Postgres | `v4.35.0` | W4.1 spike, W4.2 `luma connect postgres`, W4.3 federada | **XL** |
| 11 | **GA** | `v4.36.0` → `1.0` | W5.2 cuotas, W5.3 supply chain, W5.5 docs de producto, W5.6 criterio GA | L |

Tamaños: S ≈ días · M ≈ 1–2 semanas · L ≈ 3–5 semanas · XL ≈ 6+ semanas, a
dedicación parcial. **El plan completo es trabajo de varios meses.** Los
bloques 0–6 son la ruta que produce el salto de adopción; 7–11 son lo que
convierte eso en producto operable.

### Hitos con significado externo

- **Fin de Bloque 1** — se puede afirmar "no pierde datos confirmados" con un
  harness que lo demuestra, no con una afirmación en el README.
- **Fin de Bloque 3** — se puede prometer un RPO concreto. Es el primer punto
  donde tiene sentido poner datos reales de terceros.
- **Fin de Bloque 6** — `REDIS_URL=redis://luma:6379/0` funciona con Celery y
  arq sin tocar código del cliente. Criterio de éxito global de SPEC-resp.
- **Fin de Bloque 11** — GA 1.0, API v1 congelada.

---

## Detalle por bloque

Cada ítem conserva su identificador del SPEC de origen para poder cruzarlo.
Los criterios de aceptación completos viven en el SPEC; aquí va el enfoque y lo
que cierra el ítem.

### Bloque 0 — Barrido de deuda del roadmap · `v4.25.0`

Cierra los 4 ítems que llevan pendientes desde `SPEC-roadmap.md`. Todos son
baratos y ninguno depende de los bloques siguientes: es el arranque de menor
riesgo y deja el roadmap viejo a cero.

- `[x]` **M3.2 — modelo/dim por colección + validar mismatch.** Añadir
  `embedding_model: Option<String>` y `embedding_dim: Option<usize>` a
  `vector::persist::Manifest` con `#[serde(default)]` (el struct ya tiene
  `version` y ese patrón, así que no hay migración). En el ingest por texto,
  si el dim del cliente activo ≠ el de la colección → 400 explicativo en vez
  de escribir un vector corrupto.
- `[x]` **M3.4 — hot-reload de embeddings/LLM.** `EmbeddingClient` detrás de
  `ArcSwap`; `PUT /v1/config` reconstruye el cliente sin reiniciar.
  Verificable con `POST /v1/config/embedding/probe` mostrando el dim nuevo.
- `[x]` **M4.1 — SDKs + OpenAPI al día.** Los endpoints que existen en el
  router pero no en los SDKs ni en el YAML: `my-orgs`, `switch-org`,
  `sessions`, `sessions/revoke-all`, `DELETE /v1/vector/:collection`,
  `config/embedding/probe`, `meta/:collection/execute`, invitaciones y
  miembros por org, `users/:id/orgs`, CRUD de roles.
- `[x]` **M3.3 — reindexado al cambiar de modelo.** El único medium-high del
  bloque: `POST /v1/vector/:col/reindex {target_model}` como job en background
  con progreso. **Si el bloque se alarga, este es el ítem que se mueve al
  Bloque 1**, no los otros tres.
- `[x]` **Fix — `content-type` de `/v1/metrics`.** El handler
  (`routes_state.rs:52`) devuelve el texto sin cabecera; Prometheus espera
  `text/plain; version=0.0.4`. Dos líneas, y es prerequisito de W5.1.

**Puerta de verificación 0:** fmt + clippy + `cargo test`.

> **Bloque 0 cerrado.** Puerta verde: fmt, `clippy --all-targets
> --all-features` con 0 findings, **260 tests**, y `tsc --noEmit` del SDK TS
> limpio.
>
> Corrida parcial previa tras M3.2 / M3.4 / metrics (el refactor a `EmbeddingHandle`
> tocó 25 ficheros de test, así que se adelantó una pasada): fmt y
> `clippy --all-targets --all-features` limpios, **257 tests verdes**. Destapó
> un bug preexistente de Windows en `write_manifest` — ver W1.2 del Bloque 1,
> que ya tiene su primera fila de tabla en `PROD_READINESS.md`.

### Bloque 1 — Durabilidad demostrada · `v4.26.0`

El bloque del que dependen todos los demás. Construye el harness que F0.2
también necesita (ver conflicto 2).

- `[x]` **W1.1 — matriz de crash-recovery.** Harness que mata el proceso en
  puntos aleatorios durante ráfagas de escritura de **cada** motor (KV, blob,
  colas, doc, vector, memoria, SQLite) y verifica al reinicio: prefijo
  confirmado íntegro, cero divergencia memoria↔redb. Diseñarlo **desde el
  inicio con tipos de registro extensibles**, porque el Bloque 2 le añade los
  de estructuras. Job nightly (no en el ciclo normal, por la regla 2).
- `[x]` **W1.2 — fsync por motor auditado.** Auditar la política extremo a
  extremo. Foco en los dos puntos débiles conocidos: las colas escriben JSON
  por mensaje (¿fsync del archivo *y* del directorio?) y blob (write + rename
  atómico + fsync). Entrega una tabla en `PROD_READINESS.md`: primitiva → qué
  garantiza cuando devuelve OK.
- `[x]` **Fixture dorado en CI** (política de compatibilidad, punto 4). Un
  `data_dir` generado por la última release publicada, versionado; si la rama
  no lo lee íntegro, el build falla. Se regenera por release. Sin esto, la
  regla 4 es una intención y no un control.

**Puerta de verificación 1:** fmt + clippy + `cargo test` + primera corrida
verde del harness.

### Bloque 2 — Cimientos del motor RESP · `v4.27.0` · **el bloque más grande**

Sin RESP todavía. Todo lo de aquí sirve también a la API HTTP.

- `[x]` **F0.1 — `StoredVal { Json, Raw }`.** El ítem de mayor riesgo de
  compatibilidad del plan: toca `state.rs`, `state_db.rs`, `persist.rs` y la
  API HTTP (expone `Raw` como base64 + `content_type`). Registro WAL nuevo
  versionado; **los registros legados se leen como `Json` y no se
  reinterpretan nunca**. Property test: cualquier `Vec<u8>` sobrevive
  put→crash→replay→get idéntico.
- `[x]` **F0.2 — motor de estructuras** (`engine/structures.rs`): list, hash,
  set, zset con semántica Redis, durables por el mismo WAL + redb. Decisiones
  que se toman aquí y no se parchean después:
  - zset con `BTreeMap<(score, member)>` — el orden lexicográfico en empates
    de score es contrato y se testea;
  - **contador de revisión por clave**, que F3.2 (`WATCH`) necesita;
  - una clave, un tipo — operar con el tipo equivocado da error tipado (será
    `-WRONGTYPE`);
  - límites `MAX_STRUCTURE_ENTRIES` / `MAX_MEMBER_LEN` configurables.
- `[x]` **F0.3 — notificadores por clave.** `tokio::sync::Notify` por clave
  para los bloqueantes de F3.1. Un `LPUSH` despierta exactamente un `BLPOP`
  (sin thundering herd); limpieza al quedar sin waiters.

**Puerta de verificación 2 — verde.** 338 tests, clippy 0 findings, fixture dorado incluido.

**Puerta de verificación 2:** fmt + clippy + `cargo test` + harness del Bloque 1
extendido a los registros nuevos.

### Bloque 3 — RPO prometible · `v4.28.0`

- `[x]` **W1.3 — backup remoto** a destino S3-compatible con el crate
  `object_store` (S3/R2/GCS/MinIO con una API). Artefacto cifrado con la master
  key; `luma restore s3://…` directo; retención remota. Dependencia nueva →
  pasa por `deny.toml`.
- `[x]` **W1.4 — `luma backup --verify`.** Restaura a un temporal, corre
  checks, reporta. `backup_last_verified_ts` en métricas + regla Prometheus de
  ejemplo. Un backup no verificado no cuenta como backup.
- `[x]` **W2.1 — WAL shipping continuo** al bucket (estilo Litestream).
  Snapshot + cadena de segmentos = punto en el tiempo;
  `luma restore --to-timestamp`. **Requiere un spike previo** para decidir qué
  hacer con el WAL de SQLite (¿shipping directo o checkpoint + copia?);
  ese spike es la primera tarea del bloque, no un detalle de implementación.
  Métrica `wal_ship_lag_seconds`.

**Puerta de verificación 3:** fmt + clippy + `cargo test` + demo de
destrucción/restauración con pérdida ≤ `ship_interval`.

### Bloque 4 — Listener RESP + strings/keys · `v4.29.0`

El hito visible: `redis-cli -p 6379` conversa con Luma.

- `[x]` **F1.1 — framing RESP2 + ciclo de conexión.** Listener propio en
  `resp_port` (default **desactivado**), mismo proceso. Decisión pendiente de
  D2: evaluar el crate `redis-protocol` contra `deny.toml` primero; si no
  pasa, parser propio. Soportar *inline commands* (`PING\r\n`), porque
  redis-cli los usa. Pipelining. Fuzzing del parser sin panics.
- `[x]` **F4.1 - limites y TLS** (adelantado, ver D-1): `resp_max_clients`,
  `resp_idle_timeout_secs`, buffer maximo por conexion, backpressure en
  suscriptores lentos, y **TLS** reusando `rustls` (`resp_tls_enabled`, con
  certificado propio o el de HTTP).

  TLS dejo de ser higiene en cuanto `AUTH` empezo a llevar la api key de una
  organizacion: en claro, cualquiera en el camino tiene una credencial que liga
  la conexion a todo el keyspace de esa org. Encenderlo sin certificado
  **impide arrancar** en vez de servir en claro por un fichero que falta.
- `[x]` **F1.2 — AUTH multi-tenant.** `AUTH` mapea a las api keys/roles
  actuales; keyspace prefijado por `{org_id}\x1f`; `KEYS`/`SCAN` filtran por
  tenant. Dos orgs con la misma clave no se ven. Key revocada corta en el
  siguiente comando.
  **Corrección de contabilidad (Bloque 8):** esto estuvo marcado `[x]` sin
  estarlo. El plumbing existía en la capa de comandos y sus tests lo ejercitaban
  poniendo `session.tenant` a mano, pero el listener autenticaba solo contra la
  clave estática y devolvía `tenant: None` — toda conexión RESP compartía un
  keyspace plano y la api key de una org no servía por el protocolo. Cerrado de
  verdad al abrir el Bloque 8, con 9 tests sobre socket real.
- `[x]` **F1.3 — comandos de strings/keys** El set completo del SPEC
  (`GET SET SETEX … SCAN RENAME`), con `FLUSHDB` solo bajo
  `resp_allow_flush = true`. **La suite diferencial contra Redis 7 real en
  docker es la fuente de verdad**, no la documentación de Redis: ~200
  operaciones, salidas byte-idénticas. `SCAN` con cursor real, no snapshot.
- `[x]` **F1.4 — observabilidad RESP.** `resp_connections_gauge`,
  `resp_commands_total{cmd}`, `resp_errors_total{kind}`,
  `resp_auth_failures_total`. `INFO` con secciones reales — kombu las lee.

**Puerta de verificación 4:** fmt + clippy + `cargo test` + suite diferencial +
smoke de redis-py.

### Bloque 5 — Estructuras por RESP · `v4.30.0`

Exponer por protocolo lo que el Bloque 2 ya implementó. Riesgo bajo de diseño,
alto de detalle.

Cobertura **medida** contra las listas de comandos de `SPEC-resp.md`, no
estimada: **57/57**.

- `[x]` **F2.1 listas** — 14/14.
- `[x]` **F2.2 hashes** — 12/12.
- `[x]` **F2.3 sets** — 8/8.
- `[x]` **F2.4 sorted sets** — 17/17, con los modificadores `NX`/`XX`/`GT`/
  `LT`/`CH`/`INCR` de `ZADD` y las opciones `REV`/`BYSCORE`/`BYLEX`/`LIMIT`/
  `WITHSCORES` de `ZRANGE`.

F2.1 estuvo marcado `[x]` con 9/14 comandos. El recuento se obtiene comparando
las ramas del dispatcher contra el SPEC — no los literales del fichero, que
daban un falso 57/57 porque `LPUSHX`/`RPUSHX` aparecian en `pushed_list_keys`
sin estar implementados.

Las dos trampas que la aceptación tiene que cazar explícitamente: **nil vs
array vacío** y **`-WRONGTYPE` cruzando tipos**. Ahí es donde los clientes
rompen en silencio. Los patrones reales de la matriz de clientes (kombu unacked
con `HSET`+`ZADD`, arq con `ZRANGEBYSCORE`+`ZREM`) entran como tests con frames
capturados de los clientes de verdad, no inventados.

**Puerta de verificación 5:** fmt + clippy + `cargo test` + suite diferencial
ampliada a estructuras.

### Bloque 6 — Celery y arq funcionando · `v4.31.0` · **el hito de adopción**

- `[x]` **F3.1 — bloqueantes** (BLPOP BRPOP BLMOVE BRPOPLPUSH BZPOPMIN BZPOPMAX):
  BZPOPMAX`, multi-clave con orden de argumentos como contrato. Cierre de
  conexión con waiters pendientes no filtra memoria (test con 1k conexiones).
- `[x]` **F3.2 — `MULTI/EXEC/DISCARD/WATCH/UNWATCH`.** `WATCH` usa la
  `revision` del `StateStore` y el contador por clave que el Bloque 2 dejó en
  las estructuras. Test de carrera: 100 clientes incrementando la misma clave
  → suma exacta.
- `[x]` **F3.3 — Pub/Sub** sobre el `EventBus` existente, canal interno
  `resp:{org}:{canal}`. `PUBLISH` devuelve receptores **del tenant**, no
  globales.
- `[x]` **F3.4 - E2E de los clientes objetivo.** `tests/e2e/clients.py` corre
  redis-py, kombu, Celery y arq reales contra Luma y contra un Redis 7 de
  control, con un **worker Celery de verdad** que consume, ejecuta y devuelve
  el resultado, y con el escenario de **matar el worker a media tarea**: el
  mensaje sigue en `unacked` o de vuelta en la cola, nunca en ninguno de los
  dos. Job `client-e2e` en CI.

  Encontro en su primera ejecucion que `PUBLISH` dentro de `MULTI` se
  rechazaba como comando desconocido: redis-py envuelve todo pipeline en
  MULTI/EXEC y el backend de resultados de Celery escribe con `SETEX` +
  `PUBLISH`, asi que el worker ejecutaba la tarea y el llamante esperaba para
  siempre. Ni los tests unitarios ni el corpus diferencial cubrian una
  transaccion.

  El escenario de unacked se hizo **despues** de cerrar la ventana de
  atomicidad multiclave, que es lo que lo ponia en duda.
- `[x]` **F4.3 — `docs/RESP.md`** (adelantado, ver D-2): tabla de comandos con
  notas de divergencia, guía "migrar de Redis a Luma en 5 minutos" para Celery,
  arq, redis-py e ioredis, y qué NO se soporta y por qué. README enlaza.

**Puerta de verificación 6:** fmt + clippy + `cargo test` + los dos E2E verdes.
**Antes de cerrar el bloque, validar la matriz de comandos contra las versiones
fijadas de cada cliente** — `SPEC-resp.md` advierte que arq puede exigir
`SCRIPT/EVAL` según versión (backlog B-R.3). Si aparece, es una decisión de
alcance, no un bug.

### Bloque 7 — Operar con vista · `v4.32.0`

- `[~]` **W5.1 — métricas y dashboard** hechos; trazas OTLP pendientes. Parte del trabajo ya está hecho
  (`render_prometheus` existe): falta el `content-type` (Bloque 0),
  histogramas por endpoint y por motor, OTLP opt-in (`otel_endpoint`) y un
  dashboard Grafana commiteado + docker-compose de demo que lo levante sin
  editar nada.
- `[x]` **W2.2 — réplica de lectura caliente.** El replay ya existe (arrancar
  *es* replay); convertirlo en replay continuo con offset expuesto. Alcance
  **congelado**: solo lecturas, promoción manual (`luma promote`), sin failover
  automático. El riesgo declarado del SPEC es que este ítem se convierta en un
  proyecto de consenso; Raft es backlog con criterio de entrada explícito
  (demanda multi-escritor real), no una extensión natural de esto.

### Bloque 8 — RESP endurecido · `v4.33.0`

- `[x]` **F4.2 - backup/restore de estructuras y panel RESP.** Las estructuras
  entran en el backup y vuelven del restore (`tests/backup_restore.rs`,
  incluido un score infinito, que es el caso que se serializaba como `null`).
  El panel tiene pestaña **RESP** con conexiones por organizacion y
  comandos/s, sobre `GET /v1/admin/resp`.

  La tasa se calcula **en el cliente**, dividiendo dos muestras. Un servidor
  que devolviera "comandos/s" tendria que elegir una ventana, y esa ventana
  seria un numero cuyo significado el lector no puede ver: suavizado sobre
  que, desde cuando. Consecuencia honesta: recien abierto el panel no muestra
  tasa hasta la segunda lectura, y lo dice en pantalla en vez de pintar un
  cero que se leeria como "nadie lo usa".

  El contador por org lo sostiene un guarda cuyo `Drop` lo suelta:
  `serve_inner` retorna desde una docena de sitios y un decremento olvidado
  deja una conexion fantasma para siempre, que se lee como una fuga del
  servidor y no como un bug de contabilidad. El listener ya aprendio esto una
  vez con `drop_connection`.
- `[x]` **F4.5 - harness permanente en CI.**

  - **Fuzzing del parser con corpus versionado** (`tests/resp_fuzz.rs`): corre
    en cada push, determinista a proposito -- un fallo con semilla aleatoria no
    es reproducible, y eso es la diferencia entre un informe y un encogimiento
    de hombros. Encontro un **desbordamiento de pila**: 40 KB de `*1` repetido
    mataba el proceso, y un desbordamiento no se puede capturar. Un peer sin
    autenticar tumbaba el servidor por el precio de un paquete, antes de que
    `AUTH` entrara en juego.
  - **Nightly** (`.github/workflows/nightly.yml`): matriz crash-recovery con
    200 iteraciones por motor, atomicidad multiclave, `cargo-fuzz` de verdad,
    diferencial contra Redis 7, E2E de clientes y `cargo audit` -- este ultimo
    tambien de noche, porque un advisory se publica en el calendario de otros y
    una dependencia limpia al mergear puede estar vulnerable por la mañana sin
    un solo commit.
  - **Matriz por tipo de registro del WAL** (`tests/wal_record_matrix.rs`): los
    7 tipos, cada uno replicado intacto, roto a medias y con checksum
    corrompido. Ahí es donde las respuestas de verdad difieren: cada tipo tiene
    su propia rama de aplicación, y un tipo sin rama reproduce como nada. No es
    hipotético — a `state_batch` le faltaba la rama en la ruta de redb cuando se
    escribió.

    La expectativa se declara **por caso** en vez de asumirse uniforme, porque
    difiere de verdad: para los registros de KV el WAL es la única fuente de
    verdad, así que un registro roto es como si no hubiera existido; para los de
    vector no, porque el store escribe sus propios ficheros durables y una
    mutación que ya llegó a disco no se deshace. El que se lee raro es
    `vector_collection_dropped`: sin su registro, el `created` anterior
    reproduce y la colección vuelve aunque su directorio se haya borrado.

  **Criterio para quitar el flag "experimental" del listener RESP: nightly
  verde 7 días seguidos.**

### Bloque 9 — Adopción por protocolo · `v4.34.0`

- `[ ]` **W2.3 — failover asistido.** Health-check para proxy, secuencia de
  promoción documentada, fencing por epoch contra split-brain. Honesto: no es
  HA automática.
- `[ ]` **W3.2 — API S3-compatible** sobre el blob store: `PUT/GET/HEAD/DELETE
  Object`, `ListObjectsV2`, buckets, multipart, presigned URLs, **SigV4**. XML
  idéntico al de S3. Entra **por spike con criterio de salida**: validar contra
  la suite mint de MinIO recortada. SigV4 es la parte con esquinas oscuras.
- `[ ]` **W3.3 — OpenAPI generado desde el código**, CI falla si difiere del
  commiteado; SDKs regenerables. Elimina la clase de problema que M4.1 tapa a
  mano en el Bloque 0.

### Bloque 10 — Conector Postgres · `v4.35.0`

La decisión estratégica del producto: Luma **no** reemplaza Postgres, se
conecta a él. Postgres sigue siendo la fuente de verdad transaccional.

- `[ ]` **W4.1 — spike de replicación lógica** (`pgoutput`): crear publicación,
  consumir INSERT/UPDATE/DELETE, confirmar LSN. **Criterio de salida: informe
  de decisión (crate existente vs subset propio) antes de diseñar nada.**
- `[ ]` **W4.2 — `luma connect postgres`:** configuración declarativa
  (fuente + slot + publicación + mapeos tabla → colección/vectores/namespace),
  backfill por `COPY`, reanudación por LSN persistido con el mismo patrón
  `applied_offset` de redb, idempotencia por PK+LSN. Columnas nuevas se
  ingieren; tipos no mapeables se registran y saltan sin romper el stream.
- `[ ]` **W4.3 — búsqueda federada mínima:** hits con referencia de origen
  (tabla, PK) para que la app lea el registro canónico en Postgres.

### Bloque 11 — GA · `v4.36.0` → `1.0`

- `[x]` **W5.2 - cuotas por organización** (las cuatro aplicadas) (≡ B.1 del roadmap). Excedido → 507 tipado con
  los tres números (en uso, pedido, límite). Test de aceptación: org A en su
  límite no degrada a org B, para las dos.

  `max_keys` fue aplicable de inmediato porque las claves KV llevan prefijo de
  tenant, así que el uso de una org es un escaneo por prefijo. Los bytes de
  blob no, y el primer intento de guarda se **borró en vez de publicarse**: sin
  forma de decir qué bytes son de quién, habría cobrado a una org el
  almacenamiento de otra, que es justo el fallo que el criterio prohíbe.

  Lo que lo hizo posible es que el middleware de aislamiento ya registra la
  propiedad en `sys_collections` al primer contacto, así que cada bucket tiene
  exactamente una org dueña.

  El total se **guarda** en SQLite y lo ajusta cada escritura y borrado, en vez
  de medirse: medir significa recorrer los buckets de la org en cada escritura,
  que es O(ficheros) en una ruta caliente. En SQLite y no en memoria porque un
  contador en memoria hay que reconstruirlo en cada arranque, y reconstruirlo
  ES el recorrido que se quería evitar. Se siembra con un recorrido la primera
  vez que se ve una org, así que una org anterior a esta contabilidad se cobra
  lo que ya tiene en vez de empezar de cero.

  Si los ficheros cambian por fuera el total deriva, y para eso está `recount`.
  Dicho claramente porque una cuota calladamente equivocada es peor que una
  ausente.

  **Colas y vectores** salieron más baratos, por razones opuestas. Las colas ya
  están aisladas por directorio (`queues/t_{org}/…`), así que los mensajes de una
  org son un recorrido de un subárbol que le pertenece en exclusiva: ni registro
  de propiedad ni contador guardado, la respuesta está ahí y es pequeña. Los
  vectores los cuenta ya el store (`live_count` por colección), así que la única
  pregunta es qué colecciones son de la org, y eso lo responde el registro.

  Ninguno necesita la maquinaria de total guardado que los bytes de blob sí
  necesitaban, y añadirla habría sido coste sin beneficio: un contador que puede
  derivar, protegiendo un número que nunca fue caro de calcular.

  Se cuenta `live_count` y no el total de registros: un vector con tombstone no
  es almacenamiento que el llamante pueda leer, y cobrarlo dejaría una colección
  permanentemente llena tras suficientes borrados.
- `[~]` **W5.3 - supply chain** (hecho salvo verificar la publicación real de
  la imagen, que necesita un tag).

  - **`unsafe` acotado por el compilador**: los 16 sitios viven en
    `src/vector/` (mmap de segmentos y productos escalares SIMD) y **todos**
    los demás módulos llevan `#[forbid(unsafe_code)]`. Un bloque `unsafe`
    fuera de ahí es un error de compilación, no un comentario de revisión que
    alguien pueda pasar por alto. Inventariado con su justificación en
    `docs/SECURITY.md`.
  - `tests/unsafe_inventory.rs` tapa el hueco que el atributo deja: un
    `pub mod` nuevo sin marcar compila igual, y la protección dejaría de
    cubrir en silencio el código más reciente. Verificado que el guarda falla
    con un módulo sin marcar, y que el `forbid` rechaza un `unsafe` real.
  - **Imagen `FROM scratch`** ya existía en el `Dockerfile`; lo que faltaba es
    que nada la publicaba, nada describía qué llevaba dentro y nada permitía a
    un usuario comprobar que lo que descargó es lo que este repositorio
    construyó. Nuevo job `container` en el release: push a GHCR, SBOM SPDX con
    syft, firma **keyless** con cosign (una clave en un secreto es una clave
    que se filtra con el secreto, y rotarla invalida toda firma pasada), y
    `cosign verify` del propio artefacto recién publicado.
  - `cargo audit` ya estaba en CI; ahora también en el nightly, porque un
    advisory se publica en el calendario de otros.

  **Sin verificar de verdad:** la publicación a GHCR y la firma solo se
  ejecutan al empujar un tag `v*`. La sintaxis está validada y los pasos son
  los estándar, pero nadie ha visto todavía este job pasar.
- `[~]` **W5.5 - documentación de producto** (runbooks hechos; la
  reorganización de carpetas y la prueba con una persona externa, pendientes).

  `docs/RUNBOOKS.md`: respaldo/restore, montar y seguir una réplica, promoción,
  rotación de master key, upgrade, y una tabla de síntoma → dónde mirar. Cada
  comando ejecutado contra el binario real — un runbook con un flag inventado es
  peor que ninguno, porque se descubre en el peor momento.

  Escribirlo destapó dos huecos, ambos arreglados antes de publicarlo:

  - **No había `luma --help`.** Cualquier subcomando mal escrito caía a `serve`,
    así que `luma backupp` arrancaba un servidor en el puerto de producción.
  - **No se podía crear una réplica.** `promote` existía y su par no:
    `mark_replica` vivía en el crate y solo lo llamaban los tests. Una réplica
    que nadie puede crear no es una funcionalidad. Ahora hay `luma demote`.

  Y que el README afirmaba dos veces que RESP **no estaba implementado**, que
  llevaba siendo falso desde el Bloque 6.

  **Pendiente y no negociable para GA:** la aceptación real es *alguien externo
  monta Luma con réplica y respaldo remoto solo con los docs*. Eso necesita una
  persona y no se ha hecho. El documento lo dice de sí mismo en su encabezado.
- `[ ]` **W5.6 — criterio GA:** W1 completo + W2.1 + W2.2 en verde 30 días en
  producción propia + RESP F1–F3 en GA + W5.1 + W5.5. Congelar API v1:
  rupturas ⇒ v2, nunca dentro de v1.

---

## Ítems aparcados (regla 3: benchmarks desactivados)

No se cancelan; quedan fuera del ciclo hasta que se reactiven los benchmarks.

- `[~]` **F4.4 — benchmark honesto vs Redis** (`redis-benchmark`, SET/GET/
  LPUSH/ZADD con y sin pipelining). Es puramente medición. **Nota:** cuando se
  reactive, publicar también dónde Redis gana — es lo que hace creíble la tabla
  vectorial del README.
- `[~]` **W5.4 — suite de carga sostenida** por primitiva (KV ops/s,
  enqueue/dequeue 1 h, blob MB/s, SSE con 5k suscriptores) en el perfil de
  máquina objetivo. Su cláusula de "regresión >15 % rompe el nightly" queda
  también aparcada.

Consecuencia honesta de aparcarlos: **el plan no detecta regresiones de
rendimiento.** Se detectan regresiones de corrección (tests) y de durabilidad
(harness), no de latencia ni de throughput. Es una decisión consciente, no un
olvido; conviene reactivar W5.4 antes de declarar GA en el Bloque 11.

---

## Hallazgos abiertos fuera del alcance del plan

Defectos reales encontrados al pasar las puertas de verificación, que no
pertenecen a ningún bloque y no se arreglan de paso para no ensanchar el
alcance en silencio.

- `[x]` **Atomicidad multiclave de `LMOVE`/`RPOPLPUSH`/`BLMOVE`/`BRPOPLPUSH`.**
  Cerrado. El motor tiene ahora un registro de WAL multiclave
  (`Engine::put_state_batch`) y el movimiento se aplica como un solo registro
  con compare-and-swap sobre ambas claves. Era la garantía que kombu compra al
  usar `BRPOPLPUSH` para su cola de *unacked*. Fijado en `tests/atomic_move.rs`.

  Escribir el arnés destapó un segundo defecto: la ruta de replay de la
  proyección redb no tenía rama para el registro de lote, así que cada
  movimiento se habría perdido en el primer arranque tras un corte — el mismo
  fallo que el lote existe para evitar, reintroducido una capa más abajo.

- `[ ]` **IVF pierde el vecino correcto tras `retrain_ivf`.**
  `tests/vector_ivf.rs::ivf_large_dataset_retrain_consistent` (1M vectores,
  128 clústeres, `nprobe=8`) falla: el top-1 pasa de `vec-102857`
  (coseno ≈ 0.9999 contra la query) a `vec-109999` (coseno ≈ 0.87). No es un
  desempate entre resultados equivalentes — es recall que se cae después de
  reentrenar.

  **No es una regresión de este plan:** `src/vector/ivf.rs` no se ha tocado
  desde `v3.0.0`, y el único cambio en `src/vector/mod.rs` que roza la ruta
  Q8 es `!A && !B` reescrito como early-return, equivalente.

  **Por qué llevaba tiempo oculto:** el test está tras la feature
  `ivf_stress_tests` y CI corre `cargo test --locked`, sin `--all-features`.
  Nunca se ejecutó en CI. Tarda ~200 s en debug.

  Cuando se aborde, la pregunta a responder primero es si la asignación a
  clúster y el ranking usan la misma métrica: con `Metric::Cosine`, unos
  centroides entrenados por distancia L2 sobre vectores sin normalizar
  probarían los clústeres equivocados, que es exactamente esta forma de fallo.

---
## Decisiones de negocio pendientes (no bloquean código)

Heredadas de `SPEC-producto.md`, marcadas ahí como `[DECISIÓN]`:

1. **Licencia.** Hoy MIT. Si se abre a terceros, ¿MIT puro o dual/BSL para
   proteger un futuro servicio gestionado? Revisar **antes del primer cliente
   externo**, no antes de escribir código.
2. **SLA de soporte formal.** Solo si aparece un cliente externo de pago.

---

## Riesgos del plan

| Riesgo | Mitigación |
|---|---|
| El alcance total es de varios meses y se abandona a media ruta | Los bloques 0–3 ya entregan valor por sí solos (deuda cerrada + durabilidad demostrada + RPO). El corte natural si hay que parar es al final de un bloque, nunca a mitad |
| El Bloque 2 (XL) es el más grande y está al principio | F0.1, F0.2 y F0.3 son commiteables por separado; F0.3 es pequeño y podría adelantarse para tener avance visible |
| La semántica de Redis rompe clientes en silencio | La suite diferencial contra Redis real es la fuente de verdad, no la doc. Ningún comando entra sin su caso diferencial |
| `StoredVal` (F0.1) rompe compatibilidad de datos | Registros nuevos etiquetados, los legados se leen como `Json`; el fixture dorado del Bloque 1 lo vigila en CI — por eso el Bloque 1 va **antes** del 2 |
| Tests en lote (regla 2) retrasan la detección de fallos | Cada bloque acota el radio: si la puerta falla, el conjunto de commits sospechosos es un bloque, no el repo |
| Aparcar benchmarks oculta regresiones de rendimiento | Declarado arriba de forma explícita; reactivar W5.4 antes de GA |
| Confusión de identidad ("¿Luma es un Redis?") | RESP es **una interfaz** de la plataforma, no el producto. Fijado así en README y en `docs/RESP.md` |
