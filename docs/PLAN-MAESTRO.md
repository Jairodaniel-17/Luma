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
| 0 | Barrido de deuda del roadmap | `v4.25.0` | M3.2, M3.4, M4.1, M3.3 + fix de `content-type` en métricas | S |
| 1 | **Durabilidad demostrada** | `v4.26.0` | W1.1 (harness compartido), W1.2, fixture dorado en CI | M |
| 2 | Cimientos del motor RESP | `v4.27.0` | F0.1 `StoredVal`, F0.2 estructuras, F0.3 notificadores | **XL** |
| 3 | RPO prometible | `v4.28.0` | W1.3 backup remoto, W1.4 verificación, W2.1 WAL shipping | L |
| 4 | Listener RESP + strings/keys | `v4.29.0` | F1.1–F1.4 + **F4.1** (ver D-1) | L |
| 5 | Estructuras por RESP | `v4.30.0` | F2.1 listas, F2.2 hashes, F2.3 sets, F2.4 zsets | L |
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

- `[ ]` **M3.2 — modelo/dim por colección + validar mismatch.** Añadir
  `embedding_model: Option<String>` y `embedding_dim: Option<usize>` a
  `vector::persist::Manifest` con `#[serde(default)]` (el struct ya tiene
  `version` y ese patrón, así que no hay migración). En el ingest por texto,
  si el dim del cliente activo ≠ el de la colección → 400 explicativo en vez
  de escribir un vector corrupto.
- `[ ]` **M3.4 — hot-reload de embeddings/LLM.** `EmbeddingClient` detrás de
  `ArcSwap`; `PUT /v1/config` reconstruye el cliente sin reiniciar.
  Verificable con `POST /v1/config/embedding/probe` mostrando el dim nuevo.
- `[ ]` **M4.1 — SDKs + OpenAPI al día.** Los endpoints que existen en el
  router pero no en los SDKs ni en el YAML: `my-orgs`, `switch-org`,
  `sessions`, `sessions/revoke-all`, `DELETE /v1/vector/:collection`,
  `config/embedding/probe`, `meta/:collection/execute`, invitaciones y
  miembros por org, `users/:id/orgs`, CRUD de roles.
- `[ ]` **M3.3 — reindexado al cambiar de modelo.** El único medium-high del
  bloque: `POST /v1/vector/:col/reindex {target_model}` como job en background
  con progreso. **Si el bloque se alarga, este es el ítem que se mueve al
  Bloque 1**, no los otros tres.
- `[ ]` **Fix — `content-type` de `/v1/metrics`.** El handler
  (`routes_state.rs:52`) devuelve el texto sin cabecera; Prometheus espera
  `text/plain; version=0.0.4`. Dos líneas, y es prerequisito de W5.1.

**Puerta de verificación 0:** fmt + clippy + `cargo test`.

### Bloque 1 — Durabilidad demostrada · `v4.26.0`

El bloque del que dependen todos los demás. Construye el harness que F0.2
también necesita (ver conflicto 2).

- `[ ]` **W1.1 — matriz de crash-recovery.** Harness que mata el proceso en
  puntos aleatorios durante ráfagas de escritura de **cada** motor (KV, blob,
  colas, doc, vector, memoria, SQLite) y verifica al reinicio: prefijo
  confirmado íntegro, cero divergencia memoria↔redb. Diseñarlo **desde el
  inicio con tipos de registro extensibles**, porque el Bloque 2 le añade los
  de estructuras. Job nightly (no en el ciclo normal, por la regla 2).
- `[ ]` **W1.2 — fsync por motor auditado.** Auditar la política extremo a
  extremo. Foco en los dos puntos débiles conocidos: las colas escriben JSON
  por mensaje (¿fsync del archivo *y* del directorio?) y blob (write + rename
  atómico + fsync). Entrega una tabla en `PROD_READINESS.md`: primitiva → qué
  garantiza cuando devuelve OK.
- `[ ]` **Fixture dorado en CI** (política de compatibilidad, punto 4). Un
  `data_dir` generado por la última release publicada, versionado; si la rama
  no lo lee íntegro, el build falla. Se regenera por release. Sin esto, la
  regla 4 es una intención y no un control.

**Puerta de verificación 1:** fmt + clippy + `cargo test` + primera corrida
verde del harness.

### Bloque 2 — Cimientos del motor RESP · `v4.27.0` · **el bloque más grande**

Sin RESP todavía. Todo lo de aquí sirve también a la API HTTP.

- `[ ]` **F0.1 — `StoredVal { Json, Raw }`.** El ítem de mayor riesgo de
  compatibilidad del plan: toca `state.rs`, `state_db.rs`, `persist.rs` y la
  API HTTP (expone `Raw` como base64 + `content_type`). Registro WAL nuevo
  versionado; **los registros legados se leen como `Json` y no se
  reinterpretan nunca**. Property test: cualquier `Vec<u8>` sobrevive
  put→crash→replay→get idéntico.
- `[ ]` **F0.2 — motor de estructuras** (`engine/structures.rs`): list, hash,
  set, zset con semántica Redis, durables por el mismo WAL + redb. Decisiones
  que se toman aquí y no se parchean después:
  - zset con `BTreeMap<(score, member)>` — el orden lexicográfico en empates
    de score es contrato y se testea;
  - **contador de revisión por clave**, que F3.2 (`WATCH`) necesita;
  - una clave, un tipo — operar con el tipo equivocado da error tipado (será
    `-WRONGTYPE`);
  - límites `MAX_STRUCTURE_ENTRIES` / `MAX_MEMBER_LEN` configurables.
- `[ ]` **F0.3 — notificadores por clave.** `tokio::sync::Notify` por clave
  para los bloqueantes de F3.1. Un `LPUSH` despierta exactamente un `BLPOP`
  (sin thundering herd); limpieza al quedar sin waiters.

**Puerta de verificación 2:** fmt + clippy + `cargo test` + harness del Bloque 1
extendido a los registros nuevos.

### Bloque 3 — RPO prometible · `v4.28.0`

- `[ ]` **W1.3 — backup remoto** a destino S3-compatible con el crate
  `object_store` (S3/R2/GCS/MinIO con una API). Artefacto cifrado con la master
  key; `luma restore s3://…` directo; retención remota. Dependencia nueva →
  pasa por `deny.toml`.
- `[ ]` **W1.4 — `luma backup --verify`.** Restaura a un temporal, corre
  checks, reporta. `backup_last_verified_ts` en métricas + regla Prometheus de
  ejemplo. Un backup no verificado no cuenta como backup.
- `[ ]` **W2.1 — WAL shipping continuo** al bucket (estilo Litestream).
  Snapshot + cadena de segmentos = punto en el tiempo;
  `luma restore --to-timestamp`. **Requiere un spike previo** para decidir qué
  hacer con el WAL de SQLite (¿shipping directo o checkpoint + copia?);
  ese spike es la primera tarea del bloque, no un detalle de implementación.
  Métrica `wal_ship_lag_seconds`.

**Puerta de verificación 3:** fmt + clippy + `cargo test` + demo de
destrucción/restauración con pérdida ≤ `ship_interval`.

### Bloque 4 — Listener RESP + strings/keys · `v4.29.0`

El hito visible: `redis-cli -p 6379` conversa con Luma.

- `[ ]` **F1.1 — framing RESP2 + ciclo de conexión.** Listener propio en
  `resp_port` (default **desactivado**), mismo proceso. Decisión pendiente de
  D2: evaluar el crate `redis-protocol` contra `deny.toml` primero; si no
  pasa, parser propio. Soportar *inline commands* (`PING\r\n`), porque
  redis-cli los usa. Pipelining. Fuzzing del parser sin panics.
- `[ ]` **F4.1 — TLS y límites** (adelantado, ver D-1): `resp_max_clients`,
  `resp_idle_timeout_secs`, buffer máximo por conexión, backpressure en
  suscriptores lentos. TLS reusando `rustls`, que ya es dependencia.
- `[ ]` **F1.2 — AUTH multi-tenant.** `AUTH` mapea a las api keys/roles
  actuales; keyspace prefijado por `{org_id}\x1f`; `KEYS`/`SCAN` filtran por
  tenant. Dos orgs con la misma clave no se ven. Key revocada corta en el
  siguiente comando.
- `[ ]` **F1.3 — comandos de strings/keys.** El set completo del SPEC
  (`GET SET SETEX … SCAN RENAME`), con `FLUSHDB` solo bajo
  `resp_allow_flush = true`. **La suite diferencial contra Redis 7 real en
  docker es la fuente de verdad**, no la documentación de Redis: ~200
  operaciones, salidas byte-idénticas. `SCAN` con cursor real, no snapshot.
- `[ ]` **F1.4 — observabilidad RESP.** `resp_connections_gauge`,
  `resp_commands_total{cmd}`, `resp_errors_total{kind}`,
  `resp_auth_failures_total`. `INFO` con secciones reales — kombu las lee.

**Puerta de verificación 4:** fmt + clippy + `cargo test` + suite diferencial +
smoke de redis-py.

### Bloque 5 — Estructuras por RESP · `v4.30.0`

Exponer por protocolo lo que el Bloque 2 ya implementó. Riesgo bajo de diseño,
alto de detalle.

- `[ ]` **F2.1 listas** · `[ ]` **F2.2 hashes** · `[ ]` **F2.3 sets** ·
  `[ ]` **F2.4 sorted sets** (los comandos exactos, en `SPEC-resp.md`).

Las dos trampas que la aceptación tiene que cazar explícitamente: **nil vs
array vacío** y **`-WRONGTYPE` cruzando tipos**. Ahí es donde los clientes
rompen en silencio. Los patrones reales de la matriz de clientes (kombu unacked
con `HSET`+`ZADD`, arq con `ZRANGEBYSCORE`+`ZREM`) entran como tests con frames
capturados de los clientes de verdad, no inventados.

**Puerta de verificación 5:** fmt + clippy + `cargo test` + suite diferencial
ampliada a estructuras.

### Bloque 6 — Celery y arq funcionando · `v4.31.0` · **el hito de adopción**

- `[ ]` **F3.1 — bloqueantes:** `BLPOP BRPOP BLMOVE BRPOPLPUSH BZPOPMIN
  BZPOPMAX`, multi-clave con orden de argumentos como contrato. Cierre de
  conexión con waiters pendientes no filtra memoria (test con 1k conexiones).
- `[ ]` **F3.2 — `MULTI/EXEC/DISCARD/WATCH/UNWATCH`.** `WATCH` usa la
  `revision` del `StateStore` y el contador por clave que el Bloque 2 dejó en
  las estructuras. Test de carrera: 100 clientes incrementando la misma clave
  → suma exacta.
- `[ ]` **F3.3 — Pub/Sub** sobre el `EventBus` existente, canal interno
  `resp:{org}:{canal}`. `PUBLISH` devuelve receptores **del tenant**, no
  globales.
- `[ ]` **F3.4 — E2E de los clientes objetivo.** `tests/resp/e2e_arq/` y
  `tests/resp/e2e_celery/` con versiones fijadas, en CI. Celery incluye revoke
  y restore de unacked tras matar el worker.
- `[ ]` **F4.3 — `docs/RESP.md`** (adelantado, ver D-2): tabla de comandos con
  notas de divergencia, guía "migrar de Redis a Luma en 5 minutos" para Celery,
  arq, redis-py e ioredis, y qué NO se soporta y por qué. README enlaza.

**Puerta de verificación 6:** fmt + clippy + `cargo test` + los dos E2E verdes.
**Antes de cerrar el bloque, validar la matriz de comandos contra las versiones
fijadas de cada cliente** — `SPEC-resp.md` advierte que arq puede exigir
`SCRIPT/EVAL` según versión (backlog B-R.3). Si aparece, es una decisión de
alcance, no un bug.

### Bloque 7 — Operar con vista · `v4.32.0`

- `[ ]` **W5.1 — métricas y trazas.** Parte del trabajo ya está hecho
  (`render_prometheus` existe): falta el `content-type` (Bloque 0),
  histogramas por endpoint y por motor, OTLP opt-in (`otel_endpoint`) y un
  dashboard Grafana commiteado + docker-compose de demo que lo levante sin
  editar nada.
- `[ ]` **W2.2 — réplica de lectura caliente.** El replay ya existe (arrancar
  *es* replay); convertirlo en replay continuo con offset expuesto. Alcance
  **congelado**: solo lecturas, promoción manual (`luma promote`), sin failover
  automático. El riesgo declarado del SPEC es que este ítem se convierta en un
  proyecto de consenso; Raft es backlog con criterio de entrada explícito
  (demanda multi-escritor real), no una extensión natural de esto.

### Bloque 8 — RESP endurecido · `v4.33.0`

- `[ ]` **F4.2 —** las estructuras entran en `/v1/admin/backup` y restore; el
  panel muestra conexiones RESP por org y comandos/s.
- `[ ]` **F4.5 —** harness permanente en CI: matriz crash-recovery por tipo de
  registro WAL, fuzzing del parser con corpus versionado, suite diferencial
  completa como job nightly. **Criterio para quitar el flag "experimental" del
  listener RESP: nightly verde 7 días seguidos.**

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

- `[ ]` **W5.2 — cuotas por organización** (≡ B.1 del roadmap): bytes en blob,
  claves KV, mensajes en cola, vectores, rps por org. Excedido → error tipado +
  métrica + evento de auditoría, visible en el panel. Test: org A en su límite
  no degrada a org B.
- `[ ]` **W5.3 — supply chain:** imagen `FROM scratch` firmada con cosign,
  SBOM por release, `cargo audit` en CI, inventario de `unsafe` y
  `#![forbid(unsafe_code)]` donde ya se cumpla.
- `[ ]` **W5.5 — documentación de producto:** reorganizar `docs/` en *Empezar*
  / *Operar* (runbooks: backup/restore, promoción de réplica, rotación de
  master key, upgrade) / *Integrar* (RESP, S3, CDC, SSE) / *Referencia*.
  Aceptación real: **alguien externo monta Luma con réplica y backup remoto
  solo con los docs.** Probarlo de verdad con una persona.
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
