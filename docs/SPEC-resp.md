# SPEC — Protocolo Redis (RESP) y endurecimiento de producto (post v4.24.0)

Plan ejecutable para que Luma **hable el protocolo de Redis** y se vuelva adoptable
sin cambiar código de aplicación: cualquier cliente que hoy apunta a
`redis://host:6379` (Celery/kombu, arq, redis-py, ioredis, node-redis) debe poder
apuntar a Luma. Cada ítem es shippable de forma independiente, con criterios de
aceptación verificables. Mismo flujo probado: rama → commit → tag → CI verde
(rustfmt/clippy/test) → build musl → deploy → verificación E2E.

Convención de estado por ítem: `[ ]` pendiente · `[~]` en curso · `[x]` hecho.

## Por qué

La superficie actual (KV `/v1/state`, colas `/v1/queue`, eventos `/v1/stream`)
es funcional pero **propietaria**: adoptarla exige tocar código del cliente.
El ecosistema Python/Node del mundo real (Celery, arq, BullMQ, cachés de
sesión) habla RESP. Un listener RESP convierte a Luma en reemplazo *drop-in*
de Redis para ese ecosistema — la puerta de adopción más barata que existe:
cambiar una URL de entorno.

**Criterio de éxito global del SPEC:** `REDIS_URL=redis://luma:6379/0` en una
app arq y en una app Celery reales, sin tocar una línea de su código, con los
workers procesando trabajos de punta a punta.

## Alcance y no-alcance

**Dentro:** RESP2 completo + saludo RESP3 (`HELLO`), strings/keys/TTL,
estructuras (list/hash/set/zset), comandos bloqueantes, `MULTI/EXEC/WATCH`,
Pub/Sub, AUTH multi-tenant, TLS, límites anti-DoS, métricas, pruebas
diferenciales contra Redis real, documentación de compatibilidad.

**Fuera (backlog al final):** protocolo de cluster de Redis (`MOVED/ASK`),
replicación por protocolo Redis (`REPLICAOF`), Lua (`EVAL`), Streams (`XADD`),
keyspace notifications. Ninguno lo exigen los clientes objetivo.

---

## Decisiones de diseño (fijas para todo el SPEC)

- **D1 — Listener propio, mismo proceso.** Nuevo `TcpListener` en
  `resp_port` (default **desactivado**; convención 6379), montado en
  `server.rs` junto al HTTP. Comparte `StateStore`, `EventBus` y el motor de
  estructuras nuevo. Sin proceso aparte: se mantiene la tesis del binario único.
- **D2 — Parser/encoder RESP propio o crate `redis-protocol`.** Evaluar el
  crate primero (mantenido, sin unsafe crítico, cumple `deny.toml`); si no pasa,
  parser propio (~500 líneas: RESP2 es un protocolo de prefijos triviales).
  Soportar *inline commands* (`PING\r\n` a secas) porque redis-cli los usa.
- **D3 — AUTH = api keys existentes.** `AUTH <password>` y
  `AUTH <user> <password>` mapean a las api keys/roles actuales
  (`src/api/auth*`). El usuario resuelve la organización; sin AUTH válido solo
  se aceptan `PING`, `HELLO`, `AUTH`, `QUIT` (paridad con `requirepass`).
- **D4 — Multi-tenant por prefijo interno de keyspace.** Toda clave RESP se
  almacena como `{org_id}\x1f{clave}`. `KEYS`/`SCAN` filtran por el prefijo del
  tenant autenticado. Ninguna operación cruza organizaciones — misma garantía
  de aislamiento que la API HTTP.
- **D5 — Bytes crudos como tipo de valor de primera clase.** Redis es binario;
  el KV actual guarda `serde_json::Value`. Se extiende el valor almacenado a
  `enum StoredVal { Json(Value), Raw(Vec<u8>) }` con serialización etiquetada
  **retrocompatible** (registros WAL/redb existentes se leen como `Json`; no hay
  migración). La API HTTP expone `Raw` como base64 con `content_type`.
- **D6 — Estructuras nuevas en un motor propio (`src/engine/structures.rs`).**
  Listas, hashes, sets y zsets tipados por entrada, en memoria con el mismo
  patrón sharded del `StateStore`, durables vía los **mismos** WAL + redb
  (registros nuevos: `LPush`, `HSet`, `ZAdd`, … + replay). El zset usa
  `BTreeMap<(score, member)>` — el orden lexicográfico de Redis en empates de
  score es parte del contrato y se testea.
- **D7 — Bloqueantes con `tokio::sync::Notify` por clave.** `BLPOP/BRPOP`
  esperan en un `Notify` que dispara cualquier `*PUSH` sobre esa clave, con
  timeout del comando. Un solo mapa de notifiers, limpieza al quedar sin waiters.
- **D8 — TTL y expiración reutilizan el heap existente.** La expiración es
  perezosa en lectura (como hoy) + barrido del heap; `TTL/PTTL` leen
  `expires_at_ms`. `SELECT` acepta solo `0` (error `-ERR DB index` para otros,
  igual que Redis con `databases 1`).
- **D9 — Errores con formato Redis.** Prefijos exactos (`-ERR`, `-WRONGTYPE`,
  `-NOAUTH`, `-EXECABORT`): los clientes hacen *matching* de esos strings.

---

## Matriz de comandos exigidos por los clientes objetivo

| Cliente | Comandos que usa (verificar contra su versión al implementar) |
|---|---|
| redis-py / ioredis (conexión) | `HELLO`, `AUTH`, `PING`, `ECHO`, `SELECT`, `CLIENT SETNAME/SETINFO`, `INFO`, `COMMAND DOCS` (puede responderse vacío) |
| **arq** | `ZADD`, `ZRANGEBYSCORE`, `ZREM`, `ZCARD`, `SET` (EX/PX/NX), `GET`, `DEL`, `EXPIRE`, `PSETEX`, `INCR`, `WATCH/MULTI/EXEC`, `SCRIPT`* |
| **Celery (kombu transporte redis)** | `LPUSH`, `RPUSH`, `BRPOP` (multi-clave), `LREM`, `LLEN`, `HSET/HDEL/HGETALL` (unacked), `ZADD/ZREM/ZRANGEBYSCORE` (unacked_index), `SADD/SREM/SMEMBERS` (bindings), `PUBLISH/SUBSCRIBE/PSUBSCRIBE` (fanout), `MULTI/EXEC`, `SETEX`, `EXPIRE` |
| Caché de sesión genérica | `GET/SET/DEL/EXPIRE/TTL/INCR/MGET/MSET` |

\* Si la versión objetivo de arq exige `SCRIPT/EVAL`, ver Backlog B-R.3; las
versiones que usan `WATCH/MULTI/EXEC` puro no lo necesitan. **Fijar la versión
de cada cliente en el harness de pruebas y validar esta tabla en F1.**

---

## Fase 0 — Cimientos del motor (release **v4.25.0**)
Sin RESP todavía: preparar el motor para binarios y estructuras. Todo lo de
esta fase es útil por sí solo (la API HTTP también gana tipos).

### 0.1 `[ ]` Valor crudo (bytes) en el KV  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** guardar bytes arbitrarios sin pasar por JSON (D5).
**Enfoque:** `StoredVal { Json, Raw }` en `state.rs`/`state_db.rs`/`persist.rs`;
WAL con registro nuevo versionado; lectura de registros legados → `Json`.
**Aceptación:**
- Round-trip de 1 MB de bytes aleatorios por la API HTTP (base64) y tras
  reinicio (WAL replay + redb).
- Snapshot + restore conserva ambos tipos. Property test: cualquier `Vec<u8>`
  sobrevive put→crash→replay→get idéntico.

### 0.2 `[ ]` Motor de estructuras (list/hash/set/zset)  · impacto ALTO · esfuerzo ALTO
**Objetivo:** las 4 estructuras con semántica Redis, durables (D6).
**Enfoque:** `engine/structures.rs` + registros WAL nuevos + tablas redb nuevas
+ replay. Cada clave tiene **un** tipo; operar con el tipo equivocado devuelve
error tipado (será `-WRONGTYPE` en RESP). `DEL`/TTL aplican a estructuras.
**Aceptación:**
- Suite de semántica: orden de listas, empates de score en zset
  (orden lexicográfico), `HGETALL` estable, TTL sobre estructura completa.
- Crash-test: matar el proceso durante ráfaga de `LPUSH/ZADD` 500 veces
  (harness) → replay sin pérdida de prefijo confirmado ni corrupción
  (`wal_replay_corrupt_total` = 0).
- Límites anti-DoS: `MAX_STRUCTURE_ENTRIES`, `MAX_MEMBER_LEN` configurables.

### 0.3 `[ ]` Notificadores por clave  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** primitiva de espera para bloqueantes (D7).
**Aceptación:** test con 50 waiters concurrentes sobre 1 clave: un `LPUSH`
despierta exactamente uno en `BLPOP` (sin thundering herd), timeout preciso ±50 ms.

---

## Fase 1 — Listener RESP + strings/keys (release **v4.26.0**)
El hito visible: `redis-cli -p 6379` conversa con Luma.

### 1.1 `[ ]` Framing RESP2 + ciclo de conexión  · impacto ALTO · esfuerzo MEDIO
**Enfoque:** listener en `resp_port` (D1), parser (D2), pipeline de comandos
(leer N comandos sin responder = pipelining), `HELLO` responde mapa RESP2/3,
`QUIT`, timeouts de idle, `max_resp_clients`.
**Aceptación:**
- `redis-cli` interactivo: `PING`, `ECHO`, pipelining con `--pipe`.
- Fuzzing del parser (cargo-fuzz, corpus de frames malformados): sin panics.
- Conexión sin AUTH solo acepta el subconjunto D3.

### 1.2 `[ ]` AUTH multi-tenant  · impacto ALTO · esfuerzo BAJO
**Enfoque:** D3 + D4. Auditoría: login RESP registra evento igual que HTTP.
**Aceptación:** dos orgs con la misma clave `foo` no se ven entre sí vía
`GET/KEYS/SCAN`; api key revocada corta la conexión en el siguiente comando.

### 1.3 `[ ]` Comandos de strings/keys  · impacto ALTO · esfuerzo MEDIO
`GET SET (EX PX EXAT NX XX GET KEEPTTL) SETEX PSETEX SETNX GETSET GETDEL APPEND
STRLEN INCR DECR INCRBY DECRBY INCRBYFLOAT MGET MSET DEL UNLINK EXISTS EXPIRE
PEXPIRE EXPIREAT TTL PTTL PERSIST TYPE KEYS SCAN (MATCH COUNT) RANDOMKEY DBSIZE
RENAME FLUSHDB` (este último solo con `resp_allow_flush = true`).
**Aceptación:**
- Suite diferencial: mismo guion de ~200 operaciones contra Redis 7 real
  (docker) y contra Luma → salidas byte-idénticas (harness en `tests/`).
- `SCAN` con cursor real (no snapshot completo): recorre 100k claves en lotes
  sin O(n) por llamada.
- redis-py: suite de smoke propia verde (`tests/resp/test_redis_py.py`).

### 1.4 `[ ]` Observabilidad RESP  · impacto MEDIO · esfuerzo BAJO
`/v1/metrics`: `resp_connections_gauge`, `resp_commands_total{cmd}`,
`resp_errors_total{kind}`, `resp_auth_failures_total`. `INFO` responde
secciones mínimas (`server`, `clients`, `memory`, `stats`) con datos reales —
kombu y dashboards las leen.
**Aceptación:** `redis-cli INFO` muestra versión Luma y contadores que avanzan.

---

## Fase 2 — Estructuras por RESP (release **v4.27.0**)

### 2.1 `[ ]` Listas  · impacto ALTO · esfuerzo MEDIO
`LPUSH RPUSH LPUSHX RPUSHX LPOP RPOP (count) LLEN LRANGE LREM LINDEX LSET LTRIM
RPOPLPUSH LMOVE`.
### 2.2 `[ ]` Hashes  · impacto ALTO · esfuerzo BAJO
`HSET HSETNX HGET HMGET HDEL HGETALL HLEN HEXISTS HKEYS HVALS HINCRBY HSCAN`.
### 2.3 `[ ]` Sets  · impacto MEDIO · esfuerzo BAJO
`SADD SREM SMEMBERS SISMEMBER SCARD SPOP SRANDMEMBER SSCAN`.
### 2.4 `[ ]` Sorted sets  · impacto ALTO · esfuerzo MEDIO
`ZADD (NX XX GT LT CH INCR) ZREM ZSCORE ZMSCORE ZCARD ZCOUNT ZINCRBY ZRANGE
(REV BYSCORE BYLEX LIMIT WITHSCORES) ZRANGEBYSCORE ZREVRANGEBYSCORE
ZREMRANGEBYSCORE ZREMRANGEBYRANK ZRANK ZREVRANK ZPOPMIN ZPOPMAX ZSCAN`.

**Aceptación (común a 2.1–2.4):**
- Suite diferencial contra Redis 7 ampliada a estructuras (incluye
  `-WRONGTYPE` cruzando tipos, respuestas nil vs array vacío — trampas
  clásicas de compatibilidad).
- Los patrones exactos de la matriz de clientes (kombu unacked con
  `HSET`+`ZADD`, arq con `ZRANGEBYSCORE`+`ZREM`) reproducidos como tests con
  los frames reales capturados de cada cliente.

---

## Fase 3 — Bloqueantes, transacciones y Pub/Sub (release **v4.28.0**)
Lo que desbloquea a Celery y arq de verdad.

### 3.1 `[ ]` Comandos bloqueantes  · impacto ALTO · esfuerzo MEDIO
`BLPOP BRPOP (multi-clave, timeout decimal) BLMOVE BRPOPLPUSH BZPOPMIN BZPOPMAX`.
**Enfoque:** 0.3; multi-clave espera en varios notifiers y toma el primero
disponible en orden de argumentos (contrato Redis).
**Aceptación:** worker kombu real bloqueado en `BRPOP` de 3 colas recibe el
mensaje correcto < 5 ms tras el `LPUSH`; timeout 0 = infinito; cierre limpio de
conexión con waiters pendientes no filtra memoria (test con 1k conexiones).

### 3.2 `[ ]` MULTI / EXEC / DISCARD / WATCH / UNWATCH  · impacto ALTO · esfuerzo MEDIO
**Enfoque:** cola de comandos por conexión; `WATCH` registra `(clave, revision)`
usando la `revision` que ya existe en el `StateStore`; en `EXEC`, si alguna
revision cambió → `nil` (aborto optimista). Las estructuras ganan un
contador de revisión por clave en F0.2 para esto.
**Aceptación:** test de carrera: 100 clientes `WATCH/MULTI/EXEC` incrementando
la misma clave → suma exacta; `EXECABORT` en comando malformado dentro de MULTI.

### 3.3 `[ ]` Pub/Sub  · impacto ALTO · esfuerzo MEDIO
`SUBSCRIBE UNSUBSCRIBE PSUBSCRIBE PUNSUBSCRIBE PUBLISH PUBSUB CHANNELS/NUMSUB`.
**Enfoque:** mapeo al `EventBus` existente con canal interno
`resp:{org}:{canal}`; patrones glob de `PSUBSCRIBE` con el matcher de `KEYS`.
Una conexión en modo suscriptor solo acepta el subconjunto que Redis permite.
**Aceptación:** fanout de kombu (exchange tipo fanout) entrega a 2 workers;
`PUBLISH` devuelve el número de receptores del tenant, no globales.

### 3.4 `[ ]` E2E de clientes objetivo  · impacto ALTO · esfuerzo MEDIO
**El criterio de éxito global, automatizado:**
- `tests/resp/e2e_arq/`: app arq de ejemplo (enqueue + worker + resultado)
  corriendo contra Luma en CI.
- `tests/resp/e2e_celery/`: tarea Celery round-trip + revoke + restore de
  unacked tras matar el worker.
- Ambos con versiones fijadas y matriz documentada en `docs/integrar/RESP.md`.

---

## Fase 4 — Producto sólido (release **v4.29.0**)
Lo que separa "funciona" de "se opera con confianza".

### 4.1 `[ ]` TLS + límites en el listener RESP  · impacto ALTO · esfuerzo BAJO
rustls reutilizando la config TLS existente; `resp_max_clients`,
`resp_idle_timeout_secs`, buffer máximo por conexión, backpressure en
suscriptores lentos (paridad con la política SSE: desconectar, no crecer).
**Aceptación:** `redis-cli --tls` conecta; 10k conexiones idle no pasan de
límites de memoria definidos; cliente lento en pubsub es desconectado con log.

### 4.2 `[ ]` Backups y panel  · impacto MEDIO · esfuerzo BAJO
Las estructuras F0.2 entran en `/v1/admin/backup` y restore; el panel admin
muestra conexiones RESP activas por org y comandos/s.
**Aceptación:** backup→restore→suite diferencial verde sobre datos restaurados.

### 4.3 `[ ]` Documentación y matriz de compatibilidad  · impacto ALTO · esfuerzo BAJO
`docs/integrar/RESP.md`: comandos soportados (tabla completa con notas de divergencia),
guía "migrar de Redis a Luma en 5 minutos" (Celery, arq, redis-py, ioredis),
qué NO soporta y por qué. README enlaza la matriz.
**Aceptación:** un dev externo configura Celery contra Luma solo con el doc.

### 4.4 `[ ]` Benchmark honesto vs Redis  · impacto MEDIO · esfuerzo BAJO
`redis-benchmark` (SET/GET/LPUSH/ZADD, pipelining on/off) Luma vs Redis 7 en la
misma máquina, mismo formato que los benchmarks vectoriales del README:
tablas medidas, sin cifras vagas, publicando también donde Redis gana.
**Aceptación:** sección en `docs/referencia/BENCHMARKS.md` + resumen en README.

### 4.5 `[ ]` Pruebas de resiliencia continuas  · impacto ALTO · esfuerzo MEDIO
Harness permanente en CI: crash-recovery matrix (kill -9 durante escritura de
cada tipo de registro WAL), fuzzing del parser con corpus versionado, suite
diferencial completa contra Redis real como job nightly.
**Aceptación:** job nightly verde 7 días seguidos antes de declarar GA el
listener RESP (quitar el flag "experimental" de la config).

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Semántica sutil de Redis (nil vs array vacío, enteros vs strings, orden de zset) rompe clientes en silencio | La suite **diferencial contra Redis real** es la fuente de verdad, no la doc; todo comando nuevo entra con su caso diferencial |
| Estructuras grandes en RAM (el motor es memoria + WAL) | Límites por estructura (F0.2) + métricas de memoria por org; documentar que Luma no es para colas de 10M de mensajes residentes |
| `WATCH` sobre estructuras exige revisiones por clave | Se diseña en F0.2 desde el inicio (contador por clave), no se parcha después |
| El crate RESP elegido queda sin mantenimiento | D2 contempla parser propio; el protocolo es estable desde hace una década |
| Confusión de identidad de producto ("¿Luma es un Redis?") | Posicionamiento en README: Redis-compatible es **una interfaz** de la plataforma, no el producto |

## Backlog (fuera de este SPEC)

- **B-R.1 Replicación/HA** — prerequisito de producción seria; se especifica
  aparte (réplicas embebidas libSQL / WAL shipping primero, consenso después).
  Este SPEC deja los registros WAL de estructuras listos para reproducirse.
- **B-R.2 Streams (`XADD/XREADGROUP`)** — solo si aparece un cliente objetivo
  que los use (p. ej. Celery con streams o BullMQ Pro).
- **B-R.3 `EVAL`/Lua vía `mlua`** — solo si la versión fijada de un cliente
  objetivo lo exige de forma insalvable.
- **B-R.4 Keyspace notifications** — el EventBus lo hace casi gratis; esperar
  demanda real.
- **B-R.5 Protocolo de cluster** — no: la escala horizontal de Luma se
  resolverá en B-R.1, no imitando cluster de Redis.
