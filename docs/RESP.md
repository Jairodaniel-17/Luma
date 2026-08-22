# Compatibilidad con el protocolo de Redis (RESP)

Luma habla RESP2 en un puerto propio, así que un cliente que hoy apunta a
`redis://host:6379` puede apuntar a Luma **cambiando una variable de entorno**.

> **Estado: experimental.** Los comandos de esta página están implementados y
> cubiertos por tests, pero **falta la suite diferencial contra un Redis 7
> real**, que [`SPEC-resp.md`](SPEC-resp.md) define como la fuente de verdad de
> la semántica. Hasta que exista y esté verde varios días seguidos (criterio
> F4.5 del [plan maestro](PLAN-MAESTRO.md)), esto no debe considerarse
> equivalente a Redis en producción. La sección
> [Cómo validar](#cómo-validar-esto-de-verdad) explica qué falta ejecutar.

---

## Activarlo

Apagado por defecto: un motor que empieza a escuchar en el 6379 al actualizar
es una sorpresa, y en un host compartido choca con el Redis de verdad.

```toml
# luma.toml
resp_port = 6379
```

```bash
# o por entorno
RESP_PORT=6379 LUMA_API_KEY=tu-clave ./luma serve
```

| Clave | Default | Para qué |
|---|---:|---|
| `resp_port` | `0` | Puerto. 0 desactiva el listener |
| `resp_max_clients` | `10000` | Conexiones simultáneas. Al llegar al tope se responde `-ERR max number of clients reached` y se cierra, en vez de tirar el socket: un reset seco parece un fallo de red y se reintenta para siempre |
| `resp_idle_timeout_secs` | `300` | Cierra conexiones ociosas. Las suscritas a Pub/Sub están exentas: esperan mensajes por diseño |
| `resp_max_buffer_bytes` | `64 MiB` | Cota del buffer por conexión |
| `resp_pubsub_inbox` | `1024` | Buzón por suscriptor. Lleno, se descarta ese mensaje en vez de atascar al publicador |
| `resp_allow_flush` | `false` | Permite `FLUSHDB`/`FLUSHALL` |

La contraseña es `api_key` (`LUMA_API_KEY`). Sin ella el listener acepta a
cualquiera y lo avisa ruidosamente en los logs.

```bash
# Celery
export CELERY_BROKER_URL=redis://:tu-clave@luma:6379/0
# arq
export REDIS_URL=redis://:tu-clave@luma:6379
```

---

## Comandos soportados

### Conexión

| Comando | Nota |
|---|---|
| `PING [msg]` | Con argumento responde bulk string, no simple string |
| `ECHO`, `QUIT`, `RESET` | |
| `AUTH pass` / `AUTH user pass` | Ambas formas |
| `HELLO [2\|3]` | Responde el handshake pero **declara proto 2** aunque pidas 3 — ver [Divergencias](#divergencias-conocidas) |
| `SELECT 0` | Solo la base 0 |
| `CLIENT SETNAME\|GETNAME\|SETINFO\|ID` | `SETINFO` se acepta y se ignora: redis-py lo manda al conectar |
| `COMMAND` | Devuelve array vacío |
| `INFO [section]` | Secciones `server`, `clients`, `stats`, `keyspace`. Los números son reales: el conteo de claves se mide y está acotado a tu organización |

### Strings

`SET` (con `EX` `PX` `NX` `XX` `KEEPTTL`) · `GET` · `GETDEL` · `GETSET` ·
`SETEX` · `PSETEX` · `SETNX` · `APPEND` · `STRLEN` · `INCR` · `DECR` ·
`INCRBY` · `DECRBY` · `INCRBYFLOAT` · `MGET` · `MSET`

### Claves

`DEL` · `UNLINK` · `EXISTS` · `TYPE` · `TTL` · `PTTL` · `EXPIRE` · `PEXPIRE` ·
`EXPIREAT` · `PEXPIREAT` · `PERSIST` · `RENAME` · `RENAMENX` · `KEYS` ·
`SCAN` (`MATCH`, `COUNT`, `TYPE`) · `RANDOMKEY` · `DBSIZE` · `FLUSHDB` ·
`FLUSHALL`

### Listas

`LPUSH` · `RPUSH` · `LPOP` · `RPOP` (ambos con `count`) · `LLEN` · `LRANGE` ·
`LREM` · `BLPOP` · `BRPOP`

### Hashes

`HSET` · `HGET` · `HMGET` · `HDEL` · `HGETALL` · `HLEN` · `HEXISTS` ·
`HKEYS` · `HVALS` · `HINCRBY`

### Sets

`SADD` · `SREM` · `SMEMBERS` · `SISMEMBER` · `SCARD`

### Sorted sets

`ZADD` · `ZREM` · `ZSCORE` · `ZCARD` · `ZRANGE` (con `WITHSCORES`) ·
`ZRANGEBYSCORE` (con `WITHSCORES`, acepta `+inf`/`-inf`) · `ZRANK`

### Transacciones

`MULTI` · `EXEC` · `DISCARD` · `WATCH` · `UNWATCH`

### Pub/Sub

`SUBSCRIBE` · `PSUBSCRIBE` · `UNSUBSCRIBE` · `PUNSUBSCRIBE` · `PUBLISH` ·
`PUBSUB CHANNELS|NUMSUB|NUMPAT`

---

## Divergencias conocidas

Estas son diferencias reales de comportamiento, no omisiones. Si alguna te
afecta, es mejor saberlo antes de migrar que descubrirlo en producción.

| Qué | Redis | Luma | Por qué |
|---|---|---|---|
| `HELLO 3` | Cambia a RESP3 | Responde el handshake pero sigue en RESP2 | Declarar RESP3 y luego mandar respuestas RESP2 rompe clientes de formas que solo salen bajo carga |
| Bases de datos | 16 por defecto | Solo la 0; `SELECT 1` da error | Responder OK a cualquier índice compartiría en silencio un keyspace entre bases que el cliente cree separadas |
| `SCAN` | Cursor sobre buckets de hash | Índice sobre la lista ordenada de claves | Mantiene la garantía que los clientes usan (una iteración completa devuelve toda clave presente durante ella) sin emular una tabla hash que no existe |
| `KEYS`/`SCAN`/`DBSIZE` | Recorren todo | Acotan en 100 000 claves | `KEYS *` sobre un keyspace grande es la forma clásica de atascar Redis |
| `FLUSHDB` | Siempre disponible | Requiere `resp_allow_flush = true`, y solo borra el keyspace de tu organización | Un flush accidental es irrecuperable sin restore |
| `PUBLISH` | Cuenta receptores globales | Cuenta receptores **de tu organización** | Un conteo global filtraría la existencia de suscriptores de otras |
| Suscriptor | Solo acepta un subconjunto de comandos | Acepta todos | Es un superconjunto; un `PING` de keepalive obtiene respuesta en vez de silencio |
| Tamaño de estructura | Sin límite práctico | `MAX_STRUCTURE_ENTRIES` = 1 000 000 | Una mutación es read-modify-write de la estructura entera. Luma no es para colas de diez millones de mensajes residentes |

### Verificado contra Redis 7 real

`tests/redis_differential.rs` envía **298 comandos idénticos** a un Redis 7 y
a Luma por socket crudo y compara los bytes de respuesta. Está `#[ignore]`
porque necesita un Redis que el proceso no controla:

```bash
docker run -d --name luma-diff-redis -p 16379:6379 redis:7-alpine
LUMA_DIFF_REDIS=127.0.0.1:16379 \
  cargo test --test redis_differential -- --ignored
```

Sin `LUMA_DIFF_REDIS` el test **falla** en vez de pasar en vacío: una suite
que se pone verde sin su sujeto reporta cobertura que no tiene.

La primera ejecución encontró 13 divergencias que la suite propia no podía
cazar, porque codificaba el mismo modelo equivocado que el código. La más
grave: **strings y estructuras vivían en keyspaces separados**, así que un
mismo nombre podía tener a la vez un string y una lista. `TYPE` contestaba
`none` para toda estructura, `EXISTS` contestaba 0, `DEL` no borraba,
`SETNX` tenía éxito sobre una clave ocupada, `EXPIRE`/`TTL`/`RENAME` no
alcanzaban las estructuras y `KEYS *` devolvía el prefijo interno
`struct:`. Dos tests propios afirmaban esa separación **como garantía**.

Ahora es un solo keyspace con un tipo por clave, igual que Redis. El
almacenamiento sigue usando dos ranuras — eso es detalle de implementación;
dos keyspaces era otra base de datos.

### Divergencias de los comandos añadidos

| Comando | Redis | Luma | Por qué |
|---|---|---|---|
| `SPOP`, `SRANDMEMBER` | Eligen miembros al azar | Los toman en el orden almacenado | El uso real de ambos es "dame cualquier miembro", y una respuesta determinista es verificable. Un cliente que dependa de la aleatoriedad para repartir carga no la obtendrá |
| `HSCAN`, `SSCAN`, `ZSCAN` | Cursor sobre buckets de hash | Índice sobre el orden almacenado (un `BTreeMap`/`BTreeSet`, estable) | Da la garantía que los clientes usan de verdad — un elemento presente durante toda la iteración se devuelve al menos una vez — sin emular una tabla hash que no existe. Igual que el `SCAN` de primer nivel |
| `HSCAN`/`SSCAN`/`ZSCAN` `COUNT` | Es una pista | Es un límite de página | Nunca devuelve más de lo pedido, y evita que un `COUNT 10` recorra un millón de entradas |
| `LMOVE`, `RPOPLPUSH`, `BLMOVE`, `BRPOPLPUSH` | Atómicos | **No atómicos entre las dos claves** | Ver abajo |
| `ZADD` con `+inf`/`-inf` | Se almacena | Se almacena | Fue un bug hasta que se arregló: JSON no tiene infinito y `serde_json` lo escribía como `null`, así que el `ZADD` decía OK y el sorted set entero quedaba ilegible. Los scores infinitos ahora se serializan como cadena; los finitos siguen siendo números y los datos existentes no cambian |

#### La atomicidad de `LMOVE` y compañía

El motor confirma **una clave por registro del WAL**, así que un movimiento
entre dos claves es un pop seguido de un push, no una operación. Si el push
falla, el elemento se devuelve a su lista de origen — hay un test que lo fija.
Pero una muerte del proceso entre las dos operaciones pierde el elemento.

Esto importa porque es exactamente la garantía que un cliente compra al usar
`BRPOPLPUSH`: la cola de *unacked* de kombu existe para que una tarea no se
pierda si el worker muere. Con Luma, la ventana es pequeña pero real. Cerrarla
requiere un registro de WAL transaccional multiclave, que es un cambio de diseño
del motor y no un detalle del protocolo; queda anotado como tal en vez de
implícito aquí.

### Multi-tenancy

`AUTH <api-key>` liga la conexión a la organización dueña de esa clave, y a
partir de ahí **todas** sus claves y canales llevan ese prefijo de forma
transparente: dos organizaciones usando `celery` no se ven. La respuesta nombra
siempre la clave que el cliente envió, nunca la forma interna — `BLPOP jobs`
contesta `jobs`, que es lo que kombu compara.

La clave estática de la instancia (`api_key` / `LUMA_API_KEY`) sigue siendo
válida y es la credencial **de plataforma**: no está ligada a ninguna
organización y opera sobre el keyspace sin prefijo, igual que hace por HTTP.

El separador entre organización y clave es el **separador de unidad ASCII**
(`0x1f`), no `:`. Con `:` la organización `a` guardando `b:c` y la organización
`a:b` guardando `c` producen la misma clave física — una lee el dato de la otra.
Un carácter de control no puede aparecer en un id de organización, así que la
división no es ambigua. Los canales de Pub/Sub ya usaban este separador.

**Revocación.** Una clave revocada deja de servir en el **comando siguiente**,
no al reconectar: la conexión permanece abierta y recibe `NOAUTH`, que es lo que
le dice al cliente que vuelva a autenticarse en vez de parecer un fallo de red.
El coste en régimen normal es una lectura atómica por comando; solo se consulta
la base de datos después de que alguien haya revocado algo de verdad.

---

## No implementado

Nada de esto está a medias: simplemente no está.

El set de comandos de las fases 2 y 3 de `SPEC-resp.md` está completo
(57/57), modificadores de `ZADD` y opciones de `ZRANGE` incluidos.

**Fuera de alcance por decisión** (ver el backlog de `SPEC-resp.md`): protocolo
de cluster (`MOVED`/`ASK`), replicación por protocolo Redis (`REPLICAOF`), Lua
(`EVAL`), Streams (`XADD`), keyspace notifications.

---

## Cómo validar esto de verdad

Los ~180 tests de RESP que hay hoy fijan **el comportamiento que pretendemos**.
No prueban que coincida con Redis. Para eso hacen falta dos cosas que necesitan
docker y clientes reales:

### 1. Suite diferencial contra Redis 7

El mismo guion de operaciones contra un Redis 7 real y contra Luma, comparando
salidas byte a byte. Es la fuente de verdad según el SPEC, precisamente porque
la documentación de Redis no captura todas las esquinas.

```bash
docker run -d --name redis7 -p 6380:6379 redis:7
RESP_PORT=6379 ./luma serve &
# comparar redis-cli -p 6380 contra redis-cli -p 6379 sobre el mismo guion
```

Las esquinas que más importan, y que ya están cubiertas por tests unitarios pero
sin contrastar contra Redis: nil frente a array vacío, `$0` frente a `$-1`,
`-WRONGTYPE` cruzando tipos, los centinelas `-1`/`-2` de `TTL`, y el orden de
empate en zsets.

### 2. E2E con los clientes objetivo

```bash
# Celery
CELERY_BROKER_URL=redis://luma:6379/0 celery -A app worker
# arq
REDIS_URL=redis://luma:6379 arq app.WorkerSettings
```

El criterio del SPEC no es "conecta", es: un worker procesando trabajos de punta
a punta, más revoke y restore de unacked tras matar el worker. Con versiones de
cliente fijadas y documentadas aquí cuando se ejecute.

Hasta que ambas cosas estén verdes en CI, el flag experimental de arriba se
queda.
