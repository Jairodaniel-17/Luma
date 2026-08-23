# Compatibilidad con el protocolo de Redis (RESP)

Luma habla RESP2 en un puerto propio, así que un cliente que hoy apunta a
`redis://host:6379` puede apuntar a Luma **cambiando una variable de entorno**.

> **Estado: experimental.** Los comandos de esta página están implementados y
> cubiertos por tests, pero **falta la suite diferencial contra un Redis 7
> real**, que [`SPEC-resp.md`](../SPEC-resp.md) define como la fuente de verdad de
> la semántica. Hasta que exista y esté verde varios días seguidos (criterio
> F4.5 del [plan maestro](../PLAN-MAESTRO.md)), esto no debe considerarse
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

`tests/redis_differential.rs` envía **327 comandos idénticos** a un Redis 7 y
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

### Clientes reales verificados

`tests/e2e/clients.py` corre las librerías que un usuario instalaría —
**redis-py, kombu, Celery y arq** — contra Luma y contra un Redis 7 de control.
Incluye un **worker Celery de verdad** que consume la tarea, la ejecuta y
devuelve el resultado al llamante; no basta con que el broker acepte el
mensaje.

```bash
pip install redis celery arq
python tests/e2e/clients.py \
  --redis redis://127.0.0.1:16379/0 \
  --luma  redis://127.0.0.1:16380/0
```

Su primera ejecución encontró que **`PUBLISH` dentro de `MULTI` se rechazaba**
como comando desconocido. redis-py envuelve todo pipeline en `MULTI`/`EXEC` y
el backend de resultados de Celery escribe el resultado con un `SETEX` +
`PUBLISH` en pipeline: el worker consumía la tarea, la ejecutaba, y el llamante
esperaba para siempre un resultado que nunca se guardó. Ni los tests unitarios
ni el corpus diferencial cubrían una transacción, así que nada lo cazó. Ahora
el corpus tiene transacciones.

#### `SUBSCRIBE` dentro de `MULTI`

Redis lo encola y lo ejecuta en `EXEC`, dejando la conexión en modo suscriptor.
Luma lo **rechaza** con un error: suscribirse necesita el buzón del suscriptor,
que `EXEC` no tiene. Es un error explícito, no una conexión que calladamente no
se suscribió. `PUBLISH`, `PUBSUB CHANNELS` y `PUBSUB NUMSUB` sí funcionan
dentro de una transacción, que es lo que los clientes usan.

### Divergencias de los comandos añadidos

| Comando | Redis | Luma | Por qué |
|---|---|---|---|
| `SPOP`, `SRANDMEMBER` | Eligen miembros al azar | Los toman en el orden almacenado | El uso real de ambos es "dame cualquier miembro", y una respuesta determinista es verificable. Un cliente que dependa de la aleatoriedad para repartir carga no la obtendrá |
| `HSCAN`, `SSCAN`, `ZSCAN` | Cursor sobre buckets de hash | Índice sobre el orden almacenado (un `BTreeMap`/`BTreeSet`, estable) | Da la garantía que los clientes usan de verdad — un elemento presente durante toda la iteración se devuelve al menos una vez — sin emular una tabla hash que no existe. Igual que el `SCAN` de primer nivel |
| `HSCAN`/`SSCAN`/`ZSCAN` `COUNT` | Es una pista | Es un límite de página | Nunca devuelve más de lo pedido, y evita que un `COUNT 10` recorra un millón de entradas |
| `ZADD` con `+inf`/`-inf` | Se almacena | Se almacena | Fue un bug hasta que se arregló: JSON no tiene infinito y `serde_json` lo escribía como `null`, así que el `ZADD` decía OK y el sorted set entero quedaba ilegible. Los scores infinitos ahora se serializan como cadena; los finitos siguen siendo números y los datos existentes no cambian |

#### La atomicidad de `LMOVE` y compañía

**Son atómicos.** Un movimiento entre dos claves escribe **un solo registro**
de WAL con las dos mutaciones, protegido por un compare-and-swap sobre cada
clave. Los registros llevan checksum y el replay se detiene en el primero
corrupto, así que tras un corte el lote está entero o no está: no hay estado
en el que el elemento esté en ninguna de las dos listas, ni en las dos.

Importa porque es exactamente la garantía que un cliente compra al usar
`BRPOPLPUSH`: la cola de *unacked* de kombu existe para que una tarea no se
pierda si el worker muere a media entrega. Antes esto era un pop seguido de un
push, con un push de compensación si el destino fallaba — cubría un destino del
tipo equivocado, pero no una muerte del proceso en medio.

`RPOPLPUSH mylist mylist`, el idiom de rotación, pasa por el mismo camino: una
clave, una lectura, una escritura. Tratarlo como dos claves prepararía dos
revisiones para la misma y la segunda escritura fallaría su propio
compare-and-swap.

Fijado en `tests/atomic_move.rs`, que **no** mata el proceso a propósito:
acertar el microsegundo entre dos escrituras depende del reloj, y una ejecución
verde no probaría nada. En su lugar demuestra el mecanismo — un registro, y un
registro truncado que reproduce como si el movimiento nunca hubiera ocurrido.
Escribir el arnés destapó de paso que la ruta de replay de la proyección redb
no tenía rama para el registro de lote: cada movimiento se habría perdido en el
primer arranque tras un corte.

### TLS

El puerto RESP habla texto plano por defecto, igual que Redis. Para cifrarlo:

```toml
resp_tls_enabled = true
# Opcional: si no se ponen, se usan tls_cert_path / tls_key_path
resp_tls_cert_path = "/etc/luma/resp.pem"
resp_tls_key_path  = "/etc/luma/resp.key"
```

La clave debe ser PKCS#8 (`openssl pkcs8 -topk8 -nocrypt`).

Es un interruptor explícito y no se deduce de que haya certificado: encender
HTTPS no debe cambiar en silencio el protocolo de otro puerto y romper a todos
los clientes conectados. Y con `resp_tls_enabled = true` sin certificado el
servidor **no arranca** — servir en claro porque faltaba un fichero es como las
credenciales acaban en la red mientras el operador cree que el puerto está
cifrado.

Sin TLS y con contraseña configurada, el arranque avisa: `AUTH` lleva la api
key de la organización, y en claro la tiene cualquiera en el camino.

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
de cluster (`MOVED`/`ASK`), replicación por protocolo Redis (`REPLICAOF`),
Streams (`XADD`), keyspace notifications.

**`EVAL`/`EVALSHA` existen, pero no hay Lua.** Esta línea decía que Lua estaba
fuera de alcance, y se quedó desactualizada: el trabajo de E2E encontró que el
worker de Celery moría con `unknown command EVALSHA`, y que la demanda real era
mucho más estrecha que «Lua» — es el `Lock` de redis-py, que usa kombu, y solo
sus `release`, `extend` y `reacquire` son scripts, cada uno seis líneas fijas.
`src/resp/scripts.rs` reconoce esos tres por su texto y los ejecuta de forma
nativa, sin intérprete. **Un script cualquiera se rechaza con un error que lo
dice**, que es el fallo honesto: un cliente con su propio Lua recibe un «no
soportado» en vez de una respuesta equivocada. Ahí está la costura por donde
entraría un intérprete de verdad si algún cliente objetivo lo necesitara.

---

## ioredis: tres bugs de protocolo que solo aparecen con un cliente Node

El README promete que ioredis funciona y nadie lo había ejecutado.
`tests/e2e/ioredis_client.mjs` lo ejecuta — 12 comprobaciones, contra Redis 7 y
contra Luma. Salieron **tres divergencias reales**, y ninguna la podía ver la
suite diferencial:

1. **`HELLO` ignoraba su cláusula `AUTH`.** `HELLO <proto> AUTH <user> <pass>` es
   como autentican ioredis, node-redis y redis-py en modo RESP3: un viaje en vez
   de dos. Se parseaba hasta la versión y nada más, así que la conexión quedaba
   sin autenticar y **todo comando posterior respondía `NOAUTH`** — ioredis no
   podía abrir una sola conexión a una instancia con contraseña. La resolución de
   la credencial es además asíncrona y vive en el listener, que solo la buscaba
   para `AUTH`; ahora `credential_in_command` conoce los dos comandos en un solo
   sitio, porque dos copias de ese conocimiento es como dejan de coincidir.
2. **La respuesta de `HELLO` era un mapa de RESP3 (`%7`)** en una conexión que esa
   misma respuesta declara `proto: 2`. Redis contesta `*14`, un array plano. El
   comentario del código advertía justo contra esa incoherencia y el código la
   cometía. ioredis desincronizaba su cola de comandos y moría con «Command queue
   state error» en cuanto llegaba un `message`. De paso el último campo era
   `name`, que no es un campo de `HELLO` en absoluto: Redis manda `modules`.
3. **`PING` dentro de una suscripción tiene otra forma.** Redis responde `+PONG`
   normalmente y `["pong", ""]` mientras hay suscripciones activas; Luma
   respondía `+PONG` siempre, e ioredis usa `PING` de keepalive en la conexión
   suscriptora. **El diferencial no podía cogerlo**: manda 327 comandos por una
   conexión que nunca se suscribe, y esto depende del *estado* de la conexión, no
   del comando.

Resultado: **11 de 12** comprobaciones de ioredis pasan.

### Lo que sigue fallando, dicho tal cual

`pub/sub with a second connection`: ioredis no emite el evento `message` aunque
`SUBSCRIBE` devuelve 1 y `PUBLISH` reporta 1 entregado. Las tramas de Luma en el
cable son **idénticas byte a byte a las de Redis** —comprobado con un socket
crudo por los dos caminos de autenticación— y el pub/sub sí funciona con redis-py
(`tests/e2e/clients.py`) y con sockets crudos. La causa no está identificada.
Queda abierto.

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
