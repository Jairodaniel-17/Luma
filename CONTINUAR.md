# Dónde retomar

Estado al cerrar la sesión del **2026-08-23**, sobre `v4.26.0` + los commits que
vinieron después. Este documento existe para no tener que reconstruir el contexto
leyendo el log de git.

---

## En una línea

Los tres SPEC están implementados y publicados. La sesión se fue en **11 bugs
reales** que estaban debajo de pruebas que miraban el sitio equivocado. Dos de
ellos perdían datos en silencio.

---

## Lo que quedó cerrado

### S3 — los tres huecos declarados, y siete bugs debajo

`docs/integrar/S3.md` declaraba tres cosas sin probar: la suite mint, las cargas
grandes y la concurrencia. Al cubrirlas salieron siete defectos, seis invisibles
para las 14 comprobaciones que ya existían.

| Qué estaba roto | Cómo se veía |
|---|---|
| Sin límite de cuerpo propio → heredaba **2 MiB** de axum | Conexión cerrada sin status ni log en cualquier subida mayor |
| **`Range` no se leía** | `GET` con rango devolvía el objeto entero; boto3 baja por rangos → descarga corrupta, sin un solo error |
| ETag compuesto no se persistía | `HEAD`/`GET` devolvían otro ETag que el completado |
| `Last-Modified` en ISO 8601 | No es un HTTP date; minio-py rechazaba la respuesta |
| `Content-Type` y `x-amz-meta-*` descartados | Un JPEG se sirve al navegador como descarga |
| Petición canónica firmada con método literal | `HeadBucket` → 403 indistinguible de clave mala |
| **`CopyObject` ni existía ni se rechazaba** | `aws s3 cp` dejaba el destino en 0 bytes y respondía éxito |

Ahora: **20/20 boto3**, **49/50 mint** (la que falla es `x-minio-extract`,
extensión propietaria de MinIO), un gigabyte real verificado byte a byte a
105 MiB/s, y concurrencia ejecutada en vez de razonada.

### RESP — cuatro bugs que solo aparecen con un cliente Node

El README prometía ioredis y nadie lo había ejecutado.

1. **`HELLO` ignoraba su cláusula `AUTH`** → `NOAUTH` en todo. ioredis no podía
   abrir una conexión a una instancia con contraseña.
2. **La respuesta de `HELLO` era un mapa RESP3 (`%7`)** declarando `proto: 2`.
3. **`PING` no cambiaba de forma dentro de una suscripción.** Y había un test de
   integración que **afirmaba el comportamiento equivocado**, sosteniendo el bug.
4. **Aceptar `HELLO 3` y contestar `proto: 2`.** ioredis pide RESP3, Redis
   contesta 3 y usa frames push (`>3`); Luma mandaba `*3` y el cliente leía el
   mensaje como respuesta a un comando pendiente. Ahora `HELLO 3` → `NOPROTO`.

Ninguno lo podía ver el diferencial de 327 comandos: tres dependen del **estado**
de la conexión o del handshake, no del comando.

Ahora: **12/12 ioredis**, 10/10 clientes Python, diferencial 327/327.

### El tope de objeto S3 estaba acoplado al global

`MAX_BODY_MB` gobierna el router `/v1`, blobs, búsqueda y el proxy. Prestárselo a
S3 hacía que para guardar un objeto de 1 GB hubiera que aceptar cuerpos JSON de
1 GB en `/v1/sql`. `S3_MAX_OBJECT_MB` es ahora su propio tope; sin poner, sigue a
`MAX_BODY_MB`.

---

## Lo que hay que hacer al retomar

### 1. Verificar el CI del último push

El commit que arregla `tests/resp_listener.rs` es lo último. **Comprobar que el
run está verde** antes de tocar nada:

```bash
gh run list --limit 3
```

El job `ioredis-e2e` es nuevo y entró en `all-checks`; es la primera vez que
corre en CI.

### 2. RESP: el criterio de GA es calendario

Para quitarle el flag *experimental* al listener RESP, el SPEC exige **nightly
verde 7 noches seguidas**. Estado: **1 de 7**. Solo ha habido una corrida
programada (2026-08-23 03:59), verde.

Se le añadió `ioredis` al nightly y a los `needs` del `summary` — antes no lo
corría, así que siete noches verdes habrían probado menos de lo que decían.

Dos caminos, y es decisión del dueño del proyecto:

- **Esperar las 7 noches** con el nightly ya completo.
- **Cambiar el criterio.** Se lo puso él mismo. Con 327 comandos byte a byte,
  12/12 ioredis, 10/10 Python y fuzz del parser, un criterio de 3 noches o de
  "el nightly completo verde una vez" es defendible. No se cambió por cuenta
  propia.

### 3. S3: lo que sigue sin probarse

- **Objetos por encima de `s3_max_object_mb`.** El cuerpo se bufferiza completo
  en memoria (`Bytes`), así que el tope existe a propósito. El S3 real admite
  partes de 5 GiB; igualarlo requiere **transmitir el cuerpo a disco**, que es un
  cambio de forma en `object_put` y `upload_part`, no una constante.
- **`CompleteMultipartUpload` ensambla en memoria.** Un objeto de un gigabyte
  cuesta un gigabyte de RAM en ese instante. Funciona —está medido— pero no
  escala.
- **Los ETag que un cliente devuelve en el completado no se verifican.** S3
  responde `InvalidPart`; aquí un cliente que mande los equivocados tiene éxito.
- **TLS en el puerto S3.** Va en claro, detrás de proxy o red de confianza.

### 4. Rendimiento: lo que ata ahora

El coste por escritura sube con el tamaño del valor (13.569 → 11.406 → 10.418/s a
10, 200 y 2.000 bytes) y eso apunta al **formato del evento**, no al
almacenamiento: el payload es un `serde_json::Value` que se clona a la cola, se
clona otra vez para el lote, se codifica a JSON para el WAL, se decodifica por
`StoredVal` y se re-codifica para la proyección.

No se tocó porque en la ruta de red actual ya no se notaría: `SET` va al 88% del
techo del transporte. Se notaría en proceso o en una red más rápida.

### 5. Conector Postgres (CDC): sin kilometraje

Sin ejercitar: TLS contra un servidor real, volumen (las pruebas mueven decenas
de filas), reconexión a media transacción, varios conectores sobre la misma base.
Solo protocolo lógico 1.

---

## Cómo levantar el entorno de pruebas

Los contenedores que hacen falta:

```bash
docker run -d --name luma-diff-redis -p 16379:6379 redis:7-alpine
docker run -d --name luma-cdc-pg -p 15432:5432 -e POSTGRES_PASSWORD=... postgres:16-alpine
```

Luma para las pruebas de S3 y de RESP (dos instancias, o una con los dos puertos):

```bash
env -u LUMA_API_KEY \
  DATA_DIR=/ruta/en/SSD \
  LUMA_API_KEY=una-clave-de-al-menos-16 \
  LUMA_MASTER_KEY=otra-clave \
  PORT_LUMA_VDB=18080 S3_PORT=19000 RESP_PORT=16380 RESP_ALLOW_FLUSH=1 \
  ./target/release/luma serve --bind 0.0.0.0
```

`--bind 0.0.0.0` solo hace falta para que el contenedor de mint alcance el host.
`RESP_ALLOW_FLUSH=1` lo necesita `clients.py`, que usa `FLUSHDB` para limpiar
entre comprobaciones — sin él **falla y el error no dice que es eso**.

Las suites:

```bash
# S3
python tests/e2e/s3_client.py  --admin http://127.0.0.1:18080 --s3 http://127.0.0.1:19000 --api-key <clave>
python tests/e2e/s3_chunked.py --admin ... --s3 ... --api-key ...
cd tests/e2e && python s3_scale.py --admin ... --s3 ... --api-key ... --size-mb 64

# mint (a mano; imagen de más de 1 GB, no está en CI)
docker run --rm -e SERVER_ENDPOINT=host.docker.internal:19000 \
  -e ACCESS_KEY=<ak> -e SECRET_KEY=<sk> -e ENABLE_HTTPS=0 \
  -e MINT_MODE=core minio/mint:latest minio-py

# RESP
npm install ioredis
node tests/e2e/ioredis_client.mjs --redis redis://127.0.0.1:16379/0 \
  --luma redis://:<clave>@127.0.0.1:16380/0
python tests/e2e/clients.py --redis redis://127.0.0.1:16379/0 \
  --luma redis://:<clave>@127.0.0.1:16380/0
LUMA_DIFF_REDIS=127.0.0.1:16379 cargo test --test redis_differential -- --ignored
```

---

## Lecciones de esta sesión, para no repetirlas

**Correr la suite entera antes de pushear, no solo el módulo tocado.** El arreglo
de `PING` rompió `tests/resp_listener.rs` y se fue a CI en rojo dos veces. El
módulo (`cargo test --lib resp::`) estaba verde; el test de integración no se
corrió.

**Un test puede sostener un bug.** `a_subscribed_connection_still_answers_commands`
afirmaba `+PONG` dentro de una suscripción, que es lo que Redis **no** hace. Y
`a_multipart_etag_has_the_dash_and_part_count` leía el ETag solo de la respuesta
de completado, nunca del servidor después. Los dos pasaban con el defecto puesto.

**Los clientes reales encuentran lo que las suites propias no.** Los 11 bugs
salieron de ejecutar boto3, mint e ioredis. Nada salió de leer el código.

**Para un bug de protocolo, poner un proxy que registre el cable.** El último de
ioredis (`HELLO 3`) no se veía de ninguna otra forma; la conversación lado a lado
contra Redis lo dijo en una línea. El script está en el scratchpad de la sesión;
son 40 líneas y merece la pena reescribirlo si vuelve a hacer falta.

**Un dato del entorno que costó tiempo:** hay un `LUMA_API_KEY` exportado en el
entorno de la máquina de desarrollo. Hace que el servidor arranque sin quejarse
aunque no se pase clave, y que `dev` dé `WRONGPASS`. Conviene rotarlo y no
depender de él.

---

## Referencias

| Documento | Qué contiene |
|---|---|
| [`MANUAL_USUARIO.md`](MANUAL_USUARIO.md) | Qué soporta Luma y cómo configurarlo. §1.1 explica por qué cada superficie es experimental |
| [`docs/integrar/S3.md`](docs/integrar/S3.md) | «Lo que las pruebas encontraron»: los siete bugs con su detalle |
| [`docs/integrar/RESP.md`](docs/integrar/RESP.md) | «ioredis: tres bugs de protocolo…» más el cuarto |
| [`docs/referencia/BENCHMARKS.md`](docs/referencia/BENCHMARKS.md) | El camino de escritura capa por capa, y los dos diseños descartados con números |
| [`docs/operar/CONFIG.md`](docs/operar/CONFIG.md) | Las 122 claves con sus defaults verificados |
| [`docs/SPEC-resp.md`](docs/SPEC-resp.md) | El criterio de GA del listener RESP (F4.5) |
