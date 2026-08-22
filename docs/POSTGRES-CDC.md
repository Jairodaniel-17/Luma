# Conector Postgres — informe de decisión (W4.1)

> Criterio de salida del spike, tal como lo fijó `docs/PLAN-MAESTRO.md`:
> *«informe de decisión (crate existente vs subset propio) antes de diseñar
> nada»*. Esto es ese informe, y lo que hay debajo de él son 7 pruebas contra un
> Postgres 16 real, no una lectura de documentación.

## La decisión de producto que esto implementa

**Luma no reemplaza Postgres, se conecta a él.** Postgres sigue siendo la fuente
de verdad transaccional; lo que fluye por aquí es una copia derivada, con forma
de índice de búsqueda. Por eso nada en `src/pgcdc/` escribe de vuelta, y por eso
un hit federado lleva la tabla y la clave primaria de origen: para que la
aplicación vaya a leer la fila canónica donde de verdad vive.

## La pregunta

¿Se usa una crate existente para la replicación lógica, o se escribe el
subconjunto que hace falta?

## Lo que se encontró

| Candidato | Estado |
|---|---|
| **`postgres-replication`** — la crate del workspace de rust-postgres que hace exactamente esto | **No está publicada.** Existe solo en la rama master del repositorio |
| **`tokio-postgres` 0.7.18** (la versión publicada) | No tiene `ReplicationMode` ni `copy_both_simple`; ambas son de master. Su parser de cadena de conexión **rechaza `replication=database`** como clave desconocida, así que no puede ni abrir el tipo de conexión correcto. Su parser de mensajes tampoco conoce la etiqueta `CopyBothResponse` |
| **`pg_replicate`** | 0.1.0 |
| **`rustcdc`** | Arrastra `wasmtime`, `mysql_async` y `tiberius` por features por defecto |

Verificado leyendo la fuente vendorizada, no la documentación:
`postgres-protocol` 0.6.12 no tiene ningún módulo de replicación, y el enum
`Message` de `backend.rs` no incluye `CopyBothResponse`.

## La decisión

**Subconjunto propio**, en `src/pgcdc/`.

La elección real nunca fue «crate contra código propio» — fue **«dependencia git
sobre una rama sin publicar contra código propio»**. Una revisión git fijada es
una dependencia de la que nadie puede auditar una versión, y `cargo deny` no
tiene nada contra qué comprobarla. Para un binario único que ya pasa por
`cargo audit`, `cargo deny` y `#[forbid(unsafe_code)]`, eso es peor que el
código.

Y `rustcdc` compra la funcionalidad al precio de una superficie de suministro
varias veces mayor que la funcionalidad misma.

### Lo que sí se delega

**SCRAM-SHA-256 viene de `postgres-protocol`**, que sí está publicada. La
autenticación es la única parte de esto donde un error sutil es un fallo de
seguridad y no un error de parseo. Incluye la verificación de la firma del
servidor: sin ella habríamos probado quiénes somos a quien contestara, y no
habríamos aprendido nada sobre quién era.

### Lo que se escribió

| | |
|---|---|
| `pgoutput.rs` | El formato de mensajes. Pequeño y estable: un archivo cubre la versión 1 entera |
| `conn.rs` | Arranque con `replication=database`, autenticación, consulta simple, y el dúplex COPY-BOTH |
| `slots.rs` | Publicación, slot, y las comprobaciones que hacen visibles sus modos de fallo |

`tokio-postgres` **sí** se usa, para lo que sirve: el backfill por `COPY` y las
consultas de catálogo. Lo que no puede es replicar.

## Lo que el servidor real corrigió

Dos suposiciones mías, las dos escritas primero como aserción y las dos falsas.
Se dejan registradas porque las dos cambian el diseño de W4.2:

**1. Una conexión de replicación lógica *sí* acepta SQL corriente.** Escribí la
prueba afirmando que no. `replication=database` abre un walsender atado a una
base de datos, y esa conexión acepta tanto comandos de replicación como
consultas. Es `replication=true` —el modo físico— el que rechaza SQL. Consecuencia
para W4.2: una sola conexión puede hacer las comprobaciones de catálogo y luego
transmitir. La línea que no se puede cruzar es `START_REPLICATION`, no el modo.

**2. Un valor largo no es un valor TOASTed.** La prueba del TOAST elidido usaba
`repeat('x', 12000)` y la columna llegaba entera: 12 kB de un mismo carácter se
comprimen a casi nada, así que Postgres lo guardó *inline* y no tenía razón para
omitirlo. Solo un valor almacenado **fuera de línea** se elide del tuple del WAL.
La prueba ahora fija `STORAGE EXTERNAL`, que lo hace determinista en vez de una
propiedad de los datos de ejemplo.

Esa segunda es la que importa de verdad, porque protege contra pérdida de datos:
una columna grande que no cambió se **omite** del UPDATE, y un consumidor que lea
la omisión como NULL destruye un valor que Postgres sigue teniendo y no vuelve a
mencionar. `Value::Unchanged` y `Value::Null` son variantes distintas por eso.

## Los modos de fallo que el conector hace visibles

Los tres son silenciosos por naturaleza: el stream funciona perfectamente
mientras ocurren.

| | |
|---|---|
| **El slot que llena el disco** | Un slot de replicación es la promesa de Postgres de guardar WAL hasta que alguien lo lea. Un consumidor que transmite y nunca manda un *standby status update* fija cada segmento desde que se creó el slot. Por eso `send_standby_status` existe y por eso hay una prueba que verifica que `confirmed_flush_lsn` se mueve de verdad |
| **La tabla sin identidad de réplica** | `REPLICA IDENTITY DEFAULT` sin clave primaria replica los INSERT perfectamente, y Postgres luego rechaza cada UPDATE y DELETE con un error que nombra la publicación, no la clave que falta. Se comprueba **al configurar**, no en el primer UPDATE |
| **El system id equivocado** | Es lo único que distingue un servidor de una copia restaurada de sí mismo. Reanudar desde un LSN guardado contra otro sistema aterriza en una posición perfectamente plausible del WAL de otro |

## Decisiones menores, con su razón

- **`sslmode=prefer` se rechaza.** Es el ajuste que parece seguro en un fichero
  de configuración y no lo es: si el servidor declina TLS, el stream —que es el
  contenido entero de la base de datos— sigue en claro y nada lo reporta. Solo
  `require` y `disable`.
- **Autenticación md5 y cleartext se rechazan** con un mensaje que dice qué
  cambiar. md5 está superado desde Postgres 10.
- **Las publicaciones nombran tablas explícitamente**, nunca `FOR ALL TABLES`:
  eso significaría también cada tabla futura, incluida alguna que nadie pensaba
  exponer a un índice de búsqueda.
- **Los valores llegan como texto**, no convertidos. `42.00` en una columna
  `numeric` es exactamente eso; un viaje por `f64` lo convertiría en casi eso.
- **Hay techo al tamaño de mensaje** (256 MiB). La longitud es un número que
  elige el par, y sin techo un par corrupto u hostil es una reserva de hasta
  4 GiB.

## Cómo se verifica

```bash
docker run -d --name luma-cdc-pg \
  -e POSTGRES_PASSWORD=luma -e POSTGRES_USER=luma -e POSTGRES_DB=luma \
  -p 15432:5432 postgres:16-alpine \
  -c wal_level=logical -c max_replication_slots=8 -c max_wal_senders=8

LUMA_PG_URL="postgres://luma:luma@127.0.0.1:15432/luma?sslmode=disable" \
  cargo test --test pgcdc_stream -- --ignored --test-threads=1
```

Sin `LUMA_PG_URL` la suite se niega a correr en vez de pasar en vacío: una suite
que se pone verde cuando su sujeto no está reporta una cobertura que no tiene.

## Lo que este spike **no** establece

- **Sin TLS contra un servidor real.** El camino existe (`sslmode=require`,
  SSLRequest, rustls con las raíces de webpki) pero el Postgres del contenedor
  no sirve TLS, así que solo está probado el rechazo de `prefer`.
- **Sin reconexión.** El stream se abre y se lee; qué pasa cuando se corta a
  media transacción es trabajo de W4.2.
- **Sin volumen.** Las pruebas mueven filas sueltas.
- **Solo protocolo 1.** Las versiones 2–4 añaden streaming de transacciones
  grandes antes del commit. No hacen falta hasta que una transacción grande sea
  un problema, y entonces serán un cambio localizado en `pgoutput.rs`.
