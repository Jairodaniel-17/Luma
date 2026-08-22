# Runbooks

W5.5 de `docs/PLAN-MAESTRO.md`. Procedimientos para operar Luma, escritos para
leerse **durante** un incidente: los pasos primero, el razonamiento después.

Todos los comandos de este documento se han ejecutado contra el binario real. Un
runbook con un flag inventado es peor que ninguno, porque se descubre en el peor
momento.

> **Estado honesto de este documento.** El criterio de aceptación de W5.5 es que
> *alguien externo monte Luma con réplica y respaldo remoto usando solo los
> docs*. Eso no se ha probado con una persona todavía. Hasta que se haga, esto es
> lo mejor que sabemos, no algo verificado por un lector.

---

## 0. Antes de nada: qué está y qué no

| | Estado |
|---|---|
| Respaldo local y restore | Verificado, con `--verify` |
| Respaldo remoto (S3/R2/GCS/MinIO) | Implementado; **SigV4 contra MinIO real sin probar** |
| Réplica de lectura | Verificada en tests; promoción manual |
| Failover automático | **No existe.** W2.3 en el plan |
| Fencing por epoch contra split-brain | Implementado. Ventana de un intervalo de envío, ver §3 |
| Puerto RESP | Experimental. Ver `docs/integrar/RESP.md` |

---

## 1. Respaldo y restauración

### Hacer un respaldo

```bash
luma backup --verify
```

`--verify` no es opcional en la práctica. Restaura el respaldo a un temporal,
corre `PRAGMA integrity_check` sobre SQLite y compara los conteos contra el
manifiesto. **Un respaldo que nadie ha restaurado es una hipótesis**, y el
momento de descubrir que está roto no es cuando lo necesitas.

Salida esperada:

```
Backup creado en backups/20260822T125301Z
Verificado: sqlite=true snapshot=true wal=3 colecciones=2 blobs=14 mensajes=0
```

Si `wal=0` **y** `snapshot=false`, el respaldo está vacío: comprueba que
`data_dir` apunta a donde crees.

### Qué contiene, y qué no

Contiene: la base SQLite (vía `VACUUM INTO`, así que es consistente), el
`snapshot.json`, los segmentos del WAL, `vectors/`, `blobs/` y `queues/`.

**No contiene `state.redb`**, a propósito. Es una proyección del WAL: el restore
la reconstruye reproduciendo. Copiarla en caliente falla en Windows con violación
de compartición y en Linux produce una lectura desgarrada — y una redb desgarrada
es peor que ninguna, porque el restore arrancaría desde estado derivado corrupto
en vez de reconstruirlo limpio.

### Restaurar

```bash
# 1. Para el servidor. El restore escribe sobre el data_dir.
#    (systemd: systemctl stop luma)

# 2. Restaura.
luma restore backups/20260822T125301Z

# 3. Arranca. El arranque reproduce el WAL y reconstruye redb.
luma serve
```

El primer arranque tras un restore es más lento de lo normal: está reproduciendo.
En el log verás `replayed wal events applied=N`.

### Respaldo remoto

En `luma.toml`:

```toml
backup_enabled = true
backup_interval_secs = 3600
backup_retention = 7
backup_remote_url = "s3://mi-bucket/luma"   # o r2://, gs://, minio://
```

Las credenciales van por entorno (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION`, o el equivalente del proveedor), nunca en el fichero.

El manifiesto se sube **al final**, después de todos los objetos. Un `download`
que no encuentra manifiesto rechaza el prefijo en vez de restaurar a medias: si
la subida se cortó, no hay manifiesto, y el prefijo se descarta entero.

---

## 2. Réplica de lectura

### Montar una réplica

Las dos mitades usan **el mismo destino remoto** (`backup_remote_url`); lo que
las distingue es el intervalo que cada una tiene puesto y el marcador de rol.

En el primario, `luma.toml`:

```toml
backup_remote_url = "s3://mi-bucket/luma"
# Este intervalo ES el objetivo de punto de recuperación: un host perdido entre
# ticks pierde como mucho un intervalo de escrituras. 0 lo desactiva.
wal_ship_interval_secs = 10
```

En la réplica, `luma.toml`:

```toml
backup_remote_url = "s3://mi-bucket/luma"
replica_poll_interval_secs = 10
```

Y marcar el directorio como réplica, que es lo que hace que se comporte como tal:

```bash
luma demote
# → Marcado como réplica de solo lectura: /var/lib/luma
#   Las escrituras se rechazan. `luma promote` lo revierte.

luma role
# → replica
```

El rol vive en un **marcador en disco** (`REPLICA` dentro del `data_dir`), no en
la configuración: sobrevive a un reinicio y no depende de que alguien recuerde
pasar un flag. `replica_poll_interval_secs` solo se consulta si el marcador está
presente.

Una réplica **rechaza escrituras**. El snapshot se escribe antes que los
segmentos: al revés, un corte dejaría segmentos sin la base sobre la que se
aplican.

### Cuánto va por detrás

`GET /v1/metrics` expone los bytes seguidos. El retraso es, como mucho, un
intervalo de `wal_ship_interval_secs` más el de `replica_poll_interval_secs`.
Ese es el número que va en un SLA, no un promedio.

---

## 3. Promover una réplica

> **Hay fencing por epoch, y aun así para el primario antiguo primero.** Al
> promover, el nodo nuevo reclama la siguiente *epoch* en el prefijo remoto. El
> primario antiguo la lee en su siguiente pasada de envío y **deja de escribir**.
>
> La ventana es un intervalo de `wal_ship_interval_secs`, no cero. Dentro de esa
> ventana los dos pueden escribir y sus segmentos se entrelazan, lo que no se
> arregla después. Por eso el paso 1 sigue siendo pararlo: el fencing es la red,
> no el plan.
>
> Cerrar la ventana del todo requiere un lease con quórum real, que es un sistema
> de consenso; el plan lo mantiene en backlog con criterio de entrada explícito.

```bash
# 1. PARA el primario antiguo. De verdad. Comprueba que el proceso murió.
#    systemctl stop luma  &&  systemctl is-active luma   → inactive

# 2. En la réplica, quita el marcador.
luma promote
# → Promovido a primario: /var/lib/luma
#   Asegúrate de que el primario anterior está detenido antes de que vuelva a escribir.

# 3. Comprueba el rol y arranca.
luma role      # → primary
luma serve

# 4. Comprueba que el nuevo primario se anuncia como tal.
curl -s -o /dev/null -w '%{http_code}
' localhost:1234/v1/health/primary
# → 200 en un primario, 503 en una réplica

# 5. Reapunta a los clientes.
```

El paso 4 es el health-check que debe usar un proxy: responde con un **código de
estado**, no con un campo dentro de un cuerpo, porque un balanceador puede rutar
sobre eso sin parsear JSON. Una réplica contesta **503** y no 404: el endpoint
existe y el nodo está sano, simplemente no es el que acepta escrituras — un 404
se leería como un proxy mal configurado.

Si `luma promote` no puede alcanzar el destino remoto, lo dice y **no falla**: la
promoción local ya ocurrió, así que devolver un error sugeriría que nada cambió.
En ese caso el primario antiguo NO está cercado y hay que pararlo a mano.

`luma promote` sobre algo que ya es primario **falla**, no reporta éxito:

```
Error: /var/lib/luma is not a replica: nothing to promote
```

Eso es deliberado. Un "promovido" sobre el directorio equivocado a mitad de un
incidente te diría que has hecho algo que no has hecho.

---

## 4. Rotar la master key

`LUMA_MASTER_KEY` cifra los secretos en reposo (ChaCha20-Poly1305). El
ciphertext es auto-descriptivo (`enc:v1:…`), así que la rotación no es un
intercambio atómico.

> **No hay reencriptado automático.** Cambiar la variable y reiniciar deja los
> secretos existentes ilegibles. El procedimiento es descifrar con la clave
> vieja y volver a escribir con la nueva.

```bash
# 1. Respalda primero. Este procedimiento reescribe secretos.
luma backup --verify

# 2. Para el servidor.

# 3. Arranca con la clave VIEJA y exporta lo que esté cifrado
#    (api keys, credenciales de proveedor) por la API de admin.

# 4. Cambia LUMA_MASTER_KEY a la nueva.

# 5. Arranca y vuelve a escribir esos secretos.

# 6. Comprueba que un login y una api key existente siguen funcionando
#    ANTES de destruir la clave vieja.
```

Guarda la clave vieja hasta haber confirmado el paso 6. Si algo quedó cifrado con
ella y se descarta, ese dato se perdió.

---

## 5. Actualizar

```bash
# 1. Respalda y verifica.
luma backup --verify

# 2. Lee el CHANGELOG entre tu versión y la nueva. Busca cambios de formato
#    en disco.

# 3. Para, sustituye el binario, arranca.

# 4. Confirma que el arranque reprodujo sin corrupción:
#    en el log, `replayed wal events ... corrupted=0`
```

El formato en disco es compatible hacia atrás por diseño y hay un fixture dorado
en CI (`tests/golden_data_dir.rs`) que comprueba que una versión nueva lee lo que
escribió la anterior. Aun así: respalda primero.

**Volver atrás** no está garantizado. Una versión vieja leyendo un WAL escrito
por una nueva puede encontrar tipos de registro que no conoce, y un tipo
desconocido se ignora — es decir, se pierde. Si necesitas volver, restaura el
respaldo.

---

## 6. Cuando algo va mal

| Síntoma | Primer sitio donde mirar |
|---|---|
| El servidor no arranca | El log. Si dice `refusing to start with N insecure secret settings`, faltan `LUMA_MASTER_KEY` o `LUMA_API_KEY` |
| Arranca en el puerto equivocado | La precedencia es flags > entorno > `luma.toml` > defaults. El log dice `listening addr=…` |
| Escribe en el directorio equivocado | El log dice `Data Directory: …`. Comprueba `DATA_DIR` en el entorno, que gana al fichero |
| Un cliente Redis no conecta | ¿Está `resp_port` puesto? Por defecto es 0, que significa apagado |
| `WRONGPASS` por RESP con una api key válida | La clave tiene que existir en `sys_api_keys`. La clave estática de la instancia también vale, y da acceso de plataforma |
| El panel no muestra comandos/s | Necesita dos lecturas. Espera un intervalo de sondeo |
| El WAL crece sin parar | Los snapshots rotan el WAL. Comprueba `snapshot_interval_secs` y busca `snapshot ok` en el log |
| `nesting too deep` por RESP | Un frame con más de 32 niveles de anidamiento. Es un rechazo deliberado: sin él, 40 KB de `*1` desbordaban la pila y mataban el proceso |

### Dos cosas que este documento provocó

Escribir estos pasos hizo aparecer dos huecos, ambos arreglados antes de
publicarlo:

- **No había `luma --help`.** Cualquier subcomando mal escrito caía a `serve`, así
  que `luma backupp` arrancaba un servidor en el puerto de producción. Ahora es
  un error que nombra el typo.
- **No se podía crear una réplica.** `promote` existía y su par no: `mark_replica`
  vivía en el crate y solo lo llamaban los tests. Una réplica que nadie puede
  crear no es una funcionalidad. Ahora hay `luma demote`.

Los dos se descubrieron por escribir el procedimiento y comprobar cada comando
contra el binario, no leyendo el código.

### Lo que el log dice al arrancar

Con `RUST_LOG=info` (sin él, solo se ven errores):

```
[config] max_body_mb = 100.0000        ← límites efectivos, tras la precedencia
Data Directory: /var/lib/luma          ← comprueba que es el que esperas
replayed wal events applied=0 duplicates=2 corrupted=0
RESP listener started port=6379 auth=true tls=false
listening addr=127.0.0.1:1234
```

`corrupted=N` con N > 0 significa que el replay se detuvo en un registro dañado.
Todo lo anterior a él se aplicó; nada posterior. Eso es intencionado: aplicar
registros posteriores encima de estado que nunca se construyó es divergencia
silenciosa, y es peor que detenerse y decirlo.
