# SPEC — Luma como producto sólido (plan maestro, post v4.24.0)

Plan maestro para llevar Luma de "motor convergente funcional" a **producto que
un tercero opera con confianza y adopta sin fricción**. Agrupa 5 frentes (W1–W5)
que avanzan en paralelo pero con dependencias explícitas. El frente de
compatibilidad Redis tiene su propio SPEC detallado
([`SPEC-resp.md`](SPEC-resp.md)) y aquí solo se referencia.

Mismo flujo probado: rama → commit → tag → CI verde (rustfmt/clippy/test) →
build musl → deploy → verificación E2E. Convención: `[ ]` · `[~]` · `[x]`.

## La tesis

Luma ya tiene la superficie (blob, KV, colas, eventos, imágenes, doc store,
vectores, NS-Mem, SQL, cuentas — ver README «Superficie de plataforma»). Lo que
separa eso de un producto son cuatro cosas, en este orden:

1. **Que no pierda datos ni estando solo** (durabilidad verificada, restore probado).
2. **Que sobreviva a perder la máquina** (réplica, punto-en-el-tiempo).
3. **Que se adopte sin reescribir al cliente** (protocolos que el mundo ya habla:
   RESP, S3; y conexión al Postgres que ya existe, no su reemplazo).
4. **Que se opere a ciegas sin miedo** (métricas, alertas, límites, docs, benchmarks honestos).

El orden importa: nadie apunta su Celery ni sus documentos a un almacén que no
puede demostrar 1 y 2.

## Definición de producto

Lo que Luma **es** y para quién, para que cada ítem técnico tenga un norte:

- **Qué es:** plataforma de datos convergente para aplicaciones de IA — las
  primitivas que rodean a la base transaccional (blob, KV, colas, eventos,
  vectores, memoria de agentes) en un binario, con compatibilidad de
  protocolos estándar (RESP, S3) como interfaz de adopción.
- **Qué no es:** un reemplazo de PostgreSQL (W4 existe para lo contrario), ni
  un Redis clusterizado, ni una base distribuida multi-escritor.
- **Usuario objetivo, por orden:** (1) las aplicaciones propias del autor como
  primer cliente exigente (dogfooding con los 5 pilotos), (2) equipos pequeños
  que hoy pagan 4–6 servicios gestionados para una app de IA y quieren un solo
  binario operable, (3) despliegues on-premise/edge donde los servicios
  gestionados no llegan.
- **Licencia:** MIT (ya vigente en `LICENSE`). Decisión pendiente de negocio
  `[DECISIÓN]`: si al abrirse a terceros conviene mantener MIT puro o mover a
  licencia dual / BSL para proteger un futuro servicio gestionado. No bloquea
  nada de este SPEC; revisar antes del primer cliente externo.
- **Promesa de soporte** (desde GA): API v1 congelada (rupturas ⇒ v2);
  correcciones de seguridad sobre la última minor; los datos siempre migran
  hacia adelante (ver política de compatibilidad); ventana de deprecación
  mínima de una minor con aviso en CHANGELOG. `[DECISIÓN]` SLA formal de
  soporte solo si aparece un cliente externo de pago.

## Política de compatibilidad de datos entre versiones

Regla general, obligatoria para todo ítem de este SPEC y los siguientes:

1. **Toda versión N lee los datos escritos por N−1** (WAL, redb, snapshots,
   SQLite, blobs, backups). Los registros nuevos se introducen **versionados y
   etiquetados** (patrón D5 de SPEC-resp: lo viejo se lee con el significado
   viejo; nunca se reinterpreta).
2. **Migraciones de formato: automáticas al arrancar, nunca destructivas.**
   Si una migración reescribe datos, primero exige un backup verificado
   (1.4) o crea uno propio; el formato anterior no se borra hasta confirmar
   la reescritura (write-new → verify → swap).
3. **Downgrade:** no se garantiza N−1 leyendo datos de N; el camino de vuelta
   es siempre restaurar el backup previo al upgrade. El runbook de upgrade
   (5.5) lo dice en su primer paso: backup verificado antes de actualizar.
4. **CI lo vigila:** test de arranque con un `data_dir` dorado generado por la
   última release publicada (fixture versionada) — si la rama actual no lo lee
   íntegro, el build falla. Se agrega el fixture en W1 y se regenera por release.
5. Un cambio que no pueda cumplir 1–4 no entra en v1: espera a v2.

## Mapa de frentes y dependencias

```
W1 Durabilidad ──► W2 Réplica/DR ──► (habilita vender producción)
W1 ──► W3 Protocolos (RESP · S3) ──► adopción drop-in
W1 ──► W4 Conector Postgres (CDC) ──► adopción junto a un ERP/stack existente
W1+W2 ──► W5 Operabilidad & GA
```

---

## W1 — Durabilidad verificada (releases **v4.25.x**)
Hoy: WAL segmentado con checksums + snapshots + redb + `luma backup/restore`
(SQLite + snapshot + WAL a directorio con retención). Es una base correcta;
falta convertirla en garantía demostrada.

### 1.1 `[ ]` Matriz de crash-recovery en CI  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** demostrar, no suponer, que ningún kill -9 pierde datos confirmados.
**Enfoque:** harness que mata el proceso en puntos aleatorios durante ráfagas
de escritura de **cada** motor (KV, blob, colas, doc, vector, memoria, SQLite)
y verifica al reinicio: prefijo confirmado íntegro, `wal_replay_corrupt_total`
en métrica, cero divergencia entre memoria y redb (offset aplicado).
**Aceptación:** job nightly, 500 iteraciones por motor, verde 7 días seguidos.
Es el mismo harness que exige SPEC-resp F0.2 — se construye una vez.

### 1.2 `[ ]` fsync por motor auditado  · impacto ALTO · esfuerzo BAJO
**Objetivo:** política de sync explícita y documentada por primitiva.
**Enfoque:** auditar `wal_sync_mode` (always/group) extremo a extremo; las
colas escriben archivos JSON por mensaje — garantizar fsync del archivo y del
directorio en enqueue confirmado; blob igual (write + rename atómico + fsync).
**Aceptación:** tabla en `docs/PROD_READINESS.md`: primitiva → cuándo devuelve
OK → qué garantiza en disco. Test de corte de energía simulado (dm-flakey o
kill de contenedor) para colas y blob.

### 1.3 `[ ]` Backup remoto a objeto storage  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** el backup no puede vivir solo en el mismo disco que los datos.
**Enfoque:** destino S3-compatible para `luma backup` (crate `object_store`:
S3/R2/GCS/MinIO/OCI con una sola API), cifrado del artefacto con la master key,
`luma restore s3://…` directo, retención remota.
**Aceptación:** ciclo completo backup→borrar data_dir→restore desde bucket→
suite de integridad verde; documentado como runbook.

### 1.4 `[ ]` Verificación de restore automática  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** un backup no verificado no existe.
**Enfoque:** `luma backup --verify`: restaura a un directorio temporal, abre en
modo lectura, corre checks de integridad y reporta; la tarea de fondo lo hace
en cada N backups y expone `backup_last_verified_ts` en métricas.
**Aceptación:** alerta ejemplo (Prometheus rule) incluida en docs.

---

## W2 — Réplica y recuperación ante desastre (releases **v4.26.x–v4.27.x**)
Sin inventar un protocolo de consenso: primero replicar lo que ya existe (WAL
con offsets totales + segmentos), consenso solo si algún día hay quórum real.

### 2.1 `[ ]` WAL shipping continuo a objeto storage  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** RPO de segundos sin segundo servidor (estilo Litestream).
**Enfoque:** tarea de fondo que sube cada segmento WAL cerrado (y parciales
cada `ship_interval`) al bucket de 1.3; snapshot + cadena de segmentos = punto
en el tiempo. `luma restore --to-timestamp` reconstruye replay hasta ahí.
Incluye el WAL de SQLite (o checkpoint + copia si el modo WAL de SQLite no
coopera — decidir con un spike primero).
**Aceptación:** demo reproducible: escribir 10k ops, destruir la máquina,
restaurar en otra desde el bucket con pérdida ≤ `ship_interval`; métrica
`wal_ship_lag_seconds`.

### 2.2 `[ ]` Réplica de lectura caliente  · impacto ALTO · esfuerzo ALTO
**Objetivo:** un segundo proceso Luma que sigue el WAL (del bucket o por
streaming HTTP `/v1/replica/stream` autenticado) y sirve **solo lecturas**.
**Enfoque:** el replay ya existe (arranque = replay); convertirlo en replay
continuo con offset de avance expuesto. Sin failover automático en esta fase:
promoción manual documentada (`luma promote`).
**Aceptación:** réplica sirve `GET`/search con lag < 2 s bajo carga de
escritura sostenida; promoción manual probada en runbook; lecturas en réplica
rechazan escrituras con error claro.

### 2.3 `[ ]` Failover asistido  · impacto MEDIO · esfuerzo MEDIO
**Objetivo:** recuperación en minutos sin sorpresas, honesta (no HA automática).
**Enfoque:** health-check estándar para poner primaria/réplica detrás de un
proxy (nginx/ALB), documento de secuencia de promoción, protección contra
split-brain simple: la primaria vieja rehúsa arrancar si el bucket tiene un
epoch más nuevo (fencing por epoch en metadata remota).
**Aceptación:** simulacro completo documentado con tiempos medidos.
**Backlog explícito:** consenso/quórum (Raft) solo si aparece el caso de uso
multi-escritor; no antes.

---

## W3 — Protocolos estándar (adopción drop-in)

### 3.1 `[ ]` RESP (protocolo Redis) — **SPEC propio**
Ver [`SPEC-resp.md`](SPEC-resp.md) (fases F0–F4, releases v4.25–v4.29).
Es el frente de mayor palanca: Celery/arq/redis-py sin cambiar código.

### 3.2 `[ ]` API S3-compatible sobre blob  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** que boto3 / aws-sdk-js / rclone apunten a Luma como endpoint S3
(igual que MinIO/R2), sin SDK propio.
**Enfoque:** subconjunto que usan los SDKs: `PUT/GET/HEAD/DELETE Object`,
`ListObjectsV2`, `CreateBucket/ListBuckets`, multipart upload (init/part/
complete/abort), presigned URLs, SigV4 (crear credenciales AWS-style por org
mapeadas a las api keys). XML de respuesta idéntico al de S3 — validar con la
suite de conformidad de MinIO (mint) recortada.
**Aceptación:** boto3 sube/lista/descarga/borra y hace multipart de 100 MB
contra Luma sin configuración especial más allá de `endpoint_url`; rclone
sync funciona. Para cualquier app con adaptador «S3-compatible» la adopción se
vuelve literal: cambiar `endpoint_url`.

### 3.3 `[ ]` OpenAPI como contrato  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** `docs/openapi.yaml` generado desde el código (no mantenido a
mano), SDKs Python/TS regenerables, versionado semántico de la API v1.
**Aceptación:** CI falla si el spec generado difiere del commiteado; SDKs
publican junto a cada release.

---

## W4 — Conector PostgreSQL (CDC): completar, no competir (releases **v4.28.x**)
La decisión estratégica: Luma **no** reemplaza Postgres; se conecta a él.
Postgres queda como fuente de verdad transaccional; Luma es el plano de datos
de IA (vectores, búsqueda, memoria, eventos) que se mantiene sincronizado solo.

### 4.1 `[ ]` Spike: replicación lógica en Rust  · impacto ALTO · esfuerzo BAJO
**Objetivo:** validar la vía técnica antes de comprometer diseño.
**Enfoque:** prototipo con el protocolo de replicación lógica de Postgres
(`pgoutput`) leyendo un slot: crear publicación, consumir INSERT/UPDATE/DELETE,
confirmar LSN. Evaluar crates existentes vs implementación propia del subset.
**Aceptación:** demo: tabla `documents` en PG → cada fila aparece como
documento en Luma < 1 s; informe de decisión en `docs/` (crate elegido o propio).

### 4.2 `[ ]` `luma connect postgres`  · impacto ALTO · esfuerzo ALTO
**Objetivo:** configuración declarativa de sincronización.
**Enfoque:** en `luma.toml`: fuente (`postgres://…`, slot, publicación) +
mapeos: tabla → colección de documentos / colección vectorial (columnas a
embeber con el `embedding_provider` ya existente) / namespace de memoria.
Reanudación por LSN persistido (mismo patrón `applied_offset` de redb),
backfill inicial por `COPY`, manejo de esquema: columnas nuevas se ingieren,
tipos no mapeables se registran y saltan (no rompen el stream).
**Aceptación:** E2E contra Postgres 16 en CI: backfill de 100k filas + stream
en vivo + kill del conector a mitad → reanuda sin duplicar ni perder (idempotencia
por PK+LSN); lag expuesto en `/v1/metrics` (`cdc_lag_bytes`, `cdc_last_lsn`).

### 4.3 `[ ]` Búsqueda federada mínima  · impacto MEDIO · esfuerzo MEDIO
**Objetivo:** el valor visible del conector: preguntar en un solo lugar.
**Enfoque:** `/v1/db/:namespace/search` sobre colecciones sincronizadas
devuelve hits con referencia de origen (tabla, PK) para que la app cierre el
círculo leyendo el registro canónico en Postgres.
**Aceptación:** demo documentada: ERP con pgvector elimina pgvector — inserta
en su tabla de siempre y consulta semánticamente en Luma.

---

## W5 — Operabilidad y GA (releases **v4.29.x**)

### 5.1 `[ ]` Métricas Prometheus + OTel  · impacto ALTO · esfuerzo BAJO
**Objetivo:** operar a ciegas sin miedo.
**Enfoque:** `/v1/metrics` en formato Prometheus (hoy es JSON propio — mantener
ambos), histogramas de latencia por endpoint y por motor, trazas OTLP opt-in
(`otel_endpoint`), y un dashboard Grafana de referencia commiteado en `docs/`.
**Aceptación:** docker-compose de demo con Prometheus+Grafana levanta el
dashboard sin edición; alertas de referencia (WAL lag, backup verificado,
corrupción detectada, memoria por org).

### 5.2 `[ ]` Cuotas y presupuestos por organización  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** multi-tenant creíble = límites, no solo aislamiento.
**Enfoque:** por org: bytes en blob, claves KV, mensajes en cola, vectores,
rps (el rate limiting existente, por org). Excedido → error tipado + métrica +
evento de auditoría. Visible en el panel admin.
**Aceptación:** test: org A en su límite no degrada a org B; el panel muestra
consumo/límite por primitiva.

### 5.3 `[ ]` Endurecimiento de imagen y supply chain  · impacto MEDIO · esfuerzo BAJO
**Enfoque:** imagen `FROM scratch` firmada (cosign), SBOM publicado por release,
`cargo deny` ya existe — sumar `cargo audit` en CI y política de unsafe
(`#![forbid(unsafe_code)]` donde ya se cumpla, inventario donde no).
**Aceptación:** release v4.29 publica binario musl + imagen firmada + SBOM.

### 5.4 `[ ]` Suite de carga sostenida  · impacto MEDIO · esfuerzo MEDIO
**Objetivo:** números de capacidad honestos por primitiva (no solo vector).
**Enfoque:** harness de carga (goose o k6) por motor: KV ops/s, enqueue/dequeue
sostenido 1 h, blob MB/s, SSE con 5k suscriptores; medir en el perfil de
máquina objetivo (VM 2 vCPU/8 GB tipo OCI) y publicar en `docs/BENCHMARKS.md`
con la misma honestidad que el benchmark vectorial (incluye dónde pierde).
**Aceptación:** tabla de capacidad por primitiva con la máquina especificada;
regresión de rendimiento >15 % rompe el nightly.

### 5.5 `[ ]` Documentación de producto  · impacto ALTO · esfuerzo MEDIO
**Enfoque:** reorganizar `docs/` en: *Empezar* (quickstart 5 min por SDK),
*Operar* (runbooks: backup/restore, promoción de réplica, rotación de master
key, upgrade), *Integrar* (RESP, S3, CDC, SSE), *Referencia* (OpenAPI, config
completa, límites). El README enlaza; nada de docs solo-en-el-código.
**Aceptación:** una persona externa monta Luma con réplica y backup remoto
solo con los docs (probarlo de verdad con alguien).

### 5.6 `[ ]` Criterio GA  · impacto — · esfuerzo —
Se declara **1.0 / GA** cuando: W1 completo + 2.1 + 2.2 en verde 30 días en
producción propia (OCI) + RESP F1–F3 GA + 5.1 + 5.5. Todo lo demás puede ser
post-GA. Versionado: congelar API v1 (cambios rompientes ⇒ v2, nunca dentro de v1).

---

## Secuencia recomendada (ruta crítica)

| Orden | Ítems | Por qué primero |
|---|---|---|
| 1 | 1.1, 1.2 (+ F0 de RESP) | todo lo demás se apoya en durabilidad demostrada; el harness se construye una vez |
| 2 | 1.3, 1.4, 2.1 | backup remoto + WAL shipping = se puede prometer RPO; desbloquea pilotos con datos reales |
| 3 | RESP F1–F3 | la palanca de adopción más barata (Celery/arq drop-in) |
| 4 | 5.1, 2.2 | operar con métricas y una réplica antes de sumar más superficie |
| 5 | 3.2 (S3) y W4 (CDC) en paralelo | las dos vías de adopción restantes; elegir por demanda del piloto |
| 6 | 5.2–5.6 | cierre de GA |

## Pilotos de validación (dogfooding con carga real)

Cada frente se valida con un piloto interno real, del más barato al más valioso:

1. **Una app interna pequeña de recolección de datos** (hoy: 1 bucket S3 +
   1 tabla NoSQL + 1 función serverless) → blob + KV de Luma. Valida W1 + SDK.
   Esfuerzo mínimo, riesgo mínimo.
2. **Un servicio de memoria/conocimiento de equipo** → NS-Mem + doc store.
   Sustituye una imagen pública de terceros con acceso a datos internos:
   valida W1/W2 y elimina un riesgo de cadena de suministro.
3. **Las colas de una app con arq/Celery** → RESP F3. El drop-in de verdad.
4. **El almacén de documentos de una app con puerto S3-compatible** → 3.2 o
   SDK. El piloto que demuestra datos de producción reales.
5. **El RAG de una app que hoy usa pgvector** → W4. El cierre: Postgres +
   Luma conectados.

## Riesgos del plan maestro

| Riesgo | Mitigación |
|---|---|
| Abarcar los 5 frentes a la vez y no terminar ninguno | La ruta crítica es lineal; W3/W4 no arrancan hasta W1 en verde. Un solo frente activo por release |
| La réplica (2.2) crece hasta volverse un proyecto de consenso | Alcance congelado: solo-lectura + promoción manual; Raft es backlog con criterio de entrada explícito (demanda multi-escritor real) |
| S3/SigV4 y pgoutput tienen esquinas oscuras | Ambos entran por spike con criterio de salida (3.2 valida con mint de MinIO; 4.1 decide crate vs propio antes de diseñar) |
| El SQL embebido (SQLite) se malinterpreta como promesa de reemplazar Postgres | Posicionamiento fijado en README y docs: W4 existe precisamente para lo contrario |
