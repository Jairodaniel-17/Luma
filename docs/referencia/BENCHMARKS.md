# Benchmarks

Tres familias de mediciones distintas:

1. **[Comparativa contra Qdrant y Milvus](#comparativa-contra-qdrant-y-milvus-50k--768)** — Luma frente a otros motores, misma máquina y mismo dataset.
2. **[Camino de escritura KV / RESP](#camino-de-escritura-kv--resp)** — de dónde salió el 29× de escrituras por segundo, capa por capa, contra un Redis 7 real, y por qué se eligió un LSM.
3. **[Benchmarks internos reproducibles](#benchmarks-internos-reproducibles)** — el binario `src/bin/bench.rs` para comparar modos de índice entre sí.

---

## Comparativa contra Qdrant y Milvus (50k × 768)

Misma máquina, mismo dataset y misma métrica para los tres motores. Todas las columnas están medidas; ninguna es estimada.

> ⚠️ **Reproducibilidad:** esta corrida se hizo con scripts ad-hoc que **no están versionados** en el repo. Las cifras son las observadas, pero hoy no se pueden reejecutar con un comando del repositorio. Portarlos a `src/bin/bench.rs` está pendiente.

### Máquina de prueba

| | |
|---|---|
| **CPU** | Intel Core i7-1355U (12 hilos, laptop; escalado de frecuencia activo) |
| **RAM** | 15 GiB |
| **Disco** | NVMe SSD |
| **SO** | Ubuntu 24.04.4 LTS · kernel 6.17 |

Es una laptop con throttling térmico: los valores absolutos suben en servidor, pero la **comparación relativa entre motores es válida** porque los tres corrieron en el mismo equipo, uno a la vez.

### Qué se midió y cómo

- **Dataset:** 50.000 vectores de 768 dimensiones + 200 consultas, distribución aleatoria uniforme (`base.npy` / `queries.npy`, `float32`).
- **Métrica:** cosine, `k = 10`.
- **Ground-truth:** top-10 exacto por fuerza bruta sobre el mismo dataset → recall@10 real, no estimado.
- **Configuración:** valores **por defecto** de cada motor. Qdrant y Milvus con HNSW `M=16, ef_construct=200, ef=64`.
- **RAM:** memoria anónima real del proceso tras cargar (no caché de página de disco).
- **Protocolo:** ingesta por lotes vía API HTTP → espera de indexación → 200 consultas secuenciales → recall contra ground-truth.

### Resultados

| Motor / modo | Ingesta (vec/s) | Consulta (qps) | Latencia media | Recall@10 | RAM |
|---|---:|---:|---:|---:|---:|
| **Qdrant** (HNSW, ef=64) | **1.859** | 237 | 4.22 ms | 0.416 | 320 MB |
| **Milvus** (HNSW, ef=64) | 1.284 | 476 | 2.09 ms | 0.086 | 846 MB |
| **Luma — DiskANN** (low-RAM) | 807 | **515** | **1.94 ms** | 0.056 | **133 MB** |
| **Luma — HNSW** (equilibrio, ef=128 default) | 926 | 99 | 10.08 ms | 0.853 | 464 MB |
| **Luma — HNSW** (velocidad, ef=32) | 926 | 199 | 5.02 ms | 0.474 | 406 MB |

<sub>Ingesta HNSW = 926 vec/s tras paralelizar el build por lote (antes 293); es independiente del `ef` de consulta.</sub>

### Cómo leer la columna de recall

Con vectores **aleatorios uniformes** (sin estructura de clusters) el ANN es un caso adversarial: incluso Qdrant solo alcanza 0.42 de recall. Por eso esa columna refleja sobre todo **el punto de equilibrio velocidad↔precisión que cada motor elige de fábrica**, más que la calidad absoluta del índice. En embeddings reales (que sí forman clusters) todos los recalls suben.

Comparar recalls entre filas solo tiene sentido **a latencia equivalente**. Es lo que hace la tercera viñeta de abajo.

### Discusión

- **Luma DiskANN** ocupa el extremo *rápido y ligero*: la consulta más veloz del grupo (515 qps, 1.94 ms) con la menor RAM (133 MB — 2,4× menos que Qdrant, 6,4× menos que Milvus), a costa de recall.
- **Luma HNSW** es el modo *precisión*, ahora calibrado (ver abajo): a su punto de velocidad (`ef=32`) iguala la latencia de Qdrant (5.0 vs 4.2 ms) con mejor recall (0.474 vs 0.416); subiendo `ef` escala hasta recall 0.95.
- **Qdrant** es un punto medio sólido de fábrica; **Milvus** iguala en ingesta pero pesa 846 MB y su recall a igual `ef` es el más bajo.

### Dónde gana Luma

- 🏆 **Consumo de RAM** — DiskANN corre 50k en **133 MB**; ningún competidor baja de 320 MB. Es el objetivo de diseño y se cumple medido.
- 🏆 **Latencia y throughput de consulta** — **1.94 ms / 515 qps** en DiskANN, el más rápido del grupo.
- 🏆 **Precisión a igual velocidad** — a latencia equivalente a Qdrant, Luma HNSW da **más recall** (0.474 vs 0.416); y llega hasta 0.95 subiendo `ef`.

### Calibración de HNSW: el punto de equilibrio

El modo HNSW tenía un problema: el bucle de expansión de candidatos perseguía una estimación de recall inalcanzable en datos difíciles y terminaba escaneando casi todo (recall 0.98 pero **348 ms/consulta**, inservible). Se corrigió con dos cambios (`src/vector/mod.rs`, `src/config.rs`):

1. **`ef` de búsqueda configurable** (`HNSW_SEARCH_EF`, default 128) que acota la expansión a un punto fijo, como el `hnsw_ef` de Qdrant.
2. **Búsqueda única** al techo `ef` en vez de rampar 16→32→…→N. La rampa lanzaba varias búsquedas HNSW desechables por consulta: eliminarla dio **~3× más throughput al mismo recall**.

Curva medida tras la calibración (mismo dataset, `HNSW_SEARCH_EF` variando):

| ef | qps | latencia | recall@10 | RAM |
|---:|---:|---:|---:|---:|
| 32 | 199 | 5.02 ms | 0.474 | 406 MB |
| 64 | 140 | 7.13 ms | 0.675 | 406 MB |
| 96 | 115 | 8.65 ms | 0.792 | 408 MB |
| **128 (default)** | **99** | **10.08 ms** | **0.853** | **411 MB** |
| 192 | 82 | 12.19 ms | 0.921 | 409 MB |
| 256 | 74 | 13.45 ms | 0.947 | 407 MB |

Antes vs después, mismo `ef=192`: **26 qps → 82 qps** (3,15×) con recall idéntico (0.92). El usuario elige el punto: `ef=32` para máxima velocidad, `ef≥192` para máximo recall; el default 128 es el balance.

### Ingesta paralela

El upsert por lote insertaba al grafo HNSW **de uno en uno**. Se paralelizó (`apply_upsert_batch` acumula los pares del lote y hace un `insert_batch` con `parallel_insert` de rayon por segmento, la misma maquinaria del build masivo): **293 → 926 vec/s (3,16×)**, con recall idéntico (0.860). La brecha de ingesta contra Qdrant bajó de **6,3× a ~2×**.

### Lo que aún va por detrás

- **Ingesta**: 926 (HNSW) / 807 (DiskANN) vec/s siguen por debajo de Qdrant/Milvus (1.284–1.859). Lo que resta es el bookkeeping por registro (WAL, mmap, cuantización) que aún es serial; lotes más grandes lo acercan más.
- **Throughput de consulta HNSW** a igual recall: `hnsw_rs` es algo más lento que el HNSW propio de Qdrant. La ventaja neta de Luma sigue siendo **RAM (DiskANN, 133 MB)** y **precisión a igual latencia**.

---

## Camino de escritura KV / RESP

Esta sección existe porque una cifra sola no dice nada: el interés está en **qué
capa costaba qué**, y en las dos veces que la medición mató un arreglo que ya
estaba a punto de escribirse.

Punto de partida: 785 `SET`/s por el listener RESP. Punto de llegada: **22.989**
en SSD NVMe, con Redis 7 en **28.517** por la misma ruta y el mismo cliente. El
diagnóstico que abrió todo fue que **quitar el fsync entero solo compraba 2×** y
que el pipelining (`-P 16`) no compraba nada — dos hechos que juntos dicen que el
techo no era el disco ni la red, sino una serialización dentro del servidor.

### Cada capa, medida

| # | Cambio | `SET`/s por RESP | Factor acumulado | Qué era el techo antes de este cambio |
|---|---|---:|---:|---|
| 0 | Punto de partida (`wal_sync_mode = per_write`) | **785** | 1× | Un fsync por escritura *y* una transacción de la proyección por escritura, en serie |
| 1 | **Group commit** del WAL (líder/seguidor) | **4.648** | 5,9× | El fsync ya se comparte por lote, pero la proyección sigue abriendo una transacción por registro |
| 2 | La proyección aplica **el lote entero** en una transacción | **8.893** | 11,3× | Plano con 32, 128 y 256 clientes mientras la latencia subía lineal — cola con servicio fijo |
| 3 | **`block_in_place`** en el dispatch RESP | **22.989** | **29,3×** | **El camino de red.** El control (PING, que cruza la misma red y no hace trabajo) da 26.178/s, así que `SET` ya va al 88% de lo que el transporte permite |
| 4 | Proyección KV **redb → LSM (fjall)** | *no se separa aquí* | — | — |

Ninguno de los cuatro toca una garantía de durabilidad: `wal_sync_mode` sigue en
`per_write`, y el fsync no desapareció, se **comparte** entre los escritores que
coinciden en el mismo lote — el mismo patrón que usan Postgres y MySQL.

> **La fila 4 dice «no se separa» a propósito, y es una corrección.** Una versión
> anterior de esta tabla le atribuía al LSM un salto de 24.242 a 26.178 `SET`/s.
> Las dos cifras estaban mal: 26.178 es el **control de PING**, no un `SET`, y
> tras el paso 3 el listener ya está contra el techo del transporte, así que un
> cambio en la proyección no puede aparecer aquí ni debería. Donde sí aparece es
> en proceso, sin red de por medio: **26.379 → 35.622 escrituras/s** con 128
> escritores. El error salió al reejecutar el benchmark en vez de al releerlo,
> que es la única forma en que salen estos.

### El número que más cambia el resultado no es ninguno de los cuatro

Es **en qué disco vive `data_dir`**. Mismo binario, misma configuración, mismo
cliente, misma máquina — solo cambia la unidad:

| `data_dir` en | `SET`/s | `GET`/s |
|---|---:|---:|
| Un disco duro mecánico (HDD SATA, 7200 rpm) | 3.142 | 24.671 |
| Un SSD NVMe | **22.989** | 25.685 |

**7,3× de diferencia en escritura y ninguna en lectura**, que es exactamente la
firma de un camino dominado por la latencia de fsync: las lecturas se sirven de
memoria, no tocan el disco, y no se mueven ni un 4%. Un fsync en un plato que
gira cuesta un cuarto de vuelta; en NVMe cuesta un viaje al controlador.

Dos consecuencias prácticas:

1. **No pongas `data_dir` en un disco mecánico.** No es una recomendación de
   estilo: son 3.142 escrituras/s contra 22.989 en la misma máquina.
2. **Cualquier cifra de escritura sin decir en qué dispositivo se midió no
   significa nada.** Las de esta sección son NVMe salvo donde diga lo contrario.

### Contra Redis 7, por el mismo camino de red

Comparar contra Redis honestamente exige una cosa que es fácil de saltarse: que
el cliente llegue a los dos **por la misma ruta**. Medido desde dentro de su
propio contenedor, Redis da 147.783 `SET`/s; medido cruzando la NAT de Docker
hacia el host —que es por donde se llega a Luma— da 28.517. La diferencia es el
transporte, no el motor, y el control de PING lo prueba: 150.754 contra 28.763.

Así que ambos por la misma ruta, mismo `redis-benchmark`, 30.000 operaciones,
256 clientes:

| | PING (control) | `SET` | `SET` como % de su propio control | `GET` |
|---|---:|---:|---:|---:|
| **Redis 7** | 28.763 | **28.517** | 99% | 27.298 |
| **Luma** | 26.178 | **22.989** | 88% | 25.685 |

- **Luma está al 81% del `SET` de Redis y al 94% de su `GET`.**
- **Y hace fsync de cada escritura confirmada, que Redis por defecto no hace.**
  Redis con la configuración de fábrica responde OK antes de que el dato esté en
  el medio (`appendfsync everysec` o solo RDB); Luma no vuelve de un `SET` hasta
  que su lote está en disco. Comparar 22.989 durables contra 28.517 no durables
  favorece a Redis, no a Luma, y así hay que leerlo.
- **Los dos están contra el techo del transporte en esta ruta**, al 99% y al 88%
  de su propio control. El motor de Luma da 35.622 escrituras/s en proceso, por
  encima de lo que esta red deja pasar, así que en una ruta más rápida la brecha
  la decide otra cosa.

Dónde gana Redis, dicho sin adornos: en escritura pura sigue por delante, y su
latencia p50 por loopback (0,78 ms) está en otra liga que cualquier cosa que
cruce una NAT. Lo que Luma ofrece a cambio no es velocidad, es que la escritura
que confirmó está en disco y que el mismo binario también es el vectorial, el
SQL, el S3 y la memoria de agentes.

### Los dos hallazgos que solo aparecen midiendo

**El paso 3 no era un problema de base de datos.** Tras group commit el motor
daba 21.616 escrituras/s *en proceso* y por RESP salían 8.893, plano con 32, 128
y 256 clientes. `dispatch` es síncrono y hace trabajo bloqueante de verdad (el
fsync del WAL, la transacción de la proyección), así que llamarlo directo desde
la tarea async **bloqueaba un worker de Tokio**: como máximo había tantos
comandos en vuelo como workers, 20 en esta máquina. Y group commit no puede
agrupar escritores que nunca llegan, así que el lote nunca pasaba de 20 aunque
hubiera 256 clientes esperando. `block_in_place` entrega las otras tareas de ese
worker a un hilo de reemplazo y deja que este bloquee.

**Un «group commit de verdad» no habría comprado nada.** Estaba planificado y se
descartó al medir: el `append_guard` global significa que nunca hay dos
escritores dentro del WAL a la vez, así que no había nada que agrupar que no
estuviera ya agrupado.

### Por qué un LSM, y por qué no las otras dos opciones

Se midieron tres diseños antes de elegir. Los ejes son los que se pidieron:
velocidad, **no gastar RAM sino disco**, y todo en Rust en un binario.

| | **1+2** redb por lotes | **3** WAL como store + índice en RAM | **5** LSM (fjall) — *elegida* |
|---|---|---|---|
| Inserción en rol de proyección | 11.401/s | — | **432.764/s** |
| 128 escritores concurrentes | 235.679/s (por lotes) | 54.514/s | **208.944/s** |
| Con fsync cada 32 | — | — | 72.096/s |
| RAM | Acotada por la caché de páginas | **150 bytes por clave, para siempre** — 15 GiB con 100M claves | Acotada por el memtable: **+18,3 MiB** en 50k inserciones |
| Amplificación de escritura | **16 KiB para un valor de 30 bytes** (B-tree copy-on-write: reescribe el camino de páginas de la hoja a la raíz) | Ninguna: el valor ya está en el WAL | La del volcado de memtable, amortizada |
| Iteración ordenada (`KEYS`, `SCAN`, `list_range`) | Sí | **No, a ningún precio**, con índice hash; el mapa ordenado que sí puede es el que cuesta 150 B/clave | Sí, nativa: un LSM está ordenado. Escaneo por prefijo de 100 claves en **34 µs** |
| Compactación | No aplica | **Hay que escribirla** — es el núcleo riesgoso de esta opción | Ya escrita y probada por alguien más |
| Migración de datos | — | Ninguna | **Ninguna** |
| Todo en Rust | Sí | Sí | Sí, `fjall` es Rust puro |

Dos cosas de esta tabla merecen decirse en voz alta:

- **La opción 3 se descartó por la RAM, que es justo lo que se pidió no gastar.**
  Su índice hash de 37 bytes por clave era el número atractivo, pero un hash de
  la clave no se puede recorrer en el orden de la clave, y `KEYS`, `SCAN` y
  `list_range` lo necesitan. El mapa ordenado que sí sirve cuesta 150 bytes por
  clave para siempre: 15 GiB con 100M claves.
- **El LSM no tenía migración, y eso es lo que lo hizo viable.** Se había
  descartado antes por «una dependencia nueva y una migración de datos». La
  dependencia es real; la migración no existía: la proyección **no guarda datos
  propios**, es una proyección del WAL que el replay reconstruye desde
  `applied_offset`. Cambiarla es borrar un fichero y reproducir.

  Y eso no es un argumento, es un test: `tests/golden_data_dir.rs` arranca el
  binario actual sobre un `data_dir` grabado por v4.24.0 —que contiene
  `state.redb`— y lee de vuelta `kv.value.marker == "kv-v4.24.0"`. Pasa.

### Escalado con clientes, tras los cuatro cambios

| Clientes | `SET`/s |
|---:|---:|
| 32 | 12.804 |
| 128 | 23.981 |
| 256 | **24.242** |

Que escale con los clientes es el punto: antes del paso 3 era plano (8.893 /
7.755 / 8.180), que es la firma de una cola con servicio fijo y no de un límite
de red. Y que se aplane *después* de 128 clientes ya sí es el transporte: el
motor en proceso da **35.622/s** con 128 escritores, por encima de lo que esta
ruta deja pasar.

### Lo que ata ahora

El coste por escritura sube con el tamaño del valor, y el LSM ya aplanó la
parte que le tocaba:

| Tamaño del valor | Con redb | Con el LSM |
|---:|---:|---:|
| 10 bytes | 15.185/s | 13.569/s |
| 200 bytes | 12.991/s | 11.406/s |
| 2.000 bytes | 8.945/s | **10.418/s** |

El caso que mejora es el del valor **grande**, que es justo el que un B-tree
copy-on-write castigaba reescribiendo páginas; en valores pequeños el LSM queda
un pelo por debajo y no compensa perseguirlo. Lo que queda de la pendiente apunta
al formato del evento y no al almacenamiento: el payload es un
`serde_json::Value` que se clona a la cola, se clona otra vez para el lote, se
codifica a JSON para el WAL, se decodifica por `StoredVal` y se re-codifica para
la proyección. La proyección ya no es el límite —hace 432.764 inserciones/s, un
orden de magnitud más que el motor entero— y el motor en proceso escala limpio:
1.453 → 4.792 → 16.367 → **35.622/s** con 1, 8, 32 y 128 escritores.

### Reproducir

```bash
# El camino RESP completo, contra un Redis 7 real como referencia
scripts/resp-benchmark.sh <puerto-resp> <api-key>

# Los diagnósticos por capa (ignorados por defecto)
cargo test --release --test wal_sync_cost       -- --ignored --nocapture --test-threads=1
cargo test --release --test redb_ceiling        -- --ignored --nocapture --test-threads=1
cargo test --release --test lsm_ceiling         -- --ignored --nocapture --test-threads=1
cargo test --release --test wal_index_prototype -- --ignored --nocapture
cargo test --release --test ram_cost            -- --ignored --nocapture
```

Los cinco diagnósticos siguen en el repo a propósito: son la evidencia de por qué
el diseño es el que es, y de las dos opciones que se descartaron con números en
vez de con opiniones.

---

## Benchmarks internos reproducibles


src/bin/bench.rs implementa el flujo prepare → build-index → run-queries → cleanup y evita maltratar el SSD:

- --reuse-data: solo genera el dataset binario (por dim/seed/metric) si no existe.
- --reuse-index: si el fingerprint (modo + tunings + rows) coincide, reaprovecha el índice.
- --keep-data: conserva target/bench/<hash> para reusar artefactos más tarde.

> Consejo: ejecuta primero una corrida “mini” (--rows 100 --search-queries 20) para validar que el pipeline funcione antes de lanzar runs pesados.

## IVF_FLAT_Q8 vs baseline_f32 (Cosine)

```bash
cargo run --release --bin bench --   --rows 20000 --search-queries 2000   --dims 768,1024   --modes baseline_f32,ivf_flat_q8   --ivf-clusters 1024   --ivf-nprobe 8   --ivf-min-train-vectors 2048   --ivf-retrain-min-deltas 50000   --q8-refine-topk 512   --reuse-data --reuse-index
```

| Dim | Mode          | Insert p50/p95/p99 (µs) | Search p50/p95/p99 (µs) | Throughput (insert/search) | Recall@10 | RAM (MiB) | Disco (MiB) |
| --- | ------------- | ----------------------- | ----------------------- | -------------------------- | --------- | --------- | ----------- |
| 768 | baseline_f32  | 7123 / 9350 / 11563     | 9983 / 11298 / 12110    | 140 vec/s / 100 qps        | 100.00%   | 468.11    | 363.70      |
| 768 | ivf_flat_q8   | 7276 / 9102 / 10405     | 8910 / 9878 / 10396     | 139 vec/s / 112 qps        | 83.39%    | 468.08    | 363.69      |
|1024 | baseline_f32  | 7406 / 8919 / 9561      | 10658 / 11747 / 12144   | 138 vec/s / 93 qps         | 100.00%   | 597.12    | 482.22      |
|1024 | ivf_flat_q8   | 7317 / 8961 / 9695      | 8858 / 9850 / 10730     | 140 vec/s / 113 qps        | 83.78%    | 597.37    | 482.20      |

- Métrica: Cosine  
- ivf_clusters / centroid_count: 1024  
- nprobe: 8  
- q8_refine_topk: 512

## DiskANN vs IVF/HNSW (reproducible, dataset mediano)

> Requiere DATA_DIR (no usar --in-mem). Se recomienda correrlo mientras el host está libre.

```bash
cargo run --release --bin bench --   --rows 20000 --search-queries 1000   --dims 4096   --modes baseline_f32,ivf_flat_q8,diskann   --ivf-clusters 2048 --ivf-nprobe 16 --q8-refine-topk 512   --diskann-max-degree 64 --diskann-search-list-size 256   --reuse-data --reuse-index --keep-data
```

Salida esperada (resumen textual para adjuntar en PRs):

```text
--- dimension 4096 (Cosine) ---
# mode=baseline_f32 dim=4096
insert: ...
search: ...
# mode=ivf_flat_q8 dim=4096
insert: ...
search: ...
# mode=diskann dim=4096
insert: ...
search: ...
```

- graph_files aparecen en target/bench/<hash>/dim_4096/diskann/diskann/.
- Las métricas p50/p95/p99 deben mostrar menor latencia y menor RAM vs baseline; documenta cualquier gap de recall si aparece.

## Validación rápida (≈100 filas, sin estrés)

Para comprobar que el flujo prepare → build-index → run-queries sigue operativo sin castigar el SSD:

```bash
cargo run --release --bin bench --   --rows 100 --search-queries 50   --dims 768   --modes baseline_f32,ivf_flat_q8   --ivf-clusters 64 --ivf-nprobe 4   --ivf-min-train-vectors 64   --reuse-data --reuse-index --in-mem
```

No deja resultados formales, solo sirve como smoke test corto.

## Próximas capturas (pendiente)

1. Guardar una corrida con dims=4096 (comando DiskANN anterior) y adjuntar el log completo.
2. Registrar cualquier ajuste en centroid_count, q8_refine_topk o parámetros DiskANN cuando se cambien.
3. Evitar pruebas de estrés ≈1 M filas; si se requiere un dataset mayor, coordinar antes para no superar el límite de 100 k filas.
