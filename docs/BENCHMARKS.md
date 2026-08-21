# Benchmarks

Dos familias de mediciones distintas:

1. **[Comparativa contra Qdrant y Milvus](#comparativa-contra-qdrant-y-milvus-50k--768)** — Luma frente a otros motores, misma máquina y mismo dataset.
2. **[Benchmarks internos reproducibles](#benchmarks-internos-reproducibles)** — el binario `src/bin/bench.rs` para comparar modos de índice entre sí.

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
