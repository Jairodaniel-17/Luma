# Benchmarks reproducibles

src/bin/bench.rs implementa el flujo prepare → build-index → run-queries → cleanup y evita maltratar el SSD:

- --reuse-data: solo genera el dataset binario (por dim/seed/metric) si no existe.
- --reuse-index: si el fingerprint (modo + tunings + rows) coincide, reaprovecha el índice.
- --keep-data: conserva 	arget/bench/<hash> para reusar artefactos más tarde.

> Consejo: ejecuta primero una corrida “mini” (--rows 100 --search-queries 20) para validar que el pipeline funcione antes de lanzar runs pesados.

## IVF_FLAT_Q8 vs baseline_f32 (Cosine)

`
cargo run --release --bin bench --   --rows 20000 --search-queries 2000   --dims 768,1024   --modes baseline_f32,ivf_flat_q8   --ivf-clusters 1024   --ivf-nprobe 8   --ivf-min-train-vectors 2048   --ivf-retrain-min-deltas 50000   --q8-refine-topk 512   --reuse-data --reuse-index
`

| Dim | Mode          | Insert p50/p95/p99 (µs) | Search p50/p95/p99 (µs) | Throughput (insert/search) | Recall@10 | RAM (MiB) | Disco (MiB) |
| --- | ------------- | ----------------------- | ----------------------- | -------------------------- | --------- | --------- | ----------- |
| 768 | baseline_f32  | 7123 / 9350 / 11563     | 9983 / 11298 / 12110    | 140 vec/s / 100 qps        | 100.00%   | 468.11    | 363.70      |
| 768 | ivf_flat_q8   | 7276 / 9102 / 10405     | 8910 / 9878 / 10396     | 139 vec/s / 112 qps        | 83.39%    | 468.08    | 363.69      |
|1024 | baseline_f32  | 7406 / 8919 / 9561      | 10658 / 11747 / 12144   | 138 vec/s / 93 qps         | 100.00%   | 597.12    | 482.22      |
|1024 | ivf_flat_q8   | 7317 / 8961 / 9695      | 8858 / 9850 / 10730     | 140 vec/s / 113 qps        | 83.78%    | 597.37    | 482.20      |

- Métrica: Cosine  
- ivf_clusters / centroid_count: 1024  
- 
probe: 8  
- q8_refine_topk: 512

## DiskANN vs IVF/HNSW (reproducible, dataset mediano)

> Requiere DATA_DIR (no usar --in-mem). Se recomienda correrlo mientras el host está libre.

`
cargo run --release --bin bench --   --rows 20000 --search-queries 1000   --dims 4096   --modes baseline_f32,ivf_flat_q8,diskann   --ivf-clusters 2048 --ivf-nprobe 16 --q8-refine-topk 512   --diskann-max-degree 64 --diskann-search-list-size 256   --reuse-data --reuse-index --keep-data
`

Salida esperada (resumen textual para adjuntar en PRs):

`
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
`

- graph_files aparecen en 	arget/bench/<hash>/dim_4096/diskann/diskann/.
- Las métricas p50/p95/p99 deben mostrar menor latencia y menor RAM vs baseline; documenta cualquier gap de recall si aparece.

## Validación rápida (≈100 filas, sin estrés)

Para comprobar que el flujo prepare → build-index → run-queries sigue operativo sin castigar el SSD:

`
cargo run --release --bin bench --   --rows 100 --search-queries 50   --dims 768   --modes baseline_f32,ivf_flat_q8   --ivf-clusters 64 --ivf-nprobe 4   --ivf-min-train-vectors 64   --reuse-data --reuse-index --in-mem
`

No deja resultados formales, solo sirve como smoke test corto.

## Próximas capturas (pendiente)

1. Guardar una corrida con dims=4096 (comando DiskANN anterior) y adjuntar el log completo.
2. Registrar cualquier ajuste en centroid_count, q8_refine_topk o parámetros DiskANN cuando se cambien.
3. Evitar pruebas de estrés ≈1 M filas; si se requiere un dataset mayor, coordinar antes para no superar el límite de 100 k filas.
