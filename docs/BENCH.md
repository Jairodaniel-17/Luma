# Bench estable

El binario `src/bin/bench.rs` ya sigue un modelo de ejecución pensado para no castigar el SSD:

1. **prepare**: genera (o reutiliza) el dataset sintético por `dim/rows/metric/seed` y lo guarda en `target/bench/<hash>`.
2. **build-index**: construye cada índice (baseline, IVF_FLAT_Q8, etc.) una sola vez por modo/configuración.
3. **run-queries**: ejecuta las búsquedas sobre los índices ya construidos.
4. **cleanup**: borra datasets e índices temporales, salvo que se use `--keep-data`.

## Flags clave

- `--seed 123`: hace reproducible la generación de vectores y queries.
- `--data-dir target/bench/<hash>`: permite apuntar a un directorio específico si se quiere compartir data entre corridas.
- `--reuse-data`: salta la generación cuando el dataset ya existe.
- `--reuse-index`: evita reconstruir los índices si los archivos están presentes.
- `--keep-data`: no borra datasets/índices al final (útil para corridas largas).
- `--in-mem`: fuerza modo en memoria y evita escrituras en disco para comparar tiempos sin I/O.

## Cómo ejecutar (paso a paso)

1. Preparar dataset + índices baseline/IVF:
   ```
   cargo run --release --bin bench -- \
     --rows 20000 --search-queries 2000 \
     --dims 768,1024 \
     --modes baseline-f32,ivf-flat-q8 \
     --ivf-clusters 1024 --ivf-nprobe 8 \
     --ivf-min-train-vectors 2048 \
     --reuse-data --reuse-index --keep-data
   ```
2. Reutilizar los artefactos previos para cronometrar búsquedas únicamente:
   ```
   cargo run --release --bin bench -- \
     --rows 20000 --search-queries 2000 \
     --dims 768,1024 \
     --modes baseline-f32,ivf-flat-q8 \
     --reuse-data --reuse-index
   ```
3. Limpiar manualmente cuando ya no se necesiten los archivos (Windows PowerShell):
   ```
   Remove-Item -Recurse -Force target/bench
   ```

## Próximos benchmarks

Los escenarios con `dims=4096` y `nprobe=16/32` quedarán pendientes hasta tener ventana de laboratorio y evitar sesiones de más de 3 h sobre el NVMe. Cuando se agenden, bastará con ajustar `--dims`, `--ivf-nprobe` y `--ivf-clusters` siguiendo la guía anterior.
