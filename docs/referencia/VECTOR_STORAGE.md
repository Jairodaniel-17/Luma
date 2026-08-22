# VECTOR_STORAGE.md

## Segmentos activos y fríos

- Cada colección vectorial se materializa en un directorio `data_dir/vectors/<collection>`.
- Los vectores en memoria se reparten en segmentos (`DEFAULT_SEGMENT_MAX = 8192` puntos).
- Cada segmento mantiene su propio índice HNSW y un mapa `id -> data_id`.  
  - El segmento activo recibe nuevos upserts.  
  - Los segmentos fríos sólo se leen (no se escriben) hasta que se compactionan.
- El motor fusiona los resultados de todos los segmentos durante un `search`. Esto evita reconstrucciones globales cuando llegan nuevos puntos y mantiene tiempos de búsqueda estables.

## Índice por metadata

- Los metadatos tipo `{"tag":"foo"}` se normalizan y se insertan en un índice exact-match en memoria.  
- Si la consulta incluye filtros `{ "tag": "foo" }`, primero se calcula el conjunto candidato usando el índice y:
  - Si el conjunto es pequeño (<= 512 docs) se hace un ranking exacto (producto punto / coseno) sin tocar HNSW.
  - Si es grande, el `search` tradicional se limita a esos IDs para ahorrar post-filtro.
- El índice se reconstruye al cargar un snapshot o al aplicar vacuum/compaction.

## Persistencia y archivos

- `manifest.json`: describe dim, métrica, live_count, applied_offset, etc.
- `vectors.bin`: WAL append-only (registro por registro con `RecordOp::Upsert/Delete`).
- En memoria mantenemos los vectores “normalizados” para el métrico DOT (se usa `l2_normalize` antes de insertarlos).

## Vacuum / Compaction

Sin compaction los `vectors.bin` crecerían indefinidamente (tombstones).  
Se añadió una herramienta CLI:

```bash
rust-kiss-vdb vacuum --collection docs
```

### ¿Qué hace?

1. Bloquea la colección.
2. Carga todos los items vivos desde memoria.
3. Reescribe `vectors.bin` a un archivo temporal sin tombstones ni duplicados.
4. Actualiza el manifest (live_count, total_records, file_len).
5. Reconstruye segmentos e índices en memoria.

> Requisitos: `DATA_DIR` (o `SQLITE_DB_PATH`) debe estar configurado para que la CLI sepa dónde leer/escribir.

### Recomendaciones

- Ejecutar `vacuum` off-line o en una ventana donde no haya ingestas masivas; aunque es seguro, bloquea la colección mientras escribe el archivo temporal.
- Tras compaction es buena idea tomar un snapshot (`cargo run --bin ...` o usando el endpoint admin) para que el WAL reducido se refleje en backups.
