# ROADMAP.md

## P0 (en curso)

- **Batch APIs** ✅ (`/v1/state/batch_put`, `/v1/vector/*_batch`).
- **Errores consistentes** ✅ (`ErrorBody { error, message }` everywhere).
- **SDK Python oficial** ✅ (`pip install -e RAG-client-py`).
- **Bind seguro** ✅ (`127.0.0.1` default + flags `--bind/--unsafe-bind`).

## P1 (próximos pasos)

1. **Vector Store**
   - Mejorar heurística de tamaño de segmento (adaptativo en función del live_count).
   - Compaction automática (scheduler que ejecute `vacuum` cuando `tombstones > 30%`).
   - Exponer métricas por colección (live_count, segmentos, applied_offset).
2. **Metadata index**
   - Añadir soportes para filtros numéricos (`<`, `>`, rangos) usando estructuras ligeras (ej. roaring bitmaps).
   - Persistir el índice (snapshot + WAL) para evitar reconstrucción en start-up.
3. **DocStore**
   - Búsqueda full-text opcional (integración con Tantivy/Qdrant-lite).
   - Borrado en cascada (cleanup automático de índices/keys hijas).

## P2 (investigación / nice-to-have)

- **SQLite module**
  - Pool de conexiones (5-10) + migraciones básicas (ej. `sql/migrations/*.sql`).
  - Query planner guardrails (ej. timeout por request, límites de respuesta).
- **Multitenancy / Multi-collections**
  - Namespaces lógicos con `prefix + API key`.
  - Cuotas y límites (número máx. de vectores/documentos por colección).
- **Operador / Packaging**
  - Contenedor oficial (`ghcr.io/.../rust-kiss-vdb`) con sqlite + data_dir montado.
  - Helm chart ligero para clusters (sólo intra-clúster, sin exponer).

> Nota: la filosofía sigue siendo KISS; cualquier feature que complique demasiado el binario deberá documentarse primero en `docs/ROADMAP.md` antes de implementarse.
