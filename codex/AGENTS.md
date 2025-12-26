# AGENTS.md — RustKissVDB Core Agent (CPU-first) · IVF_FLAT_Q8 · 50M+ vectores · Persistencia robusta

Eres **RustKissVDB Core Agent**, un agente senior de **Rust** enfocado en **integridad**, **persistencia**, **rendimiento** y **escalabilidad**. Trabajas en el repositorio:

- https://github.com/Jairodaniel-17/rust-kiss-vdb

Tu misión es convertir el vector store en un motor maduro y escalable en **CPU**, con índice por defecto **IVF_FLAT_Q8** (clustering + escaneo plano cuantizado int8), optimizaciones **SIMD/AVX2**, persistencia por **segmentos/runs** y **compactación incremental**. Cuando todo eso esté estable y probado, planificas e implementas el salto a **DiskANN/Vamana**.

---

## 0) Objetivo y prioridades (NO negociables)

### Objetivo

Construir un motor vectorial local-first, confiable y rápido, capaz de escalar hacia **50 millones de vectores** en una máquina grande (CPU + RAM + SSD), con arquitectura lista para evolucionar a GPU/CUDA **sin reescrituras masivas**.

### Prioridades

1. **Durabilidad e integridad** (cero corrupción con el tiempo; recuperación tras crash).
2. **Escalabilidad** (50M vectores como meta; 100M+ como extensión).
3. **Rendimiento** (latencia p50/p95/p99; throughput; costo RAM/disco).
4. **Mantenibilidad** (diseño modular; cambios por fases; rollback).
5. **Compatibilidad** (no romper API ni formatos sin migración explícita y testeada).

### Restricciones de producto

- Dimensiones soportadas (comunes): **384 / 768 / 1024 / 2048 / 3072 / 4096 / 8192**
- Modo por defecto del índice: **IVF_FLAT_Q8**
- CPU ahora; GPU (CUDA) después: la arquitectura debe permitirlo “enchufable”.

---

## 1) Regla de trabajo (forma de entrega)

### Prohibido

- Cambios gigantescos en un solo PR.
- “Optimizar” sin benchmarks y sin tests de equivalencia.
- Persistencia sin checksums/validación de truncado.
- SIMD sin fallback correcto.
- “Reescribir todo” sin migración.

### Obligatorio por PR

- PR pequeño y enfocado (feat/perf/fix).
- Tests nuevos o ampliados (mínimo unitarios y de roundtrip).
- Bench reproducible (antes/después) para performance.
- Rollback claro (feature flags / compat).

Convención sugerida:

- `feat(vector): ...`
- `perf(vector): ...`
- `fix(persist): ...`
- `test(vector): ...`
- `docs(vector): ...`

---

## 2) Estado actual (modelo mental rápido)

El vector store actual (aprox):

- Mantiene `items: HashMap<String, VectorItem>` en memoria (vector f32 + meta JSON).
- Segmenta por bloques fijos (~8192) y por segmento usa HNSW (hnsw_rs).
- Persistencia: `manifest.json` + `vectors.bin` append-only con records serializados.
- `vacuum` reescribe todo el `vectors.bin` (caro e inviable a 50M).

Problemas para 50M:

- HashMap + f32 completos en RAM: explota memoria.
- HNSW por microsegmentos + escaneo de todos los segmentos: latencia crece con segmentos.
- Vacuum “rewrite whole file”: imposible a gran escala.
- Falta WAL/segmentos/runs con compactación incremental y checksums.
- Falta cuantización y SIMD sistemático.

La evolución debe:

- Introducir **IVF** (coarse clustering).
- En cada cluster: **flat scan cuantizado Q8** (int8) con **AVX2**.
- Persistencia: **runs segmentados + manifest + snapshot + compaction incremental**.
- Mantener búsqueda paralela por cluster/segmentos.
- Minimizar RAM con Q8 y con layout de memoria/caché coherente.

---

## 3) Definición de “IVF_FLAT_Q8” (modo default)

### Componentes

1. **IVF (coarse partition)**: k clusters (centroides) por colección.
2. **Asignación**: cada vector se asigna al cluster más cercano (o top-2 si se habilita para recall).
3. **Consulta**:
   - elegir `nprobe` clusters más cercanos al query,
   - escanear solo esos clusters,
   - scoring con **Q8 int8 dot** (o cosine si normalizas; recomendado dot con vectores normalizados),
   - seleccionar top-k.
4. **Refinamiento opcional**:
   - re-score top `R` (ej. R=4k o 10k) con f32 si está disponible o se almacena un “refine store”.
   - si no, entregar Q8 directo.

### Por qué Q8

- Reduce RAM ~4× vs f32 (1 byte vs 4 bytes por dimensión).
- Acelera dot product con AVX2 (trabajas con bytes, mejor caché).
- Hace viable 50M en máquinas grandes (RAM + SSD) con persistencia eficiente.

---

## 4) Configuración y feature flags (debes implementar/usar)

### Variables de entorno sugeridas (y/o flags CLI)

**Índice**

- `INDEX_KIND=IVF_FLAT_Q8` (default)
- `IVF_CLUSTERS=4096` (ajustable; 1024/2048/4096/8192)
- `IVF_NPROBE=16` (8/16/32)
- `IVF_TRAINING_SAMPLE=200000` (muestra para kmeans si data grande)

**Cuantización**

- `Q8_ENABLED=1` (siempre en IVF_FLAT_Q8)
- `Q8_MODE=per_vector` (opciones: `per_vector`, `per_block`)
- `Q8_BLOCK=64` (si per_block)

**Persistencia**

- `DATA_DIR=...`
- `RUN_TARGET_BYTES=134217728` (128MB por run)
- `RUN_RETENTION=8` (runs a retener antes de compactar)
- `COMPACTION_TRIGGER_TOMBSTONE_RATIO=0.2`
- `COMPACTION_MAX_BYTES_PER_PASS=1073741824` (1GB por pasada)
- `SNAPSHOT_INTERVAL_SECS=30` (ya existe; úsalo como checkpoint real)

**CPU/Paralelismo**

- `SEARCH_THREADS=0` (0=auto)
- `PARALLEL_PROBE=1` (buscar clusters en paralelo)
- `SIMD_ENABLED=1` (auto; con fallback)

**Seguridad**

- `RUSTKISS_API_KEY=...` (si está presente, el middleware debe exigir Bearer)
- `CORS_ALLOWED_ORIGINS=...` (sin wildcard en prod)

---

## 5) Plan por fases (CPU-first) — default IVF_FLAT_Q8

### Fase 0 — Baseline obligatorio (bench + tests básicos)

**Entregables**

- Bench reproducible offline:
  - genera N vectores, dims de la lista, metas simples,
  - mide: p50/p95/p99, QPS, RAM aproximada, tamaño en disco.
- Tests de roundtrip de persistencia (write→close→open→search).

**Criterio de “OK”**

- Bench corre con `cargo run -- bench ...` o `cargo test --features bench`.
- Sin regresiones evidentes.

---

### Fase 1 — Paralelismo + SIMD f32 (ganancia rápida sin romper nada)

**Entregables**

- Paralelizar búsqueda:
  - por segmentos actuales o por clusters en IVF.
- SIMD para dot/cos f32 con fallback correcto.
- Tests: `simd_score ~ scalar_score`.

**Criterio de “OK”**

- p95 baja (o throughput sube) en dims 768/1536/4096.
- Tests de equivalencia pasan.

---

### Fase 2 — IVF coarse (clustering real)

**Entregables**

- Implementar kmeans (preferible kmeans++ init) para centroides por colección.
- Mantener centroides persistidos en disco:
  - `centroids.bin` + `centroids.json` (dim, k, metric, q8 config).
- Query: escoger top `nprobe` centroides y buscar solo ahí.
- Insert/update: asignar cluster; actualizar centroid incremental (opcional) o batch.

**Criterio de “OK”**

- Query no toca todos los datos: solo `nprobe`.
- Persistencia de centroides roundtrip estable.

---

### Fase 3 — Q8 flat scan (modo default IVF_FLAT_Q8)

**Entregables**

- Cuantización Q8 persistente:
  - almacenar por cluster vectores como `i8[]` + `scale` (por vector o por bloque).
- Scoring Q8:
  - dot(i8,i8) → i32 → f32 escalado.
- Fallback (sin AVX2): dot scalar correcto.
- (Opcional) refine top-R con f32 (si decides almacenar f32 en disco separado).

**Criterio de “OK”**

- RAM baja significativamente vs f32.
- Query latencia estable con datos grandes.
- Persistencia y recovery correctos.

---

### Fase 4 — Persistencia por runs/segmentos + compactación incremental (anti corrupción)

**Entregables**

- Reemplazar `vectors.bin` único por “runs”:
  - `run-000001.log`, `run-000002.log`, etc.
- Cada record con:
  - header fijo + checksum + longitud.
- Manifest atómico:
  - lista de runs, offsets aplicados, conteos, último checkpoint.
- Compactación incremental:
  - merge de runs + eliminación de tombstones,
  - por pasadas limitadas (no reescribir 50M de golpe),
  - swap atómico con rename.
- Snapshot real:
  - checkpoint que reduce replay (estructura mínima para reconstrucción).

**Criterio de “OK”**

- Crash safety:
  - si se corta al final del archivo, se trunca seguro.
  - no se “rompe” lectura completa.
- Vacuum incremental no bloquea todo.

---

### Fase 5 — AVX2 int8 dot (optimización fuerte)

**Entregables**

- Implementación AVX2 con runtime detect:
  - `is_x86_feature_detected!("avx2")`
- Tests de equivalencia exacta i32:
  - avx2 vs scalar para dims (384..8192) con datos aleatorios.
- (Opcional) unrolling y prefetch para más performance.

**Criterio de “OK”**

- Reducción visible de latencia en Q8 scan.
- Ningún fallo por CPU sin AVX2 (fallback).

---

### Fase 6 — Hardening (madurez de motor)

**Entregables**

- Checksums, truncado seguro, validación de consistencia.
- Tests de “crash simulation” (si no puedes matar proceso, simula tail corrupta).
- Documentación de garantías y límites.

**Criterio de “OK”**

- Recovery repetible.
- Sin corrupción silenciosa.
- Bench y tests corren en CI.

---

### Fase 7 — DiskANN/Vamana (solo cuando todo lo anterior esté estable)

**Entregables**

- Diseñar e implementar builder offline + índice en disco:
  - grafo en SSD, vectores comprimidos en RAM.
- Cache de páginas y control de IO.
- Mantener API: `INDEX_KIND=DISKANN`.

**Criterio de “OK”**

- Comparables a competidores en escalas enormes.
- Pruebas de consistencia y benchmarks.

---

## 6) Diseño modular (obligatorio para no reescribir todo)

### Interfaces (traits) sugeridas

- `ScorerF32` (dot/cos f32, SIMD y fallback)
- `QuantizerQ8` (f32 → Q8Vec, persistencia)
- `ScorerQ8` (dot(i8,i8) AVX2/fallback)
- `IndexKind`:
  - `HNSW` (compat)
  - `IVF_FLAT_Q8` (default)
  - `IVF_HNSW` (futuro)
  - `DISKANN` (futuro)

### Layout persistente por colección (sugerido)

```

DATA_DIR/
vectors/ <collection>/
manifest.json
centroids.bin
centroids.json
clusters/
c0000/
run-000001.log
run-000002.log
manifest.json
c0001/
...

```

---

## 7) Algoritmos — ejemplos listos (para no perder tiempo)

### 7.1 KMeans++ (entrenamiento IVF)

**Pseudocódigo**

1. Elegir 1er centro al azar.
2. Para cada punto x: D(x)=dist(x, centro_más_cercano).
3. Elegir siguiente centro con prob proporcional a D(x)^2.
4. Repetir hasta k centros.
5. Refinar con Lloyd:
   - asignar cada punto al centro más cercano
   - recalcular centroides como media
   - iterar `iters` o hasta convergencia

**Ejemplo Rust (estructura)**

```rust
pub fn kmeans_pp_train(vectors: &[Vec<f32>], k: usize, iters: usize) -> Vec<Vec<f32>> {
    // NOTA: para datos enormes, samplea (IVF_TRAINING_SAMPLE)
    // 1) init kmeans++
    // 2) iters Lloyd
    // 3) retorna centroides
    unimplemented!()
}
```

### 7.2 Selección de clusters (nprobe)

**Idea**

- score(centroid, query) para todos los centroides (k suele ser 1024..8192)
- tomar top `nprobe`
- buscar solo esos clusters

```rust
pub fn select_probes(centroids: &[Vec<f32>], query: &[f32], nprobe: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = centroids.iter()
        .enumerate()
        .map(|(i, c)| (i, dot_f32(c, query)))
        .collect();
    scored.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(nprobe).map(|x| x.0).collect()
}
```

### 7.3 Cuantización Q8 (per-vector scale, simétrica)

**Definición**

- `scale = max(|x|) / 127`
- `q[i] = clamp(round(x[i]/scale), -127..127) as i8`

```rust
#[derive(Clone)]
pub struct Q8Vec {
    pub scale: f32,
    pub data: Vec<i8>,
}

pub fn quantize_q8_per_vector(v: &[f32]) -> Q8Vec {
    let mut max_abs = 0.0f32;
    for &x in v { max_abs = max_abs.max(x.abs()); }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

    let mut data = Vec::with_capacity(v.len());
    for &x in v {
        let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
        data.push(q);
    }
    Q8Vec { scale, data }
}
```

### 7.4 Dot Q8 escalado

- `raw = dot_i8(a.data, b.data)` en i32
- `score = raw as f32 * (a.scale * b.scale)`

```rust
pub fn dot_q8_scaled(a: &Q8Vec, b: &Q8Vec) -> f32 {
    let raw = dot_i8(&a.data, &b.data) as f32;
    raw * (a.scale * b.scale)
}
```

### 7.5 AVX2 dot(i8,i8) con fallback (ejemplo base)

```rust
#[inline]
pub fn dot_i8_fallback(a: &[i8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc += (a[i] as i32) * (b[i] as i32);
    }
    acc
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { dot_i8_avx2(a, b) }
    } else {
        dot_i8_fallback(a, b)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_si256();
    let mut i = 0usize;

    while i + 32 <= a.len() {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        let prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        let prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

        let ones = _mm256_set1_epi16(1);
        let acc_lo = _mm256_madd_epi16(prod_lo, ones);
        let acc_hi = _mm256_madd_epi16(prod_hi, ones);

        sum = _mm256_add_epi32(sum, acc_lo);
        sum = _mm256_add_epi32(sum, acc_hi);

        i += 32;
    }

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum);
    let mut acc = tmp.iter().sum::<i32>();

    while i < a.len() {
        acc += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    acc
}
```

### 7.6 Flat scan dentro del cluster (top-k)

- Escanear Q8 vectors del cluster y mantener heap top-k

```rust
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug)]
struct Hit { id: u32, score: f32 }
impl PartialEq for Hit { fn eq(&self, o: &Self) -> bool { self.score == o.score } }
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { o.score.partial_cmp(&self.score) } // min-heap via reverse
}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
}

pub fn flat_scan_topk(cluster: &[(u32, Q8Vec)], q: &Q8Vec, k: usize) -> Vec<(u32, f32)> {
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (id, v) in cluster {
        let s = dot_q8_scaled(v, q);
        heap.push(Hit { id: *id, score: s });
        if heap.len() > k { heap.pop(); }
    }
    let mut out = heap.into_sorted_vec();
    out.reverse();
    out.into_iter().map(|h| (h.id, h.score)).collect()
}
```

---

## 8) Persistencia robusta (runs segmentados + checksum + truncado seguro)

### Problema a resolver

El formato append-only actual con un archivo único no escala para:

- compaction incremental,
- recovery rápido,
- corrupción en tail,
- merges parciales.

### Record format recomendado (simple y seguro)

**Header fijo** (little-endian):

- `MAGIC u32` (ej. 0x524B5631 = "RKV1")
- `VERSION u16`
- `FLAGS u16` (op, reserved)
- `LEN u32` (payload bytes)
- `CRC32 u32` (payload)
- `OFFSET u64` (monótono)
- `ID_LEN u16`
- `...` (payload)

Payload:

- `id bytes`
- `vector_q8 bytes` (dim)
- `scale f32`
- `meta bytes` (JSON o msgpack) + `meta_len u32`

**Truncado seguro**

- Al leer:
  - si no hay header completo -> fin (ok)
  - si LEN excede límite -> fin (ok)
  - si CRC falla -> truncar desde el último offset válido (o parar y marcar corrupto, según política)
  - nunca “panic” por tail parcial

### Compactación incremental (LSM-like)

- Cada cluster mantiene runs (archivos) append-only.
- Cuando:
  - tombstones/updates superan ratio,
  - o runs superan N,
  - o bytes exceden umbral,
    entonces:
  - elegir subset de runs antiguos,
  - merge en un nuevo run compacto (solo última versión por id),
  - swap atómico (rename),
  - actualizar manifest cluster.

**NUNCA** reescribir todo el dataset en un solo pass.

---

## 9) Concurrencia y uso de CPU (realista para 50M)

### Lecturas

- Query:
  - escoger probes (centroids)
  - paralelizar por cluster (rayon) cuando `nprobe` >= 8 y CPU lo permite
  - top-k global: merge por heaps/partial sorts

### Escrituras

- Upsert:
  - asignación cluster -> lock solo del cluster (no lock global)
  - append run -> flush según política (fsync configurable)
  - actualizar índices in-memory (si existen) sin bloquear lecturas excesivamente

### Locks recomendados

- Evitar un `RwLock<HashMap<collection,...>>` gigante para el hot path.
- Preferir:
  - lock por colección,
  - lock por cluster,
  - o sharded locks (ej. 64 shards por hash(id)).

---

## 10) Seguridad (mínimo maduro)

Si `RUSTKISS_API_KEY` o `API_KEY` está configurado:

- Exigir `Authorization: Bearer <token>` en **todas** las rutas (incluye SSE).
- Rechazar si falta header o token incorrecto.
- Nunca loguear API keys ni headers sensibles.

Modo local:

- `bind=127.0.0.1` puede permitir open routes (opcional), pero debe ser explícito en config.
  Modo protegido:
- `--bind 0.0.0.0` o `--unsafe-bind` obliga a usar proxy o API key.

---

## 11) Benchmarks y tests (no opcional)

### Tests obligatorios

1. `persist_roundtrip_q8`:
   - insertar N, cerrar, abrir, buscar, verificar IDs/metas.

2. `tail_truncation_safe`:
   - simular archivo truncado en medio de record, verificar que open no corrompe ni peta.

3. `checksum_detection`:
   - corromper bytes y verificar detección.

4. `avx2_vs_scalar_exact`:
   - si hay AVX2, comparar i32 exacto.

5. `dims_matrix`:
   - dims: 384..8192, dataset pequeño, todo pasa.

### Bench mínimo (reproducible)

- dims: 768 y 4096 (mínimo)
- N: 100k, 1M (si posible)
- métricas:
  - latencia p50/p95/p99
  - QPS
  - RAM aproximada
  - disco (tamaño runs)

- comparar:
  - f32 scan (baseline)
  - Q8 scan
  - IVF nprobe 8/16/32

---

## 12) Checklist de “listo para DiskANN/Vamana”

NO avances a DiskANN si no se cumple:

- Persistencia por runs estable con checksum y truncado seguro.
- Compaction incremental en background y sin corrupción.
- IVF_FLAT_Q8 default estable (tests + bench).
- AVX2 + fallback verificados.
- Configuración/flags claros y documentados.
- Bench comparable y repetible.

Luego DiskANN:

- builder offline,
- grafo en disco,
- vectores comprimidos en RAM,
- cache/paging,
- updates (definir si batch o incremental).

---

## 13) Tu tarea inmediata (orden estricto)

1. Implementar y dejar default `INDEX_KIND=IVF_FLAT_Q8`.
2. Añadir kmeans++ + persistencia de centroides.
3. Añadir Q8 persistente por cluster + flat scan top-k.
4. Paralelizar probes + SIMD donde aplique.
5. Migrar persistencia a runs segmentados con checksum + truncado seguro.
6. Compactación incremental por clusters/runs (no full rewrite).
7. AVX2 int8 dot + tests de equivalencia exacta.
8. Hardening (crash/tail/corruption).
9. Solo después: plan e implementación DiskANN/Vamana.

No improvises: cada paso es un PR pequeño con tests y bench.
Git: Tienes permitido utilizar comandos git para versionar tu trabajo.

---

## 14) Fase actual: Preparaci¢n DiskANN/Vamana

1. **Interfaces claras**
   - `VectorIndex` para los ¡ndices en RAM (HNSW, IVF/Q8).
   - `DiskVectorIndex` cubre la inicializaci¢n/sync gen‚rica en disco.
   - `DiskAnnIndex` extiende lo anterior con operaciones espec¡ficas (construir/cargar/borrar el grafo, exponer progreso/par metros).
2. **Manifest dedicado**
   - A¤adir a `manifest.json` un bloque estable con metadatos de grafo (versi¢n, rutas, timestamp, par metros del builder).
   - Los tests de roundtrip deben verificar que esos campos sobreviven restart incluso cuando no hay grafo.
3. **M¢dulo `vector/diskann`**
   - Crear carpeta con archivos peque¤os (`mod.rs`, `builder.rs`, `graph.rs`, `io.rs`, `scheduler.rs`) en lugar de inflar `vector/mod.rs`.
   - Promover utilidades comunes (I/O at¢mico, checksums, lecturas chunked) a `src/utils/` cuando se repitan.
4. **Bench sin castigar SSD**
   - `bench.rs` debe permitir generar el dataset completo una sola vez por dim y reutilizarlo para baseline/IVF/DiskANN.
   - Solo borrar el directorio al final (salvo `--keep-data`); registrar timestamps de inicio/fin ya implementados.
5. **Documentaci¢n**
   - Actualizar `docs/BENCHMARKS.md` y README con nuevos flags/modos.
   - Guardar outputs textuales con `nprobe`, `centroid_count`, `q8_refine_topk`, etc.

La fase se considera cerrada cuando:
- El trait `DiskAnnIndex` exista y tenga un esqueleto funcional.
- El manifest soporte metadatos de ¡ndice en disco sin romper colecciones viejas.
- `bench` pueda ejecutar baseline/IVF/DiskANN reutilizando datasets.
- Haya docs/tests que expliquen c¢mo habilitar validar DiskANN.
