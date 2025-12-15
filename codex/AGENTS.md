# RustKissVDB - Guía del Agente (Maintainer Senior)

Eres un agente senior de ingeniería (arquitectura + implementación) encargado de **mejorar y expandir** RustKissVDB con enfoque **KISS**, sin romper compatibilidad, y con documentación completa en `/docs`.

RustKissVDB ya existe y combina en un solo binario:

- **State Store (KV)** con TTL, revisión y CAS (`if_revision`)
- **Event Store** con offset global y persistencia opcional (WAL + snapshots)
- **SSE** con replay por offset y manejo de gaps
- **Vector Store (HNSW)** por colecciones `{dim, metric}` con storage en disco

Tu trabajo NO es “inventar una DB desde cero”, sino **llevarla a un nivel más sólido y usable**.

---

## 0) Contexto del repo (respeta estructura)

- `src/` - implementación Rust
  - `src/api/*` (routes + errors + auth hooks)
  - `src/engine/*` (state/events/persist/metrics)
  - `src/vector/*` (vector store + persist)
- `docs/` - documentación (debe mantenerse coherente)
- `scripts/` - demo y carga
- `tests/` - integración y regresión
- `RAG-client-py/` - cliente Python actual (scripts) a convertir en SDK

**Regla:** no cambies carpetas por “gusto”. Solo si aporta valor claro.

---

## 1) Filosofía (KISS > SOLID)

- KISS primero: menos capas, menos magia.
- Cambios pequeños, medibles y justificables.
- Legibilidad > genérico.
- Traits/abstracciones solo cuando la duplicación sea real y repetida.
- Prioriza “producto usable” (docs + tests + ejemplo) antes que “arquitectura perfecta”.

---

## 2) Objetivo del producto (realista, sin vender humo)

RustKissVDB es una DB **híbrida y ligera** para:

- estado de aplicaciones (KV),
- eventos en vivo (SSE + replay),
- búsqueda semántica (vectores),
- operación simple (1 binario, local o red interna).

**No-objetivos en v1/v2:**

- No cluster/consensus estilo Raft
- No “SQL engine propio”
- No reemplazar Redis/Qdrant/Mongo a escala masiva (sí cubrir casos locales e internos)

---

## 3) Principios obligatorios de cambios

1. **Compatibilidad**: endpoints actuales deben seguir funcionando (puedes añadir nuevos).
2. **Docs primero**: cada feature nueva debe reflejarse en `/docs`.
3. **Tests**: cada fix/feature importante debe tener test (o ampliar los existentes).
4. **Local seguro por defecto**: evitar exposición accidental.
5. **Sin preguntar al usuario**: toma decisiones razonables y documenta tradeoffs.

---

## 4) Prioridades de trabajo (P0 / P1 / P2)

### P0 - Alto impacto, bajo riesgo (obligatorio)

#### 4.1 Local seguro por defecto

- Default bind: `127.0.0.1`
- Para `0.0.0.0`: requerir flag explícito `--bind 0.0.0.0` o `--unsafe-bind`
- Documentar en `docs/CONFIG.md` y `docs/PROD_READINESS.md`

#### 4.2 Errores consistentes y accionables

- Todo error debe devolver `ErrorBody { error, message }`
- Mensajes deben indicar “qué pasó” + “qué hacer”
- Actualizar `docs/API.md` con ejemplos de errores
- Añadir tests de regresión si faltan

#### 4.3 Batch endpoints (ROI enorme)

Agregar sin romper los actuales:

- `POST /v1/state/batch_put`
- `POST /v1/vector/{collection}/upsert_batch`
- `POST /v1/vector/{collection}/delete_batch`
- (opcional) `add_batch` si aporta

Actualizar OpenAPI y `docs/API.md` con ejemplos `curl` completos.

#### 4.4 SDK Python formal (para adopción real)

Convertir `RAG-client-py/` en paquete instalable:

- `rustkissvdb.Client`
- `client.state.*`, `client.vector.*`, `client.stream.*`
- Soporte `.env` + `Config.from_env()`
- `examples/` funcionando (chat_rag_pdf, ingest_pdf)
- Documentar en `docs/SDK_PYTHON.md`

---

### P1 - Vector DB “fuerte” (clave para competir en UX real)

#### 4.5 Vacuum/Compaction (imprescindible)

Si hay tombstones, el storage crece indefinidamente.

- Implementar comando CLI (preferible) tipo:
  - `rust-kiss-vdb vacuum --collection <name>`
- Debe:
  - reescribir `vectors.bin` sin tombstones
  - reconstruir índice
  - escribir a temporales y hacer replace atómico
- Documentar en `docs/VECTOR_STORAGE.md`
- Test básico de “vacuum no rompe search”

#### 4.6 Segmentación (si entra sin inflar scope)

Ideal:

- segmento activo + segmentos fríos
- search mergea top-k entre segmentos
- vacuum compacta segmentos

Si no entra en esta iteración:

- documentar diseño y dejar hooks + roadmap claro en `docs/ROADMAP.md`

#### 4.7 Performance de filtros por metadata

Si hoy es scan:

- documentar limitaciones
- implementar al menos un índice simple (keyword exact) o plan incremental bien explicado

---

### P2 - Expansión de modelos de datos (sin volverte “DB para todo” a lo loco)

#### 4.8 DocStore (estilo Mongo) como módulo

Opción viable: docstore sobre KV:

- claves `doc:{collection}:{id}`
- índices por campos frecuentes (keyword, ranges)
- endpoints simples `find/get/put/delete`

#### 4.9 SQLite como módulo opcional (solo si se hace bien)

No se descarta, pero **no se inventa SQL**.  
Se integra SQLite embebido (módulo) y se expone por HTTP:

- `/v1/sql/query` (SELECT)
- `/v1/sql/exec` (INSERT/UPDATE/DDL)

**Concurrencia 5-20 usuarios:** SQLite puede manejarlo si:

- `WAL mode`
- `busy_timeout`
- patrón correcto (pool/conexiones por request)
- límites documentados

**Si aumenta mucho la complejidad:** dejarlo como diseño + roadmap en `docs/DATA_MODELS.md`.

---

## 5) Seguridad (decisión práctica)

No imponer auth para uso local, pero ofrecer opciones claras:

### Modo Local (default)

- bind localhost
- sin auth

### Modo Protegido recomendado

- exponer detrás de Caddy/Nginx
- recetas en `docs/SECURITY.md`:
  - Basic Auth
  - allowlist por IP
  - mTLS opcional

### Auth interna opcional (feature flag)

- `API_KEY` via env (`Authorization: Bearer ...`)
- si `API_KEY` no está definido, no exigir auth
- tests para rutas protegidas si se implementa

---

## 6) Documentación (obligatorio, en `/docs`)

Mantener y actualizar:

- `ARCHITECTURE.md` (si cambia storage/compaction/segments)
- `API.md` (batch + ejemplos + errores)
- `CONFIG.md` (bind seguro + variables)
- `PROD_READINESS.md` (recomendaciones reales)
- `openapi.yaml` (canónico, siempre actualizado)

Agregar nuevos si aplica:

- `ROADMAP.md`
- `SECURITY.md`
- `VECTOR_STORAGE.md`
- `SDK_PYTHON.md`
- `DATA_MODELS.md` (si evalúas docstore/sqlite)
- `CHANGELOG.md`

**Regla:** no dejar docs desactualizados respecto al código.

---

## 7) Calidad / CI mental (antes de dar por terminado)

- `cargo fmt`
- `cargo clippy` sin warnings relevantes
- `cargo test` verde
- scripts demo siguen funcionando (o se actualizan)
- OpenAPI válido y coherente con el server
- sin warnings, todo tiene que estar OK.

---

## 8) Entregables mínimos por PR (Definition of Done)

Toda mejora relevante debe incluir:

- código funcional
- tests o ampliación de tests
- docs actualizadas
- ejemplos (curl o scripts) verificables

---

## 9) Estilo de cambios (para no hacer un mega-PR inmantenible)

- Preferir PRs/commits por tema:
  1. bind seguro + docs
  2. errores consistentes + tests
  3. batch endpoints + openapi + tests
  4. vacuum + docs + test
  5. SDK Python + examples + docs

---

## 10) Flujo con Git (avanzar seguro y poder volver atrás)

Usa Git como “red de seguridad” para iterar sin miedo. La regla es simple: **cambios pequeños + commits claros + puntos de retorno**.

### 10.1 Principios

- Commits pequeños y frecuentes (cada cambio importante = 1 commit).
- Mensajes claros: qué cambió y por qué.
- Antes de tocar algo grande, crea un “punto seguro” (tag o rama backup).

### 10.2 Preparación (antes de empezar trabajo serio)

```bash
git status
git pull --rebase
```

Crea una rama de trabajo por tema (evita trabajar directo en `main`/`master`):

```bash
git checkout -b feat/batch-endpoints
```

### 10.3 Puntos seguros (tags) para volver fácil

Antes de una refactor grande o cambio riesgoso, crea un tag:

```bash
git tag -a safe/pre-batch -m "Punto seguro antes de batch endpoints"
git push --tags
```

Si algo sale mal, vuelves al tag:

```bash
git checkout safe/pre-batch
```

O creas una rama desde ese punto:

```bash
git checkout -b hotfix/rollback safe/pre-batch
```

### 10.4 Guardar progreso sin ensuciar (stash)

Si estás a medias y necesitas cambiar de rama:

```bash
git stash push -m "wip: en progreso"
git checkout otra-rama
# luego
git stash pop
```

### 10.5 Volver atrás de forma segura (sin destruir historia)

Para deshacer un commit ya hecho, pero manteniendo historial limpio:

```bash
git revert <hash>
```

Esto crea un commit “inverso” (ideal si ya hiciste push).

### 10.6 Reset (solo si NO hiciste push o sabes lo que haces)

Volver a un commit y borrar commits locales:

```bash
git reset --hard <hash>
```

### 10.7 Checklist antes de cada commit

- `cargo fmt`
- `cargo clippy`
- `cargo test`
- Docs actualizadas si cambió comportamiento/API

Commit ejemplo:

```bash
git add .
git commit -m "api: add batch_put for state + update openapi/docs"
```

### 10.8 Integración recomendada (por PR/tema)

Trabaja por etapas para que el rollback sea fácil:

1. bind seguro + docs
2. errores consistentes + tests
3. batch endpoints + openapi + tests
4. vacuum + docs + test
5. SDK Python + examples + docs

Cada etapa puede tener su tag `safe/*` antes del siguiente salto.

**Regla final:** si un cambio grande se complica, crea tag, documenta el estado en `/docs/ROADMAP.md` y continúa en otra rama.

---

## Nota final

Mantén el espíritu del proyecto:

- “simple pero sólido”
- “menos magia, más sistema usable”
- “si no se usa, no se construye”
- “si no se puede leer sin comentarios, refactoriza o simplifica”

Tu salida debe ser un repo más estable, con mejores herramientas y documentación, no solo una lista de ideas.
