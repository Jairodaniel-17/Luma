# SPEC — Roadmap de endurecimiento y producto (post v4.21.1)

Plan ejecutable derivado del análisis de brechas. Cada ítem es **shippable de
forma independiente**, con criterios de aceptación verificables. Se agrupan en
releases (`vX.Y.Z`) que siguen el flujo probado: rama → commit → tag → CI verde
(rustfmt/clippy/test) → build musl en GitHub Actions → deploy a OCI → verificación
E2E en producción.

Convención de estado por ítem: `[ ]` pendiente · `[~]` en curso · `[x]` hecho.

---

## Milestone 1 — Seguridad & operación base  (release **v4.22.0**)
Lo más barato y crítico primero: que la instancia sea segura de operar y no pierda datos.

### 1.1 `[x]` Clave maestra fuerte + arranque endurecido  · impacto ALTO · esfuerzo BAJO
**Objetivo:** dejar de correr con `LUMA_ALLOW_INSECURE=1` y clave de desarrollo.
**Enfoque:**
- Generar una `LUMA_MASTER_KEY` de 32 bytes y persistirla vía drop-in de systemd
  (`/etc/systemd/system/luma.service.d/`), no en `luma.toml`.
- Quitar el override `LUMA_ALLOW_INSECURE=1`.
- Doc de rotación: procedimiento para re-cifrar si se rota (limitación conocida:
  rotación de la master key implica re-cifrar datos → documentar, no automatizar aún).
**Aceptación:**
- El server arranca sin `LUMA_ALLOW_INSECURE` y sin el WARN de clave insegura.
- Health 200, datos existentes legibles (KV/Doc/vector) tras el cambio.
- La key no aparece en `luma.toml` ni en el repo.

### 1.2 `[x]` Gestión de sesiones (listar + cerrar todo)  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** dar control sobre sesiones activas del propio usuario.
**Enfoque (backend `accounts.rs` + `routes_accounts.rs` + `mod.rs`):**
- `list_user_sessions(user_id)` → `[{created_at_ms, expires_at_ms, ...}]` (sin token).
- `revoke_all_sessions(user_id, except?)` → borra todas (o todas menos la actual).
- Endpoints: `GET /v1/auth/sessions`, `POST /v1/auth/sessions/revoke-all`.
- UI: en el perfil/header, "Cerrar sesión en todos los dispositivos".
**Aceptación:** test que crea N sesiones, lista N, revoca todas menos la actual, valida que solo queda 1.

### 1.3 `[ ]` Backups automáticos programados (offsite)  · impacto ALTO · esfuerzo MEDIO
**Objetivo:** que la pérdida del disco no sea pérdida de datos.
**Enfoque:**
- Reusar `backup.rs` (backup/restore ya existen). Script + `systemd timer` diario:
  snapshot de `data/` (state.redb, sqlite/rustkiss.db, vectors/, blobs/, queues/)
  a un tar cifrado, subido a almacenamiento offsite (OCI Object Storage o similar).
- Retención (p. ej. 7 diarios + 4 semanales).
- Runbook de restore.
**Aceptación:** timer activo; un backup generado y **restaurado** en un dir temporal → datos íntegros; documento de restore.

### 1.4 `[x]` Advisories de dependencias  · impacto MEDIO · esfuerzo BAJO  (tarea #23)
**Objetivo:** `cargo audit` / `cargo deny` sin advisories accionables.
**Enfoque:** actualizar crates con advisory; documentar los no-explotables con justificación.
**Aceptación:** jobs "Security Audit" y "Dependency Audit" verdes sin `--allow`.
**Estado:** los 3 advisories vivos (bincode/rustls-pemfile unmaintained transitivos, lru IterMut unsound no-usado) ya están triados y documentados en deny.toml; CI verde. Nada accionable.

---

## Milestone 2 — Multi-org completo  (release **v4.23.0**)
Redondear lo construido en v4.21.x. Todo esfuerzo bajo-medio.

### 2.1 `[x]` Selector de organización en el header  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** que un usuario multi-org cambie de org activa desde la UI.
**Enfoque (UI + `api.ts`):** dropdown en el header que llama `GET /v1/auth/my-orgs`
y `POST /v1/auth/switch-org`; al cambiar, guarda el token rotado y recarga.
**Aceptación:** usuario en 2 orgs cambia y ve datos/rol de la org destino; el token viejo queda inválido.

### 2.2 `[x]` Invitar usuario en un paso  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** invitar por email y que se cree+añada a la org de una.
**Enfoque:** endpoint `POST /v1/admin/orgs/:id/invite {email, role, password?}`:
si el usuario no existe, lo crea (con password temporal o pendiente) y lo añade;
si existe, lo añade. UI: un solo formulario en el panel de miembros.
**Aceptación:** invitar un email nuevo crea usuario + membresía; invitar uno existente solo añade membresía.

### 2.3 `[x]` Auto-registro a org existente por dominio  · impacto MEDIO · esfuerzo MEDIO
**Objetivo:** que registrarse con `@acme.com` caiga en la org de Acme en vez de crear una nueva.
**Enfoque:** mapa `dominio → org_id` (tabla `sys_domain_orgs`); en `register`, si el
dominio está mapeado, crear el usuario como member de esa org en vez de una org nueva.
Config en la pestaña Acceso.
**Aceptación:** con dominio mapeado, un registro nuevo aparece como member de la org destino; sin mapeo, comportamiento actual.

---

## Milestone 3 — Datos & embeddings  (release **v4.24.0**)

### 3.1 `[x]` Borrar colección completa por API  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** limpiar colecciones (hoy quedan dirs huérfanos: `_cap_*`, `_mix_*`).
**Enfoque:** `DELETE /v1/vector/:collection` → borra índice en memoria, `data/vectors/<name>`,
y la fila en `sys_collections`. Con guard de tenant-ownership.
**Aceptación:** crear colección, borrarla, verificar que desaparece de listado y del disco; test.

### 3.2 `[ ]` Guardar modelo/proveedor por colección + validar mismatch  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** que cada colección recuerde con qué modelo/dim se creó y rechazar ingest incompatibles.
**Enfoque:** añadir `embedding_model`/`embedding_dim` a la metadata de la colección
(o a `sys_collections`); en ingest por texto, si el dim del cliente activo ≠ el de la
colección → error claro (no vector corrupto).
**Aceptación:** ingest con modelo de otro dim a una colección existente → 400 explicativo; test.

### 3.3 `[ ]` Re-indexado / migración al cambiar de modelo  · impacto MEDIO · esfuerzo MEDIO-ALTO
**Objetivo:** herramienta que re-embeba una colección al nuevo modelo.
**Enfoque:** endpoint/comando `POST /v1/vector/:col/reindex {target_model}` que recorre
los docs, re-embeba y reescribe el índice con el nuevo dim (job en background, con progreso).
**Aceptación:** colección dim-384 migrada a un modelo dim-768 queda consultable con el nuevo; test con mock.

### 3.4 `[ ]` Aplicar config sin reiniciar (hot-reload embeddings/LLM)  · impacto MEDIO · esfuerzo MEDIO
**Objetivo:** cambiar proveedor/modelo sin `systemctl restart`.
**Enfoque:** `EmbeddingClient` detrás de un `ArcSwap`/`RwLock`; `PUT /v1/config` reconstruye
el cliente en caliente.
**Aceptación:** cambiar el modelo vía API y ver el nuevo dim en `probe` sin reiniciar; test.

---

## Milestone 4 — DX / integraciones  (release **v4.25.0**)

### 4.1 `[ ]` SDKs (python/ts) + OpenAPI al día  · impacto MEDIO · esfuerzo BAJO
**Objetivo:** exponer los endpoints nuevos (multi-org, probe, sesiones, delete-collection).
**Enfoque:** añadir métodos a `sdk/luma` y `sdk/typescript`; actualizar `docs/openapi.yaml`.
**Aceptación:** los SDKs cubren los endpoints nuevos; `openapi.yaml` valida y los incluye.

---

## Backlog — esfuerzo ALTO / cuando haya tráfico real

### B.1 `[ ]` Quotas por tenant aplicadas  · impacto MEDIO · esfuerzo MEDIO
Hacer cumplir `TenantContext.quotas` (almacenamiento/QPS) además del rate-limit global.

### B.2 `[ ]` SSO empresarial (OIDC → SAML)  · impacto MEDIO · esfuerzo MEDIO  (tarea #22)

### B.3 `[ ]` HA / failover  · impacto ALTO · esfuerzo ALTO
Réplica en espera o replicación; el punto único de fallo actual (un nodo con estado).

### B.4 `[ ]` Escala horizontal de escrituras  · impacto ALTO · esfuerzo ALTO
Sharding por tenant o primario + réplicas de lectura (redb es de un solo escritor).

### B.5 `[~]` Fase 3/4 del motor vectorial  (tareas #27, #28)
Cuantización binaria + segmentos incrementales — ya en curso en ramas propias.

---

## Orden de ejecución
1. Milestone 1 (1.1 → 1.2 → 1.4 → 1.3)  ← empezar aquí
2. Milestone 2 (2.1 → 2.2 → 2.3)
3. Milestone 3 (3.1 → 3.2 → 3.4 → 3.3)
4. Milestone 4 (4.1)
5. Backlog según tráfico/necesidad.

Cada milestone cierra con un release desplegado y verificado en producción.
