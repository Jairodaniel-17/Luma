# Modelo de amenazas — Luma

Complemento de `SECURITY.md` (que dice *cómo configurar seguro*); este documento
dice **qué protegemos, de quién y con qué**. Se revisa en cada release que toque
superficie de red o formato de datos.

## Activos

| Activo | Dónde vive |
|---|---|
| Datos de clientes (documentos, KV, colas, vectores, memoria) | `data_dir` (WAL, redb, archivos de cola, SQLite) + backups |
| Credenciales | api keys (hash), contraseñas (Argon2id), tokens de sesión (solo hash SHA-256), `LUMA_MASTER_KEY` |
| Aislamiento entre organizaciones | prefijos de keyspace + checks de tenant en cada ruta |
| Disponibilidad del servicio | proceso único (hasta W2 de SPEC-producto) |
| Integridad de la cadena de suministro | crates (`deny.toml`), imagen de contenedor, binarios de release |

## Actores y superficies

| Actor | Superficie | Qué intenta |
|---|---|---|
| Anónimo en la red | HTTP/S (y RESP cuando exista F1) | leer/escribir sin auth, DoS, explotar el parser |
| Tenant autenticado malicioso | API completa de su org | cruzar al keyspace de otra org, agotar recursos compartidos, escalar de rol |
| Api key filtrada | todo lo que la key permita | exfiltración, borrado |
| Atacante con acceso al disco/backup | `data_dir`, bucket de backups | leer datos en reposo |
| Dependencia comprometida | build | ejecutar código en el binario |
| Operador con error honesto | CLI/config | borrar o exponer sin querer |

## Amenazas y controles (existentes → planificados)

**T1 — Acceso sin autenticación.**
Controles: bind por defecto `127.0.0.1`; exponer exige flag explícito; api key
activa Bearer en todas las rutas (SSE incluido). RESP: sin AUTH solo
`PING/HELLO/AUTH/QUIT` (SPEC-resp D3).
Pendiente: negar arranque expuesto sin auth salvo flag `--unsafe-*` (paridad
con el patrón de bind ya existente).

**T2 — Cruce entre organizaciones (el riesgo nº 1 del multi-tenant).**
Controles: `TenantContext` en cada ruta; keyspace RESP con prefijo de org
(SPEC-resp D4); `KEYS/SCAN/PUBLISH` filtran por tenant.
Pendiente: tests de aislamiento como suite de certificación permanente en CI —
entra con SPEC-resp 1.2.

**T3 — Agotamiento de recursos (DoS de vecino ruidoso o externo).**
Controles: `MAX_BODY_BYTES`, `MAX_JSON_BYTES`, límites de dimensión/k/longitud,
rate limiting opt-in, timeouts, backpressure SSE (desconectar clientes lentos).
Pendiente: cuotas por org (SPEC-producto 5.2), `resp_max_clients` + buffers
acotados (SPEC-resp 4.1), rate limit por defecto ≠ 0 en perfil producción.

**T4 — Explotación del parser (HTTP ya lo cubre axum; RESP es nuevo).**
Controles planificados: fuzzing continuo del parser RESP con corpus versionado
(SPEC-resp 1.1 y 4.5); límites de frame antes de asignar memoria.

**T5 — Robo de datos en reposo.**
Controles: cifrado en reposo con `LUMA_MASTER_KEY` (arranque endurecido,
SPEC-roadmap M1.1); tokens/claves nunca en claro.
Pendiente: backups remotos cifrados con la misma llave (SPEC-producto 1.3);
documentar rotación de master key como runbook (hoy: limitación conocida).

**T6 — Robo o abuso de credenciales.**
Controles: Argon2id; revocación de sesiones (M1.2); auditoría de login/admin;
roles owner/admin/member/viewer.
Pendiente: la revocación de api key corta conexiones RESP vivas en el
siguiente comando (SPEC-resp 1.2); alerta de fallos de AUTH repetidos
(`resp_auth_failures_total`, SPEC-resp 1.4).

**T7 — Pérdida o corrupción de datos (el atacante también puede ser un disco).**
Controles: WAL con checksum por registro, recuperación de prefijo válido,
métricas de corrupción; backups con retención.
Pendiente: matriz de crash-recovery en CI, fsync auditado por motor, backup
remoto verificado, WAL shipping (SPEC-producto W1–W2). El fixture dorado de
datos por release vigila la compatibilidad (política de compatibilidad, SPEC-producto).

**T8 — Cadena de suministro.**
Controles: `deny.toml` (cargo-deny) ya en el repo; dependencias mínimas.
Pendiente: `cargo audit` en CI, imagen firmada (cosign) + SBOM por release,
política de `unsafe` (SPEC-producto 5.3).

**T9 — Error del operador.**
Controles: `FLUSHDB` solo tras `resp_allow_flush = true`; borrado de colección
por API con rol; auditoría.
Pendiente: `luma restore` exige confirmación explícita cuando el destino no
está vacío; runbooks de operación (SPEC-producto 5.5).

## Fuera del modelo (supuestos)

- El host y el kernel son de confianza (Luma no defiende contra root local).
- La red interna entre proxy y Luma es de confianza salvo que se active TLS.
- Side-channels de hardware (Spectre y familia) fuera de alcance.

## Proceso

- Cambio que agregue superficie de red, formato en disco o rol nuevo ⇒ este
  documento se actualiza en el mismo PR.
- Reporte de vulnerabilidades: ver `SECURITY.md`.
