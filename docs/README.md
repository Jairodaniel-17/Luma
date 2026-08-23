# Documentación de Luma

Cuatro carpetas, ordenadas por lo que estás intentando hacer.

## Empezar

Primer contacto: levantar algo y ver que funciona.

| | |
|---|---|
| **[Manual de usuario](../MANUAL_USUARIO.md)** | **Empieza aquí.** Qué soporta Luma, cómo configurarlo bien y qué falla si no |
| [Demo](empezar/DEMO.md) | Recorrido guiado desde cero |
| [Características](empezar/FEATURES.md) | Qué hay dentro, en una página |
| [SDK de Python](empezar/SDK_PYTHON.md) | Cliente Python |

## Operar

Ponerlo en producción y mantenerlo ahí.

| | |
|---|---|
| [Runbooks](operar/RUNBOOKS.md) | Respaldo, restore, réplica, promoción, rotación de clave maestra, upgrade, y una tabla de síntoma → dónde mirar |
| [Configuración](operar/CONFIG.md) | Cada opción de `luma.toml`, su valor por defecto y su efecto |
| [CLI](operar/CLI.md) | Subcomandos |
| [Preparación para producción](operar/PROD_READINESS.md) | Lo que hay que decidir antes de la primera carga real |
| [Seguridad](operar/SECURITY.md) | Modelo de seguridad e inventario de `unsafe` |
| [Modelo de amenazas](operar/THREAT_MODEL.md) | Qué se defiende, contra quién, y qué se acepta |

## Integrar

Hablar con Luma desde una aplicación.

| | |
|---|---|
| [API HTTP](integrar/API.md) | Los tres niveles: primitivas, hub, memoria |
| [RESP (Redis)](integrar/RESP.md) | El protocolo de Redis, con sus divergencias declaradas |
| [API S3](integrar/S3.md) | Almacenamiento de objetos compatible con S3 |
| [Conector de Postgres](integrar/POSTGRES-CDC.md) | CDC por replicación lógica y búsqueda federada |
| [NS-Mem](integrar/NS_MEM.md) | La capa de memoria de agentes |
| [`openapi.yaml`](openapi.yaml) | La superficie HTTP, comprobada contra el router por `tests/openapi_drift.rs` |

## Referencia

Cómo está hecho por dentro.

| | |
|---|---|
| [Arquitectura](referencia/ARCHITECTURE.md) | Los subsistemas y cómo encajan |
| [Modelos de datos](referencia/DATA_MODELS.md) | Qué se guarda y con qué forma |
| [Almacenamiento vectorial](referencia/VECTOR_STORAGE.md) | Segmentos, índices, y el formato en disco |
| [Benchmarks](referencia/BENCHMARKS.md) · [cómo medirlos](referencia/BENCH.md) | Números, y cómo reproducirlos |
| [Changelog](referencia/CHANGELOG.md) | Qué cambió y cuándo |

## Planificación

Estos no son documentación de uso: son el plan del producto y viven en la raíz
de `docs/` por eso.

- [Plan maestro](PLAN-MAESTRO.md) — el estado real de cada bloque, con lo que
  falta dicho en voz alta
- [SPEC de producto](SPEC-producto.md) · [SPEC de RESP](SPEC-resp.md) ·
  [SPEC del roadmap](SPEC-roadmap.md)
- [Roadmap](ROADMAP.md)

---

`tests/docs_links.rs` comprueba que cada enlace relativo de esta carpeta y del
`README.md` de la raíz apunta a un fichero que existe. Un índice con un enlace
roto es peor que no tener índice: manda a buscar algo que parece que debería
estar ahí.
