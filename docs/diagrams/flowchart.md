# Flujo principal de Luma: ingesta híbrida y búsqueda con planificador

_tipo: flowchart_ · _origen: 4c382dee762e_

```mermaid
flowchart TD
  ini(["Cliente HTTP"]) --> auth["auth_middleware:<br/>resolver Bearer / API key / sesión"]
  auth --> authok{"¿Credencial válida?"}
  authok -->|No| e401[/"401 No autorizado"/]
  authok -->|Sí| tenant["tenant_isolation_middleware:<br/>resolver org / TenantContext"]
  tenant --> tok{"¿Colección pertenece al org?"}
  tok -->|No| e404[/"404 No encontrado"/]
  tok -->|Sí| ruta{"¿Qué operación /v1/db?"}

  subgraph "Ingesta (POST /v1/db/{ns}/ingest)"
    ing["ingest_document"] --> store["Guardar doc fuente en KV state"]
    store --> chunk["Chunking: split_text"]
    chunk --> vac{"¿Genera chunks?"}
    vac -->|No| okvacio["Fin: nada que indexar"]
    vac -->|Sí| embed["embed_batch (proveedor de embeddings)"]
    embed --> eok{"¿Embeddings OK?"}
    eok -->|No| rbdoc["Borrar doc del KV + registrar fallo"]
    rbdoc --> eerr[/"500 Error de embedding"/]
    eok -->|Sí| ensure["ensure_collection + ensure_sqlite_table"]
    ensure --> sqlw["Escribir metadatos en SQLite (primero)"]
    sqlw --> vecw["vector_upsert por cada chunk"]
    vecw --> vok{"¿Upsert vectorial OK?"}
    vok -->|No| rback["Rollback: borrar vectores + fila SQL + KV"]
    rback --> eerr
    vok -->|Sí| idx["Encolar auto-índices de metadatos"]
    idx --> okok[/"200 Ingesta completa"/]
  end

  subgraph "Búsqueda híbrida (POST /v1/db/{ns}/search)"
    srch["search_with_plan"] --> csize["Leer tamaño de colección"]
    csize --> plan{"plan_query:<br/>¿hay sql_filter?"}
    plan -->|No| vf1["VectorFirst (sin filtro)"]
    plan -->|Sí| insp["inspect_sql_filter:<br/>estimar coincidencias"]
    insp --> empty{"¿0 coincidencias?"}
    empty -->|Sí| sfempty["SqlFirst: resultado vacío"]
    empty -->|No| selec{"¿Selectivo o set pequeño<br/>y acotado?"}
    selec -->|Sí| sf["SqlFirst: pre-filtro SQL,<br/>luego vector"]
    selec -->|No| vf2["VectorFirst: vector primero,<br/>post-filtro SQL"]
    vf1 --> exec["Ejecutar estrategia +<br/>ranking por coseno"]
    sfempty --> exec
    sf --> exec
    vf2 --> exec
    exec --> hydr["hydrate_ranked_documents:<br/>traer docs fuente"]
    hydr --> res[/"200 Resultados + plan + diagnósticos"/]
  end

  ruta -->|Ingesta| ing
  ruta -->|Búsqueda| srch
```

Flujo derivado de src/api/auth.rs (auth_middleware) y src/api/mod.rs (tenant_isolation_middleware), con el núcleo de negocio en src/engine/hub.rs: LumaDatabase::ingest_document (chunking -> embed_batch -> escritura SQLite-primero -> vector_upsert con rollback) y search_with_plan/plan_query (planificador que elige SqlFirst vs VectorFirst según selectividad del filtro SQL, luego hidratación de documentos). Rutas en src/api/routes_hub.rs (/v1/db). Se eligió el nivel 2 (LumaDatabase) por ser el flujo de negocio más representativo; se omiten los flujos primitivos (nivel 1) y NS-Mem (nivel 3) por claridad.


<!-- tooling:diagram
{"has_content": true, "title": "Flujo principal de Luma: ingesta híbrida y búsqueda con planificador", "mermaid": "flowchart TD\n  ini([\"Cliente HTTP\"]) --> auth[\"auth_middleware:<br/>resolver Bearer / API key / sesión\"]\n  auth --> authok{\"¿Credencial válida?\"}\n  authok -->|No| e401[/\"401 No autorizado\"/]\n  authok -->|Sí| tenant[\"tenant_isolation_middleware:<br/>resolver org / TenantContext\"]\n  tenant --> tok{\"¿Colección pertenece al org?\"}\n  tok -->|No| e404[/\"404 No encontrado\"/]\n  tok -->|Sí| ruta{\"¿Qué operación /v1/db?\"}\n\n  subgraph \"Ingesta (POST /v1/db/{ns}/ingest)\"\n    ing[\"ingest_document\"] --> store[\"Guardar doc fuente en KV state\"]\n    store --> chunk[\"Chunking: split_text\"]\n    chunk --> vac{\"¿Genera chunks?\"}\n    vac -->|No| okvacio[\"Fin: nada que indexar\"]\n    vac -->|Sí| embed[\"embed_batch (proveedor de embeddings)\"]\n    embed --> eok{\"¿Embeddings OK?\"}\n    eok -->|No| rbdoc[\"Borrar doc del KV + registrar fallo\"]\n    rbdoc --> eerr[/\"500 Error de embedding\"/]\n    eok -->|Sí| ensure[\"ensure_collection + ensure_sqlite_table\"]\n    ensure --> sqlw[\"Escribir metadatos en SQLite (primero)\"]\n    sqlw --> vecw[\"vector_upsert por cada chunk\"]\n    vecw --> vok{\"¿Upsert vectorial OK?\"}\n    vok -->|No| rback[\"Rollback: borrar vectores + fila SQL + KV\"]\n    rback --> eerr\n    vok -->|Sí| idx[\"Encolar auto-índices de metadatos\"]\n    idx --> okok[/\"200 Ingesta completa\"/]\n  end\n\n  subgraph \"Búsqueda híbrida (POST /v1/db/{ns}/search)\"\n    srch[\"search_with_plan\"] --> csize[\"Leer tamaño de colección\"]\n    csize --> plan{\"plan_query:<br/>¿hay sql_filter?\"}\n    plan -->|No| vf1[\"VectorFirst (sin filtro)\"]\n    plan -->|Sí| insp[\"inspect_sql_filter:<br/>estimar coincidencias\"]\n    insp --> empty{\"¿0 coincidencias?\"}\n    empty -->|Sí| sfempty[\"SqlFirst: resultado vacío\"]\n    empty -->|No| selec{\"¿Selectivo o set pequeño<br/>y acotado?\"}\n    selec -->|Sí| sf[\"SqlFirst: pre-filtro SQL,<br/>luego vector\"]\n    selec -->|No| vf2[\"VectorFirst: vector primero,<br/>post-filtro SQL\"]\n    vf1 --> exec[\"Ejecutar estrategia +<br/>ranking por coseno\"]\n    sfempty --> exec\n    sf --> exec\n    vf2 --> exec\n    exec --> hydr[\"hydrate_ranked_documents:<br/>traer docs fuente\"]\n    hydr --> res[/\"200 Resultados + plan + diagnósticos\"/]\n  end\n\n  ruta -->|Ingesta| ing\n  ruta -->|Búsqueda| srch", "notes": "Flujo derivado de src/api/auth.rs (auth_middleware) y src/api/mod.rs (tenant_isolation_middleware), con el núcleo de negocio en src/engine/hub.rs: LumaDatabase::ingest_document (chunking -> embed_batch -> escritura SQLite-primero -> vector_upsert con rollback) y search_with_plan/plan_query (planificador que elige SqlFirst vs VectorFirst según selectividad del filtro SQL, luego hidratación de documentos). Rutas en src/api/routes_hub.rs (/v1/db). Se eligió el nivel 2 (LumaDatabase) por ser el flujo de negocio más representativo; se omiten los flujos primitivos (nivel 1) y NS-Mem (nivel 3) por claridad.", "kind": "flowchart", "source_sha": "4c382dee762e8e6e772a6162c327abb7cc4fbf23"}
-->
