# Consolidación de memoria NS-Mem (ingesta → extracción → promoción de hechos)

_tipo: activity_ · _origen: e0cb9aa02235_

```mermaid
flowchart TD
  subgraph AG["Agente / Cliente"]
    ini(["Inicio"]) --> post["POST /v1/memory/{ns}/ingest_event"]
  end
  subgraph SYS["Sistema Luma (MemoryService)"]
    epi["Guardar evento episódico<br/>status=Active"]
    consol{"¿Consolidación habilitada<br/>y kind=Episodic?"}
    dedup{"¿Hecho duplicado?<br/>cosine >= 0.95"}
    prom{"¿confidence >=<br/>promotion_threshold?"}
    upf["upsert_fact"]
    contra{"¿Hecho existe y<br/>cosine < 0.55?"}
    edge["Crear arista TriggeredBy<br/>(episódico -> semántico)"]
  end
  subgraph LLM["Proveedor LLM (openai/ollama/mock)"]
    ext["extract_facts:<br/>FactCandidates"]
  end
  subgraph STO["Almacenamiento (Vector + SQLite/Graph)"]
    knn["Vector search semántica"]
    hist["append_belief_history"]
    ce{"Tipo de arista"}
    sup["Arista Supersedes"]
    con["Arista Contradicts"]
    save["Persistir fact<br/>memory_records"]
    fin(["Fin"])
  end
  post --> epi
  epi --> consol
  consol -->|No| fin
  consol -->|Sí| ext
  ext --> dedup
  dedup --> knn
  knn --> dedup
  dedup -->|Sí| fin
  dedup -->|No| prom
  prom -->|Sí| upf
  prom -->|No| upf
  upf --> contra
  contra --> hist
  hist --> ce
  ce -->|contradicción| con
  ce -->|actualización| sup
  con --> save
  sup --> save
  contra -->|nuevo| save
  save --> edge
  edge --> fin
```

Actores y actividades derivados de src/memory/consolidator.rs (extracción vía LLM, dedup cosine>=0.95, promoción Active/Draft por memory_fact_promotion_threshold, arista TriggeredBy), src/memory/ingest.rs (ingest_event episódico Active; upsert_fact con belief versioning append_belief_history y aristas Contradicts si cosine<0.55 o Supersedes), src/memory/service.rs y src/memory/llm.rs (proveedores none/mock/openai/ollama), y src/api/routes_memory.rs (endpoints /v1/memory/{ns}/ingest_event, upsert_fact, query). Almacenamiento vector + SQLite/graph según CLAUDE.md.


<!-- tooling:diagram
{"has_content": true, "title": "Consolidación de memoria NS-Mem (ingesta → extracción → promoción de hechos)", "mermaid": "flowchart TD\n  subgraph AG[\"Agente / Cliente\"]\n    ini([\"Inicio\"]) --> post[\"POST /v1/memory/{ns}/ingest_event\"]\n  end\n  subgraph SYS[\"Sistema Luma (MemoryService)\"]\n    epi[\"Guardar evento episódico<br/>status=Active\"]\n    consol{\"¿Consolidación habilitada<br/>y kind=Episodic?\"}\n    dedup{\"¿Hecho duplicado?<br/>cosine >= 0.95\"}\n    prom{\"¿confidence >=<br/>promotion_threshold?\"}\n    upf[\"upsert_fact\"]\n    contra{\"¿Hecho existe y<br/>cosine < 0.55?\"}\n    edge[\"Crear arista TriggeredBy<br/>(episódico -> semántico)\"]\n  end\n  subgraph LLM[\"Proveedor LLM (openai/ollama/mock)\"]\n    ext[\"extract_facts:<br/>FactCandidates\"]\n  end\n  subgraph STO[\"Almacenamiento (Vector + SQLite/Graph)\"]\n    knn[\"Vector search semántica\"]\n    hist[\"append_belief_history\"]\n    ce{\"Tipo de arista\"}\n    sup[\"Arista Supersedes\"]\n    con[\"Arista Contradicts\"]\n    save[\"Persistir fact<br/>memory_records\"]\n    fin([\"Fin\"])\n  end\n  post --> epi\n  epi --> consol\n  consol -->|No| fin\n  consol -->|Sí| ext\n  ext --> dedup\n  dedup --> knn\n  knn --> dedup\n  dedup -->|Sí| fin\n  dedup -->|No| prom\n  prom -->|Sí| upf\n  prom -->|No| upf\n  upf --> contra\n  contra --> hist\n  hist --> ce\n  ce -->|contradicción| con\n  ce -->|actualización| sup\n  con --> save\n  sup --> save\n  contra -->|nuevo| save\n  save --> edge\n  edge --> fin", "notes": "Actores y actividades derivados de src/memory/consolidator.rs (extracción vía LLM, dedup cosine>=0.95, promoción Active/Draft por memory_fact_promotion_threshold, arista TriggeredBy), src/memory/ingest.rs (ingest_event episódico Active; upsert_fact con belief versioning append_belief_history y aristas Contradicts si cosine<0.55 o Supersedes), src/memory/service.rs y src/memory/llm.rs (proveedores none/mock/openai/ollama), y src/api/routes_memory.rs (endpoints /v1/memory/{ns}/ingest_event, upsert_fact, query). Almacenamiento vector + SQLite/graph según CLAUDE.md.", "kind": "activity", "source_sha": "e0cb9aa02235c9bd3ece7d850d54f54655178cf1"}
-->
