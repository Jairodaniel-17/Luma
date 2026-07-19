# Consolidación de memoria NS-Mem: de evento episódico a hecho semántico

_tipo: activity_ · _origen: e19e0f9f2d5b_

```mermaid
flowchart TD
  subgraph C["Cliente / Agente"]
    ini(["Inicio"]) --> post["POST /v1/memory/{ns}/ingest_event<br/>texto + entity_id"]
  end
  subgraph S["Sistema Luma (MemoryService / Engine)"]
    persist["Persistir registro episódico<br/>memory_records + embedding"]
    working["Guardar memoria de trabajo<br/>KV con TTL por session_id"]
    guard{"¿Consolidación activa<br/>y kind=Episodic<br/>con entity_id?"}
    dedup{"¿Existe hecho duplicado?<br/>cosine >= 0.95"}
    prom{"¿confidence >=<br/>promotion_threshold?"}
    active["upsert_fact semántico<br/>status = Active"]
    draft["upsert_fact semántico<br/>status = Draft"]
    edge["Crear arista TriggeredBy<br/>episodio --> hecho"]
    emit["Emitir evento<br/>episodic_promoted_to_semantic"]
    finok(["Fin"])
    finskip(["Fin (sin consolidar)"])
  end
  subgraph L["Proveedor LLM (InferenceClient)"]
    extract["extract_facts<br/>heurística u OpenAI/Ollama"]
  end
  post --> persist
  persist --> working
  working --> guard
  guard -->|No| finskip
  guard -->|Sí| extract
  extract --> dedup
  dedup -->|Sí, otro id| finskip
  dedup -->|No| prom
  prom -->|Sí| active
  prom -->|No| draft
  active --> edge
  draft --> edge
  edge --> emit
  emit --> finok
```

Proceso derivado de src/memory/ingest.rs (ingest_event: persist_memory_record, index_memory_record, persist_working_memory), src/memory/consolidator.rs (process: guardas kind=Episodic/entity_id/memory_consolidation_enabled, find_duplicate_fact con umbral 0.95, promoción Active/Draft según memory_fact_promotion_threshold, arista TriggeredBy, emit_consolidation_event) y src/memory/llm.rs (InferenceClient.extract_facts, providers None/Mock/OpenAI/Ollama). Tres actores reales: el Cliente/Agente que llama al endpoint REST /v1/memory/{namespace}/ingest_event, el Sistema Luma que orquesta la persistencia y consolidación, y el Proveedor LLM que extrae los FactCandidate.


<!-- tooling:diagram
{"has_content": true, "title": "Consolidación de memoria NS-Mem: de evento episódico a hecho semántico", "mermaid": "flowchart TD\n  subgraph C[\"Cliente / Agente\"]\n    ini([\"Inicio\"]) --> post[\"POST /v1/memory/{ns}/ingest_event<br/>texto + entity_id\"]\n  end\n  subgraph S[\"Sistema Luma (MemoryService / Engine)\"]\n    persist[\"Persistir registro episódico<br/>memory_records + embedding\"]\n    working[\"Guardar memoria de trabajo<br/>KV con TTL por session_id\"]\n    guard{\"¿Consolidación activa<br/>y kind=Episodic<br/>con entity_id?\"}\n    dedup{\"¿Existe hecho duplicado?<br/>cosine >= 0.95\"}\n    prom{\"¿confidence >=<br/>promotion_threshold?\"}\n    active[\"upsert_fact semántico<br/>status = Active\"]\n    draft[\"upsert_fact semántico<br/>status = Draft\"]\n    edge[\"Crear arista TriggeredBy<br/>episodio --> hecho\"]\n    emit[\"Emitir evento<br/>episodic_promoted_to_semantic\"]\n    finok([\"Fin\"])\n    finskip([\"Fin (sin consolidar)\"])\n  end\n  subgraph L[\"Proveedor LLM (InferenceClient)\"]\n    extract[\"extract_facts<br/>heurística u OpenAI/Ollama\"]\n  end\n  post --> persist\n  persist --> working\n  working --> guard\n  guard -->|No| finskip\n  guard -->|Sí| extract\n  extract --> dedup\n  dedup -->|Sí, otro id| finskip\n  dedup -->|No| prom\n  prom -->|Sí| active\n  prom -->|No| draft\n  active --> edge\n  draft --> edge\n  edge --> emit\n  emit --> finok", "notes": "Proceso derivado de src/memory/ingest.rs (ingest_event: persist_memory_record, index_memory_record, persist_working_memory), src/memory/consolidator.rs (process: guardas kind=Episodic/entity_id/memory_consolidation_enabled, find_duplicate_fact con umbral 0.95, promoción Active/Draft según memory_fact_promotion_threshold, arista TriggeredBy, emit_consolidation_event) y src/memory/llm.rs (InferenceClient.extract_facts, providers None/Mock/OpenAI/Ollama). Tres actores reales: el Cliente/Agente que llama al endpoint REST /v1/memory/{namespace}/ingest_event, el Sistema Luma que orquesta la persistencia y consolidación, y el Proveedor LLM que extrae los FactCandidate.", "kind": "activity", "source_sha": "e19e0f9f2d5beb133329fd21218bfdbdc37872bc"}
-->
