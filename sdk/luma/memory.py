from typing import Any, Dict, List, Optional

from ._http import Http


class MemoryClient:
    """NS-Mem agent memory layer — episodic, semantic, procedural and working memory."""

    def __init__(self, http: Http, namespace: str):
        self._http = http
        self._ns = f"/v1/memory/{namespace}"

    # ── ingest_event ─────────────────────────────────────────────────────────

    # Store an episodic event. Triggers the LLM consolidation pipeline when enabled.
    def ingest_event(self, text: str, *, id: Optional[str] = None,
                     entity_id: Optional[str] = None, metadata: Optional[Dict] = None,
                     source: Optional[str] = None, session_id: Optional[str] = None,
                     expires_at_ms: Optional[int] = None) -> dict:
        body: Dict = {"text": text}
        _opt(body, id=id, entity_id=entity_id, metadata=metadata,
             source=source, session_id=session_id, expires_at_ms=expires_at_ms)
        return self._http.post(f"{self._ns}/ingest_event", body)

    async def aingest_event(self, text: str, *, id: Optional[str] = None,
                            entity_id: Optional[str] = None, metadata: Optional[Dict] = None,
                            source: Optional[str] = None, session_id: Optional[str] = None,
                            expires_at_ms: Optional[int] = None) -> dict:
        body: Dict = {"text": text}
        _opt(body, id=id, entity_id=entity_id, metadata=metadata,
             source=source, session_id=session_id, expires_at_ms=expires_at_ms)
        return await self._http.apost(f"{self._ns}/ingest_event", body)

    # ── upsert_fact ──────────────────────────────────────────────────────────

    # Create or update a semantic fact. Versions the previous value and creates a Supersedes edge on overwrite.
    def upsert_fact(self, content: str, *, id: Optional[str] = None,
                    entity_id: Optional[str] = None, fact_key: Optional[str] = None,
                    metadata: Optional[Dict] = None, source: Optional[str] = None,
                    confidence: Optional[float] = None, status: Optional[str] = None) -> dict:
        body: Dict = {"content": content}
        _opt(body, id=id, entity_id=entity_id, fact_key=fact_key,
             metadata=metadata, source=source, confidence=confidence, status=status)
        return self._http.post(f"{self._ns}/upsert_fact", body)

    async def aupsert_fact(self, content: str, *, id: Optional[str] = None,
                           entity_id: Optional[str] = None, fact_key: Optional[str] = None,
                           metadata: Optional[Dict] = None, source: Optional[str] = None,
                           confidence: Optional[float] = None, status: Optional[str] = None) -> dict:
        body: Dict = {"content": content}
        _opt(body, id=id, entity_id=entity_id, fact_key=fact_key,
             metadata=metadata, source=source, confidence=confidence, status=status)
        return await self._http.apost(f"{self._ns}/upsert_fact", body)

    # ── upsert_procedure ─────────────────────────────────────────────────────

    # Register or replace a procedural DAG (nodes + typed edges + constraints).
    def upsert_procedure(self, procedure_id: str, name: str,
                         nodes: List[Dict], edges: List[Dict], *,
                         version: Optional[int] = None, status: Optional[str] = None,
                         description: Optional[str] = None, confidence: Optional[float] = None,
                         source: Optional[str] = None,
                         constraints: Optional[List[Dict]] = None) -> dict:
        body: Dict = {"procedure_id": procedure_id, "name": name, "nodes": nodes, "edges": edges}
        _opt(body, version=version, status=status, description=description,
             confidence=confidence, source=source, constraints=constraints)
        return self._http.post(f"{self._ns}/upsert_procedure", body)

    async def aupsert_procedure(self, procedure_id: str, name: str,
                                nodes: List[Dict], edges: List[Dict], *,
                                version: Optional[int] = None, status: Optional[str] = None,
                                description: Optional[str] = None,
                                confidence: Optional[float] = None,
                                source: Optional[str] = None,
                                constraints: Optional[List[Dict]] = None) -> dict:
        body: Dict = {"procedure_id": procedure_id, "name": name, "nodes": nodes, "edges": edges}
        _opt(body, version=version, status=status, description=description,
             confidence=confidence, source=source, constraints=constraints)
        return await self._http.apost(f"{self._ns}/upsert_procedure", body)

    # ── query ────────────────────────────────────────────────────────────────

    # Query memory across episodic, semantic and procedural layers. Mode is auto-detected from query content.
    def query(self, query: str, *, entity_id: Optional[str] = None,
              session_id: Optional[str] = None, procedure_id: Optional[str] = None,
              current_node_id: Optional[str] = None, context: Optional[Dict] = None,
              mode: Optional[str] = None, limit: Optional[int] = None,
              include_evidence: Optional[bool] = None, include_plan: Optional[bool] = None,
              include_diagnostics: Optional[bool] = None) -> dict:
        body: Dict = {"query": query}
        _opt(body, entity_id=entity_id, session_id=session_id,
             procedure_id=procedure_id, current_node_id=current_node_id,
             context=context, mode=mode, limit=limit,
             include_evidence=include_evidence, include_plan=include_plan,
             include_diagnostics=include_diagnostics)
        return self._http.post(f"{self._ns}/query", body)

    async def aquery(self, query: str, *, entity_id: Optional[str] = None,
                     session_id: Optional[str] = None, procedure_id: Optional[str] = None,
                     current_node_id: Optional[str] = None, context: Optional[Dict] = None,
                     mode: Optional[str] = None, limit: Optional[int] = None,
                     include_evidence: Optional[bool] = None, include_plan: Optional[bool] = None,
                     include_diagnostics: Optional[bool] = None) -> dict:
        body: Dict = {"query": query}
        _opt(body, entity_id=entity_id, session_id=session_id,
             procedure_id=procedure_id, current_node_id=current_node_id,
             context=context, mode=mode, limit=limit,
             include_evidence=include_evidence, include_plan=include_plan,
             include_diagnostics=include_diagnostics)
        return await self._http.apost(f"{self._ns}/query", body)

    # ── next_step ────────────────────────────────────────────────────────────

    # Return the next valid node in a procedure given current_node_id and evaluation context.
    def next_step(self, procedure_id: str, *, current_node_id: Optional[str] = None,
                  context: Optional[Dict] = None) -> dict:
        body: Dict = {"procedure_id": procedure_id}
        _opt(body, current_node_id=current_node_id, context=context)
        return self._http.post(f"{self._ns}/next_step", body)

    async def anext_step(self, procedure_id: str, *, current_node_id: Optional[str] = None,
                         context: Optional[Dict] = None) -> dict:
        body: Dict = {"procedure_id": procedure_id}
        _opt(body, current_node_id=current_node_id, context=context)
        return await self._http.apost(f"{self._ns}/next_step", body)

    # ── timeline ─────────────────────────────────────────────────────────────

    # Fetch the episodic timeline for an entity, ordered chronologically.
    def timeline(self, entity_id: str) -> dict:
        return self._http.get(f"{self._ns}/timeline/{entity_id}")

    async def atimeline(self, entity_id: str) -> dict:
        return await self._http.aget(f"{self._ns}/timeline/{entity_id}")

    # ── graph edges ──────────────────────────────────────────────────────────

    # Create a typed weighted edge between two memory records.
    def create_edge(self, source_id: str, target_id: str, edge_type: str, *,
                    id: Optional[str] = None, weight: Optional[float] = None,
                    metadata: Optional[Dict] = None) -> dict:
        body: Dict = {"source_id": source_id, "target_id": target_id, "edge_type": edge_type}
        _opt(body, id=id, weight=weight, metadata=metadata)
        return self._http.post(f"{self._ns}/edges", body)

    async def acreate_edge(self, source_id: str, target_id: str, edge_type: str, *,
                           id: Optional[str] = None, weight: Optional[float] = None,
                           metadata: Optional[Dict] = None) -> dict:
        body: Dict = {"source_id": source_id, "target_id": target_id, "edge_type": edge_type}
        _opt(body, id=id, weight=weight, metadata=metadata)
        return await self._http.apost(f"{self._ns}/edges", body)

    # Return all outgoing and incoming edges for a memory node.
    def node_edges(self, memory_id: str) -> dict:
        return self._http.get(f"{self._ns}/edges/{memory_id}")

    async def anode_edges(self, memory_id: str) -> dict:
        return await self._http.aget(f"{self._ns}/edges/{memory_id}")

    # Delete a memory edge by its ID.
    def delete_edge(self, edge_id: str) -> dict:
        return self._http.post(f"{self._ns}/edges/{edge_id}/delete")

    async def adelete_edge(self, edge_id: str) -> dict:
        return await self._http.apost(f"{self._ns}/edges/{edge_id}/delete")

    # Return the full version history of a belief (semantic fact) by fact_key.
    def belief_history(self, fact_key: str) -> dict:
        return self._http.get(f"{self._ns}/beliefs/{fact_key}/history")

    async def abelief_history(self, fact_key: str) -> dict:
        return await self._http.aget(f"{self._ns}/beliefs/{fact_key}/history")

    # Recompute PageRank centrality scores for all nodes in the namespace.
    def refresh_centrality(self) -> dict:
        return self._http.post(f"{self._ns}/graph/centrality")

    async def arefresh_centrality(self) -> dict:
        return await self._http.apost(f"{self._ns}/graph/centrality")


def _opt(body: Dict, **kwargs: Any) -> None:
    for k, v in kwargs.items():
        if v is not None:
            body[k] = v
