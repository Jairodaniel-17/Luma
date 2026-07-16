"""
types.py — TypedDict definitions for all major Luma request/response shapes.

Import these for type-checked usage with mypy / pyright:

    from luma.types import VectorItem, SearchResult, MemoryRecord
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import NotRequired, TypedDict


# ── Vector ────────────────────────────────────────────────────────────────────


class VectorItem(TypedDict):
    """A single vector entry as returned by get_by_id / scroll."""

    id: str
    vector: NotRequired[List[float]]
    meta: NotRequired[Dict[str, Any]]


class SearchHit(TypedDict):
    """One result inside a search or rerank response."""

    id: str
    score: float
    meta: NotRequired[Dict[str, Any]]


class SearchResult(TypedDict):
    """Top-level response from POST /v1/vector/{collection}/search."""

    hits: List[SearchHit]


class BatchSearchResult(TypedDict):
    """Top-level response from POST /v1/vector/{collection}/search_batch."""

    results: List[SearchResult]


class ScrollResult(TypedDict):
    """Response from GET /v1/vector/{collection}/scroll."""

    items: List[VectorItem]
    next_cursor: NotRequired[Optional[str]]


class AggregateResult(TypedDict):
    """One bucket from POST /v1/vector/{collection}/aggregate."""

    value: str
    count: int


class CollectionInfo(TypedDict):
    """Metadata returned by GET /v1/vector/{collection}."""

    name: str
    dim: int
    metric: str
    count: NotRequired[int]


# ── State / KV ────────────────────────────────────────────────────────────────


class StateEntry(TypedDict):
    """A single KV record."""

    key: str
    value: Any
    revision: int
    ttl_ms: NotRequired[Optional[int]]


# ── Document ──────────────────────────────────────────────────────────────────


class DocEntry(TypedDict):
    """Raw JSON document record."""

    id: str
    collection: str
    document: Dict[str, Any]


# ── Hub ───────────────────────────────────────────────────────────────────────


class IngestResult(TypedDict):
    """Response from POST /v1/db/{namespace}/ingest."""

    id: str
    chunks: int


class HubSearchHit(TypedDict):
    """One hit from POST /v1/db/{namespace}/search."""

    id: str
    score: float
    text: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]


class HubSearchResult(TypedDict):
    """Top-level response from POST /v1/db/{namespace}/search."""

    hits: List[HubSearchHit]


# ── Memory / NS-Mem ───────────────────────────────────────────────────────────


class MemoryRecord(TypedDict):
    """A stored memory record (episodic or semantic)."""

    id: str
    namespace: str
    memory_type: str  # "episodic" | "semantic" | "working"
    content: NotRequired[str]
    text: NotRequired[str]
    entity_id: NotRequired[Optional[str]]
    fact_key: NotRequired[Optional[str]]
    source: NotRequired[Optional[str]]
    session_id: NotRequired[Optional[str]]
    confidence: NotRequired[float]
    status: NotRequired[str]  # "active" | "draft" | "archived"
    decay_score: NotRequired[float]
    centrality_score: NotRequired[float]
    metadata: NotRequired[Optional[Dict[str, Any]]]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]


class MemoryEdge(TypedDict):
    """A typed weighted edge in the memory graph."""

    id: str
    source_id: str
    target_id: str
    edge_type: str  # "triggered_by" | "supports" | "contradicts" | "supersedes" | "related_to"
    weight: NotRequired[float]
    metadata: NotRequired[Optional[Dict[str, Any]]]
    created_at: NotRequired[str]


class MemoryQueryResult(TypedDict):
    """Response from POST /v1/memory/{namespace}/query."""

    results: List[MemoryRecord]
    diagnostics: NotRequired[Dict[str, Any]]


class ProcedureNode(TypedDict):
    """A node in a procedural DAG."""

    id: str
    label: str
    description: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]


class ProcedureEdge(TypedDict):
    """An edge in a procedural DAG."""

    from_id: str
    to_id: str
    edge_type: str
    condition: NotRequired[Optional[str]]


class ProcedureConstraint(TypedDict):
    """A constraint attached to a procedure node."""

    node_id: str
    expression: str
    description: NotRequired[str]


class NextStepResult(TypedDict):
    """Response from POST /v1/memory/{namespace}/next_step."""

    node_id: str
    label: str
    description: NotRequired[str]


# ── Auth ──────────────────────────────────────────────────────────────────────


class ApiKey(TypedDict):
    """An API key record."""

    id: str
    name: str
    role: str  # "admin" | "user"
    created_at: str
    revoked: NotRequired[bool]


# ── Admin ─────────────────────────────────────────────────────────────────────


class BackupResult(TypedDict):
    """Response from POST /v1/admin/backup."""

    ok: bool
    offset: int


class AuditEntry(TypedDict):
    """One row from GET /v1/admin/audit."""

    ts: int
    api_key_id: NotRequired[Optional[str]]
    ip: str
    method: str
    path: str
    status: int
    latency_ms: int


# ── DiskANN ───────────────────────────────────────────────────────────────────


class DiskAnnStatus(TypedDict):
    """Response from GET /v1/vector/{collection}/diskann/status."""

    built: bool
    max_degree: NotRequired[int]
    node_count: NotRequired[int]


# ── SSE Events ────────────────────────────────────────────────────────────────


class SseEvent(TypedDict):
    """A parsed Server-Sent Event from the /v1/stream endpoint."""

    type: str
    offset: NotRequired[int]
    key: NotRequired[str]
    collection: NotRequired[str]
    payload: NotRequired[Any]
