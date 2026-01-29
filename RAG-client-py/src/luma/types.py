from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# --- Vector Models ---


class VectorMetric(str):
    COSINE = "cosine"
    DOT = "dot"


class CollectionInfo(BaseModel):
    collection: str
    dim: int
    metric: str
    live_count: int
    total_records: int
    upsert_count: int


class SearchResult(BaseModel):
    id: str
    score: float
    meta: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Alias para compatibilidad
    vector: Optional[List[float]] = None


class SearchResponse(BaseModel):
    hits: List[SearchResult]


# --- RAG Models ---


class RAGGeneration(BaseModel):
    answer: str
    sources: List[SearchResult]
    usage: Optional[Dict[str, Any]] = None


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}
