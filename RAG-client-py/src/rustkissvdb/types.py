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
    # For high-level search (text content)
    content: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[SearchResult]


# --- RAG Models ---


class RAGGeneration(BaseModel):
    answer: str
    sources: List[SearchResult]
    usage: Dict[str, Any]


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}
