from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Distance(str, Enum):
    COSINE = "cosine"
    EUCLID = "l2"
    DOT = "dot"


class CollectionConfig(BaseModel):
    dim: int
    metric: Distance = Field(default=Distance.COSINE)


class VectorRecord(BaseModel):
    id: Union[str, int]
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="meta")

    class Config:
        populate_by_name = True


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 5
    include_metadata: bool = True
    filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None


class UpdateStatus(str, Enum):
    COMPLETED = "completed"
    ACKNOWLEDGED = "acknowledged"


class OperationInfo(BaseModel):
    operation_id: int = 0
    status: UpdateStatus = UpdateStatus.COMPLETED
