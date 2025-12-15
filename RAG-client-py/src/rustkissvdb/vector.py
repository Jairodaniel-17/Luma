from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .client import Client


class VectorAPI:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create_collection(self, collection: str, dim: int, metric: str = "cosine") -> Dict[str, Any]:
        return self._client.request(
            "POST",
            f"/v1/vector/{collection}",
            json={"dim": int(dim), "metric": metric},
        )

    def upsert(
        self,
        collection: str,
        *,
        vector_id: str,
        vector: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"id": vector_id, "vector": vector}
        if meta is not None:
            payload["meta"] = meta
        return self._client.request(
            "POST",
            f"/v1/vector/{collection}/upsert",
            json=payload,
        )

    def upsert_batch(
        self,
        collection: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._client.request(
            "POST",
            f"/v1/vector/{collection}/upsert_batch",
            json={"items": items},
        )

    def delete(self, collection: str, vector_id: str) -> bool:
        data = self._client.request(
            "POST",
            f"/v1/vector/{collection}/delete",
            json={"id": vector_id},
        )
        return bool(data.get("deleted"))

    def delete_batch(self, collection: str, ids: List[str]) -> Dict[str, Any]:
        return self._client.request(
            "POST",
            f"/v1/vector/{collection}/delete_batch",
            json={"ids": ids},
        )

    def search(
        self,
        collection: str,
        vector: List[float],
        *,
        k: int = 5,
        include_meta: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "vector": vector,
            "k": int(k),
            "include_meta": bool(include_meta),
        }
        if filters is not None:
            payload["filters"] = filters
        return self._client.request(
            "POST",
            f"/v1/vector/{collection}/search",
            json=payload,
        )
