from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .client import Client



class DocAPI:
    def __init__(self, client: Client) -> None:
        self._client = client

    def put(self, collection: str, doc_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._client.request("PUT", f"/v1/doc/{collection}/{doc_id}", json=doc)

    def get(self, collection: str, doc_id: str) -> Dict[str, Any]:
        return self._client.request("GET", f"/v1/doc/{collection}/{doc_id}")

    def delete(self, collection: str, doc_id: str) -> bool:
        data = self._client.request("DELETE", f"/v1/doc/{collection}/{doc_id}")
        return bool(data.get("deleted"))

    def find(
        self,
        collection: str,
        *,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        data = self._client.request(
            "POST",
            f"/v1/doc/{collection}/find",
            json={"filter": filter or {}, "limit": int(limit)},
        )
        return data.get("documents", [])
