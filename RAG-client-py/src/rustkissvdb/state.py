from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .client import Client


class StateAPI:
    def __init__(self, client: Client) -> None:
        self._client = client

    def list(self, prefix: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        return self._client.request(
            "GET",
            "/v1/state",
            params={"prefix": prefix, "limit": int(limit)},
        )

    def get(self, key: str) -> Dict[str, Any]:
        return self._client.request("GET", f"/v1/state/{key}")

    def put(
        self,
        key: str,
        value: Any,
        *,
        ttl_ms: Optional[int] = None,
        if_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"value": value}
        if ttl_ms is not None:
            payload["ttl_ms"] = int(ttl_ms)
        if if_revision is not None:
            payload["if_revision"] = int(if_revision)
        return self._client.request("PUT", f"/v1/state/{key}", json=payload)

    def delete(self, key: str) -> bool:
        data = self._client.request("DELETE", f"/v1/state/{key}")
        return bool(data.get("deleted"))

    def batch_put(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._client.request(
            "POST",
            "/v1/state/batch_put",
            json={"operations": operations},
        )
