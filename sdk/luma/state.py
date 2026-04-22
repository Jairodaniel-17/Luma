from typing import Any, Dict, List, Optional

from ._http import Http


class StateClient:
    """KV store with JSON values, CAS optimistic locking, TTL, and secondary indexes."""

    def __init__(self, http: Http):
        self._http = http

    def list(self, prefix: Optional[str] = None, limit: int = 100) -> list:
        params: Dict = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        return self._http.get("/v1/state", params=params)

    async def alist(self, prefix: Optional[str] = None, limit: int = 100) -> list:
        params: Dict = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        return await self._http.aget("/v1/state", params=params)

    def get(self, key: str) -> dict:
        return self._http.get(f"/v1/state/{key}")

    async def aget(self, key: str) -> dict:
        return await self._http.aget(f"/v1/state/{key}")

    def put(self, key: str, value: Any, ttl_ms: Optional[int] = None,
            if_revision: Optional[int] = None) -> dict:
        """Set a key. Pass if_revision to enable CAS — raises LumaConflictError on mismatch."""
        return self._http.put(f"/v1/state/{key}", _put_body(value, ttl_ms, if_revision))

    async def aput(self, key: str, value: Any, ttl_ms: Optional[int] = None,
                   if_revision: Optional[int] = None) -> dict:
        return await self._http.aput(f"/v1/state/{key}", _put_body(value, ttl_ms, if_revision))

    def delete(self, key: str) -> dict:
        return self._http.delete(f"/v1/state/{key}")

    async def adelete(self, key: str) -> dict:
        return await self._http.adelete(f"/v1/state/{key}")

    def batch_put(self, operations: List[Dict]) -> dict:
        return self._http.post("/v1/state/batch_put", {"operations": operations})

    async def abatch_put(self, operations: List[Dict]) -> dict:
        return await self._http.apost("/v1/state/batch_put", {"operations": operations})

    def create_index(self, field: str) -> dict:
        """Register an in-memory secondary index on a value JSON field. Not persisted — recreate after restart."""
        return self._http.post("/v1/state/indexes", {"field": field})

    async def acreate_index(self, field: str) -> dict:
        return await self._http.apost("/v1/state/indexes", {"field": field})

    def query_index(self, field: str, value: str, limit: int = 100) -> list:
        """O(1) lookup by indexed field value. Requires a prior create_index call for this field."""
        return self._http.get(f"/v1/state/index/{field}/{value}", params={"limit": limit})

    async def aquery_index(self, field: str, value: str, limit: int = 100) -> list:
        return await self._http.aget(f"/v1/state/index/{field}/{value}", params={"limit": limit})


def _put_body(value: Any, ttl_ms: Optional[int], if_revision: Optional[int]) -> Dict:
    body: Dict = {"value": value}
    if ttl_ms is not None:
        body["ttl_ms"] = ttl_ms
    if if_revision is not None:
        body["if_revision"] = if_revision
    return body
