from typing import Dict, Optional

from ._http import Http


class AdminClient:
    def __init__(self, http: Http):
        self._http = http

    def backup(self) -> dict:
        return self._http.post("/v1/admin/backup")

    async def abackup(self) -> dict:
        return await self._http.apost("/v1/admin/backup")

    def audit(self, *, from_ms: Optional[int] = None, to_ms: Optional[int] = None,
              key: Optional[str] = None, limit: int = 100) -> list:
        return self._http.get("/v1/admin/audit", params=_audit_params(from_ms, to_ms, key, limit))

    async def aaudit(self, *, from_ms: Optional[int] = None, to_ms: Optional[int] = None,
                     key: Optional[str] = None, limit: int = 100) -> list:
        return await self._http.aget("/v1/admin/audit", params=_audit_params(from_ms, to_ms, key, limit))


def _audit_params(from_ms: Optional[int], to_ms: Optional[int],
                  key: Optional[str], limit: int) -> Dict:
    params: Dict = {"limit": limit}
    if from_ms is not None:
        params["from_ms"] = from_ms
    if to_ms is not None:
        params["to_ms"] = to_ms
    if key:
        params["key"] = key
    return params
