from typing import Dict, Optional

from ._http import Http


class HubClient:
    def __init__(self, http: Http, namespace: str):
        self._http = http
        self._base = f"/v1/db/{namespace}"

    def ingest(self, text: str, id: Optional[str] = None,
               metadata: Optional[Dict] = None) -> dict:
        body: Dict = {"text": text}
        if id:
            body["id"] = id
        if metadata:
            body["metadata"] = metadata
        return self._http.post(f"{self._base}/ingest", body)

    async def aingest(self, text: str, id: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> dict:
        body: Dict = {"text": text}
        if id:
            body["id"] = id
        if metadata:
            body["metadata"] = metadata
        return await self._http.apost(f"{self._base}/ingest", body)

    def search(self, query: str, sql_filter: Optional[str] = None, limit: int = 10) -> dict:
        body: Dict = {"query": query, "limit": limit}
        if sql_filter:
            body["sql_filter"] = sql_filter
        return self._http.post(f"{self._base}/search", body)

    async def asearch(self, query: str, sql_filter: Optional[str] = None, limit: int = 10) -> dict:
        body: Dict = {"query": query, "limit": limit}
        if sql_filter:
            body["sql_filter"] = sql_filter
        return await self._http.apost(f"{self._base}/search", body)
