from typing import Dict, Optional

from ._http import Http


class DocClient:
    """Raw JSON document store. Documents are independent of vectors but share the collection namespace."""

    def __init__(self, http: Http):
        self._http = http

    def put(self, collection: str, id: str, document: Dict) -> dict:
        return self._http.put(f"/v1/doc/{collection}/{id}", document)

    async def aput(self, collection: str, id: str, document: Dict) -> dict:
        return await self._http.aput(f"/v1/doc/{collection}/{id}", document)

    def get(self, collection: str, id: str) -> dict:
        return self._http.get(f"/v1/doc/{collection}/{id}")

    async def aget(self, collection: str, id: str) -> dict:
        return await self._http.aget(f"/v1/doc/{collection}/{id}")

    def delete(self, collection: str, id: str) -> None:
        return self._http.delete(f"/v1/doc/{collection}/{id}")

    async def adelete(self, collection: str, id: str) -> None:
        return await self._http.adelete(f"/v1/doc/{collection}/{id}")

    def find(self, collection: str, filter: Optional[Dict] = None, limit: int = 20) -> dict:
        return self._http.post(f"/v1/doc/{collection}/find", {"filter": filter, "limit": limit})

    async def afind(self, collection: str, filter: Optional[Dict] = None, limit: int = 20) -> dict:
        return await self._http.apost(f"/v1/doc/{collection}/find", {"filter": filter, "limit": limit})
