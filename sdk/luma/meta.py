from typing import Dict

from ._http import Http


class MetaClient:
    def __init__(self, http: Http, collection: str):
        self._http = http
        self._collection = collection

    def execute(self, query: Dict) -> list:
        return self._http.post(f"/v1/meta/{self._collection}/execute", query)

    async def aexecute(self, query: Dict) -> list:
        return await self._http.apost(f"/v1/meta/{self._collection}/execute", query)
