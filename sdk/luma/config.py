from ._http import Http


class ConfigClient:
    def __init__(self, http: Http):
        self._http = http

    def get(self) -> dict:
        return self._http.get("/v1/config")

    async def aget(self) -> dict:
        return await self._http.aget("/v1/config")

    def put(self, config: dict) -> dict:
        return self._http.put("/v1/config", config)

    async def aput(self, config: dict) -> dict:
        return await self._http.aput("/v1/config", config)
