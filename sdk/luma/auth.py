from ._http import Http


class AuthClient:
    def __init__(self, http: Http):
        self._http = http

    def list_keys(self) -> list:
        return self._http.get("/v1/auth/keys")

    async def alist_keys(self) -> list:
        return await self._http.aget("/v1/auth/keys")

    def create_key(self, name: str, role: str = "user") -> dict:
        return self._http.post("/v1/auth/keys", {"name": name, "role": role})

    async def acreate_key(self, name: str, role: str = "user") -> dict:
        return await self._http.apost("/v1/auth/keys", {"name": name, "role": role})

    def revoke_key(self, id: str) -> None:
        return self._http.delete(f"/v1/auth/keys/{id}")

    async def arevoke_key(self, id: str) -> None:
        return await self._http.adelete(f"/v1/auth/keys/{id}")
