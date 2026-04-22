from typing import Any, List, Optional

from ._http import Http


class SqlClient:
    """SQLite bridge. Use query for SELECT, exec for DDL/DML."""

    def __init__(self, http: Http):
        self._http = http

    def query(self, sql: str, params: Optional[List[Any]] = None) -> dict:
        return self._http.post("/v1/sql/query", {"sql": sql, "params": params or []})

    async def aquery(self, sql: str, params: Optional[List[Any]] = None) -> dict:
        return await self._http.apost("/v1/sql/query", {"sql": sql, "params": params or []})

    def exec(self, sql: str, params: Optional[List[Any]] = None) -> dict:
        return self._http.post("/v1/sql/exec", {"sql": sql, "params": params or []})

    async def aexec(self, sql: str, params: Optional[List[Any]] = None) -> dict:
        return await self._http.apost("/v1/sql/exec", {"sql": sql, "params": params or []})
