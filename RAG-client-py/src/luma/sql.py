from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .client import Client


class SqlAPI:
    def __init__(self, client: Client) -> None:
        self._client = client

    def query(self, sql: str, params: List[Any] | None = None) -> List[Dict[str, Any]]:
        data = self._client.request(
            "POST",
            "/v1/sql/query",
            json={"sql": sql, "params": params or []},
        )
        return data.get("rows", [])

    def execute(self, sql: str, params: List[Any] | None = None) -> int:
        data = self._client.request(
            "POST",
            "/v1/sql/exec",
            json={"sql": sql, "params": params or []},
        )
        return int(data.get("rows_affected", 0))
