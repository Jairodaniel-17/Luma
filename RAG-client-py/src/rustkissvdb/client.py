from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .doc import DocAPI
from .errors import RustKissVDBError
from .sql import SqlAPI
from .state import StateAPI
from .stream import StreamAPI
from .vector import VectorAPI


class Client:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url requerido")

        self._base_url = base_url.rstrip("/")

        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

        # Sub-APIs (no import circular si ellos NO importan Client en runtime)
        self.state = StateAPI(self)
        self.vector = VectorAPI(self)
        self.doc = DocAPI(self)
        self.sql = SqlAPI(self)
        self.stream = StreamAPI(self)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        resp = self._http.request(method, path, params=params, json=json)

        if resp.status_code >= 400:
            raise RustKissVDBError(self._error_message(resp))

        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            return resp.json()
        return resp.text

    def stream_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        req = self._http.build_request(method, path, params=params)
        return self._http.send(req, stream=True)

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            err = data.get("error") or "error"
            msg = data.get("message") or resp.text
            return f"{err} - {msg}"
        except Exception:
            return resp.text
