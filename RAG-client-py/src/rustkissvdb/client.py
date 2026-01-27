from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from .doc import DocAPI
from .errors import RustKissVDBError
from .rag import RAGClient
from .sql import SqlAPI
from .state import StateAPI
from .stream import StreamAPI
from .vector import VectorAPI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Client:
    """
    Cliente principal para RustKissVDB.

    Integra operaciones de base de datos (Vector, State, SQL) y capacidades de RAG.

    Args:
        base_url: URL del servidor RustKissVDB (ej. http://localhost:9917).
        api_key: Clave de API para RustKissVDB.
        openai_key: (Opcional) Clave de API de OpenAI para RAG. Si no se da, busca en env vars.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        # Load defaults from environment if not provided
        self._base_url = (base_url or os.getenv("KISS_VDB_URL", "http://localhost:9917")).rstrip("/")
        self._api_key = api_key or os.getenv("KISS_VDB_KEY", "dev")

        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

        # APIs de Base de Datos (Low Level)
        self.state = StateAPI(self)
        self.vector = VectorAPI(self)
        self.doc = DocAPI(self)
        self.sql = SqlAPI(self)
        self.stream = StreamAPI(self)

        # API RAG (High Level)
        # Se inicializa de forma perezosa o directa.
        # Pasamos el vector_api para que el RAG client pueda hacer upserts/search.
        self.rag = RAGClient(vector_api=self.vector)
        if openai_key:
            self.rag.openai.api_key = openai_key

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
        try:
            resp = self._http.request(method, path, params=params, json=json)
        except httpx.RequestError as e:
            raise RustKissVDBError(f"Connection error: {e}")

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
