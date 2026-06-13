"""
luma — Python SDK for the Luma convergent data engine.

Quick start (async)::

    from luma import Luma

    luma = Luma("http://localhost:8080", api_key="secret")
    await luma.vector.acreate("embeddings", dim=1536)
    await luma.vector.aupsert("embeddings", "doc-1", [0.1] * 1536)
    results = await luma.vector.asearch("embeddings", [0.1] * 1536, k=5)

Quick start (sync)::

    from luma import SyncLuma

    with SyncLuma("http://localhost:8080", api_key="secret") as db:
        db.vector.create("embeddings", dim=1536)
        results = db.vector.search("embeddings", [0.1] * 1536, k=5)

Sub-clients
-----------
luma.vector   — VectorClient  (collections, CRUD, search, batch, scroll, rerank, aggregate)
luma.state    — StateClient   (KV store, TTL, CAS, indexes)
luma.doc      — DocClient     (JSON document store)
luma.sql      — SqlClient     (raw SQL via embedded SQLite)
luma.admin    — AdminClient   (backup, audit log)
luma.auth     — AuthClient    (API key management)
luma.stream   — StreamClient  (SSE event stream)
luma.config   — ConfigClient  (runtime configuration)
luma.hub(ns)  — HubClient     (LumaDatabase: text ingestion + hybrid search)
luma.memory(ns) — MemoryClient (NS-Mem: episodic/semantic/procedural/working memory)
luma.meta(c)  — MetaClient    (collection metadata queries)
luma.diskann(c) — DiskAnnClient (DiskANN graph build/tune/status)
"""
from __future__ import annotations

from ._http import Http
from .admin import AdminClient
from .auth import AuthClient
from .config import ConfigClient
from .diskann import DiskAnnClient
from .doc import DocClient
from .exceptions import (
    LumaAuthError,
    LumaConflictError,
    LumaError,
    LumaForbiddenError,
    LumaNotFoundError,
)
from .hub import HubClient
from .memory import MemoryClient
from .meta import MetaClient
from .sql import SqlClient
from .state import StateClient
from .stream import StreamClient
from .sync import SyncLuma
from .vector import VectorClient
from . import types

__version__ = "3.0.0"

__all__ = [
    # primary entry points
    "Luma",
    "SyncLuma",
    # exceptions
    "LumaError",
    "LumaAuthError",
    "LumaForbiddenError",
    "LumaNotFoundError",
    "LumaConflictError",
    # sub-clients (re-exported for isinstance checks and type annotations)
    "VectorClient",
    "StateClient",
    "DocClient",
    "SqlClient",
    "HubClient",
    "MemoryClient",
    "MetaClient",
    "DiskAnnClient",
    "AdminClient",
    "AuthClient",
    "StreamClient",
    "ConfigClient",
    # typed shapes
    "types",
    # transport
    "Http",
]


class Luma:
    """Async entry point for the Luma SDK.

    All sub-clients are accessible as attributes (fixed subsystems) or factory
    methods (namespace-scoped subsystems).

    Parameters
    ----------
    url:
        Base URL of the Luma server, e.g. ``"http://localhost:8080"``.
    api_key:
        Bearer token for authentication.
    timeout:
        Per-request timeout in seconds (default 30).
    """

    def __init__(self, url: str, api_key: str, timeout: int = 30) -> None:
        self._http = Http(url, api_key, timeout)
        self.vector = VectorClient(self._http)
        self.state = StateClient(self._http)
        self.doc = DocClient(self._http)
        self.sql = SqlClient(self._http)
        self.admin = AdminClient(self._http)
        self.auth = AuthClient(self._http)
        self.stream = StreamClient(self._http)
        self.config = ConfigClient(self._http)

    # ── namespace-scoped factories ────────────────────────────────────────────

    def memory(self, namespace: str) -> MemoryClient:
        """Return a MemoryClient scoped to *namespace*."""
        return MemoryClient(self._http, namespace)

    def hub(self, namespace: str) -> HubClient:
        """Return a HubClient (LumaDatabase) scoped to *namespace*."""
        return HubClient(self._http, namespace)

    def meta(self, collection: str) -> MetaClient:
        """Return a MetaClient for *collection*."""
        return MetaClient(self._http, collection)

    def diskann(self, collection: str) -> DiskAnnClient:
        """Return a DiskAnnClient for *collection*."""
        return DiskAnnClient(self._http, collection)

    # ── top-level endpoints ───────────────────────────────────────────────────

    def health(self) -> dict:
        """GET /v1/health — synchronous health check."""
        return self._http.get("/v1/health")  # type: ignore[return-value]

    async def ahealth(self) -> dict:
        """GET /v1/health — async health check."""
        return await self._http.aget("/v1/health")  # type: ignore[return-value]

    def metrics(self) -> str:
        """GET /v1/metrics — Prometheus-format metrics (sync)."""
        return self._http.get("/v1/metrics")  # type: ignore[return-value]

    async def ametrics(self) -> str:
        """GET /v1/metrics — Prometheus-format metrics (async)."""
        return await self._http.aget("/v1/metrics")  # type: ignore[return-value]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying async httpx client."""
        await self._http.aclose()

    def close(self) -> None:
        """Close the underlying sync httpx client."""
        self._http.close()

    async def __aenter__(self) -> "Luma":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"Luma(url={self._http.base_url!r})"
