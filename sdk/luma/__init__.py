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
from .vector import VectorClient

__version__ = "3.0.0"
__all__ = [
    "Luma",
    "LumaError",
    "LumaAuthError",
    "LumaForbiddenError",
    "LumaNotFoundError",
    "LumaConflictError",
]


class Luma:
    """Entry point for the Luma SDK. All sub-clients are accessible as attributes or factory methods."""

    def __init__(self, url: str, api_key: str, timeout: int = 30):
        self._http = Http(url, api_key, timeout)
        self.vector = VectorClient(self._http)
        self.state = StateClient(self._http)
        self.doc = DocClient(self._http)
        self.sql = SqlClient(self._http)
        self.admin = AdminClient(self._http)
        self.auth = AuthClient(self._http)
        self.stream = StreamClient(self._http)
        self.config = ConfigClient(self._http)

    def memory(self, namespace: str) -> MemoryClient:
        return MemoryClient(self._http, namespace)

    def hub(self, namespace: str) -> HubClient:
        return HubClient(self._http, namespace)

    def meta(self, collection: str) -> MetaClient:
        return MetaClient(self._http, collection)

    def diskann(self, collection: str) -> DiskAnnClient:
        return DiskAnnClient(self._http, collection)

    def health(self):
        return self._http.get("/v1/health")

    def metrics(self) -> str:
        return self._http.get("/v1/metrics")

    def __repr__(self) -> str:
        return f"Luma(url={self._http.base_url!r})"
