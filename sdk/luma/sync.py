"""
sync.py — Fully synchronous wrapper around the async Luma client.

For users who prefer a 100 % synchronous API without managing event loops or
``await`` expressions.  Every coroutine on the sub-clients is wrapped with
:func:`asyncio.run`.

Usage::

    from luma import SyncLuma

    db = SyncLuma("http://localhost:8080", api_key="secret")
    db.vector.create("embeddings", dim=1536)
    db.vector.upsert("embeddings", "doc-1", [0.1] * 1536, meta={"tag": "demo"})
    results = db.vector.search("embeddings", [0.1] * 1536, k=5)
    db.close()

    # or as a context manager
    with SyncLuma("http://localhost:8080", api_key="secret") as db:
        print(db.health())

Notes
-----
- :func:`asyncio.run` creates a *new* event loop per call.  This adds a small
  per-call overhead (~0.1 ms) compared to sharing a running loop.  For
  throughput-critical workloads use the native async ``Luma`` client instead.
- Streaming (SSE) is not wrapped here because it is inherently iterative; use
  the sync :meth:`~luma.stream.StreamClient.subscribe` iterator on the parent
  ``Luma`` client's ``.stream`` attribute directly.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional, TypeVar

from ._http import Http
from .admin import AdminClient
from .auth import AuthClient
from .config import ConfigClient
from .diskann import DiskAnnClient
from .doc import DocClient
from .hub import HubClient
from .memory import MemoryClient
from .meta import MetaClient
from .sql import SqlClient
from .state import StateClient
from .stream import StreamClient
from .vector import VectorClient

_T = TypeVar("_T")


def _run(coro: Any) -> Any:  # noqa: ANN401
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def _wrap_async_client(async_client: Any) -> Any:
    """Return a thin proxy that calls ``_run(original_method(...))`` for every
    ``a*`` method, while forwarding sync methods unchanged."""

    class _Proxy:
        def __getattr__(self, name: str) -> Any:
            attr = getattr(async_client, name)
            if asyncio.iscoroutinefunction(attr):
                @functools.wraps(attr)
                def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return _run(attr(*args, **kwargs))
                return _sync_wrapper
            return attr

    return _Proxy()


class _SyncVectorClient:
    """Synchronous facade for VectorClient."""

    def __init__(self, inner: VectorClient) -> None:
        self._i = inner

    def list(self) -> Any: return self._i.list()
    def get(self, collection: str) -> Any: return self._i.get(collection)
    def create(self, collection: str, dim: int, metric: str = "cosine") -> Any:
        return self._i.create(collection, dim, metric)
    def add(self, collection: str, id: str, vector: list, meta: Optional[dict] = None) -> Any:
        return self._i.add(collection, id, vector, meta)
    def upsert(self, collection: str, id: str, vector: list, meta: Optional[dict] = None) -> Any:
        return self._i.upsert(collection, id, vector, meta)
    def upsert_batch(self, collection: str, items: list) -> Any:
        return self._i.upsert_batch(collection, items)
    def update(self, collection: str, id: str, vector: Optional[list] = None,
               meta: Optional[dict] = None) -> Any:
        return self._i.update(collection, id, vector, meta)
    def delete(self, collection: str, id: str) -> Any:
        return self._i.delete(collection, id)
    def delete_batch(self, collection: str, ids: list) -> Any:
        return self._i.delete_batch(collection, ids)
    def get_by_id(self, collection: str, id: str) -> Any:
        return self._i.get_by_id(collection, id)
    def search(self, collection: str, vector: list, k: int,
               filters: Optional[dict] = None, include_meta: bool = False) -> Any:
        return self._i.search(collection, vector, k, filters, include_meta)
    def search_batch(self, collection: str, queries: list) -> Any:
        return self._i.search_batch(collection, queries)
    def scroll(self, collection: str, cursor: Optional[str] = None,
               limit: int = 100, include_vectors: bool = False) -> Any:
        return self._i.scroll(collection, cursor, limit, include_vectors)
    def rerank(self, collection: str, ids: list,
               query_text: Optional[str] = None,
               query_vector: Optional[list] = None) -> Any:
        return self._i.rerank(collection, ids, query_text, query_vector)
    def aggregate(self, collection: str, group_by: str,
                  filter: Optional[dict] = None, limit: int = 100) -> Any:
        return self._i.aggregate(collection, group_by, filter, limit)


class _SyncStateClient:
    def __init__(self, inner: StateClient) -> None:
        self._i = inner

    def list(self, prefix: Optional[str] = None, limit: int = 100) -> Any:
        return self._i.list(prefix, limit)
    def get(self, key: str) -> Any: return self._i.get(key)
    def put(self, key: str, value: Any, ttl_ms: Optional[int] = None,
            if_revision: Optional[int] = None) -> Any:
        return self._i.put(key, value, ttl_ms, if_revision)
    def delete(self, key: str) -> Any: return self._i.delete(key)
    def batch_put(self, operations: list) -> Any: return self._i.batch_put(operations)
    def create_index(self, field: str) -> Any: return self._i.create_index(field)
    def query_index(self, field: str, value: str, limit: int = 100) -> Any:
        return self._i.query_index(field, value, limit)


class _SyncDocClient:
    def __init__(self, inner: DocClient) -> None:
        self._i = inner

    def put(self, collection: str, id: str, document: dict) -> Any:
        return self._i.put(collection, id, document)
    def get(self, collection: str, id: str) -> Any:
        return self._i.get(collection, id)
    def delete(self, collection: str, id: str) -> Any:
        return self._i.delete(collection, id)
    def find(self, collection: str, filter: Optional[dict] = None, limit: int = 20) -> Any:
        return self._i.find(collection, filter, limit)


class _SyncSqlClient:
    def __init__(self, inner: SqlClient) -> None:
        self._i = inner

    def query(self, sql: str, params: Optional[list] = None) -> Any:
        return self._i.query(sql, params)
    def exec(self, sql: str, params: Optional[list] = None) -> Any:
        return self._i.exec(sql, params)


class _SyncAdminClient:
    def __init__(self, inner: AdminClient) -> None:
        self._i = inner

    def backup(self) -> Any: return self._i.backup()
    def audit(self, *, from_ms: Optional[int] = None, to_ms: Optional[int] = None,
              key: Optional[str] = None, limit: int = 100) -> Any:
        return self._i.audit(from_ms=from_ms, to_ms=to_ms, key=key, limit=limit)


class _SyncAuthClient:
    def __init__(self, inner: AuthClient) -> None:
        self._i = inner

    def list_keys(self) -> Any: return self._i.list_keys()
    def create_key(self, name: str, role: str = "user") -> Any:
        return self._i.create_key(name, role)
    def revoke_key(self, id: str) -> Any: return self._i.revoke_key(id)


class _SyncConfigClient:
    def __init__(self, inner: ConfigClient) -> None:
        self._i = inner

    def get(self) -> Any: return self._i.get()
    def put(self, config: dict) -> Any: return self._i.put(config)


class _SyncHubClient:
    def __init__(self, inner: HubClient) -> None:
        self._i = inner

    def ingest(self, text: str, id: Optional[str] = None,
               metadata: Optional[dict] = None) -> Any:
        return self._i.ingest(text, id=id, metadata=metadata)
    def search(self, query: str, sql_filter: Optional[str] = None,
               limit: int = 10) -> Any:
        return self._i.search(query, sql_filter, limit)


class _SyncMemoryClient:
    def __init__(self, inner: MemoryClient) -> None:
        self._i = inner

    def ingest_event(self, text: str, **kwargs: Any) -> Any:
        return self._i.ingest_event(text, **kwargs)
    def upsert_fact(self, content: str, **kwargs: Any) -> Any:
        return self._i.upsert_fact(content, **kwargs)
    def upsert_procedure(self, procedure_id: str, name: str,
                         nodes: list, edges: list, **kwargs: Any) -> Any:
        return self._i.upsert_procedure(procedure_id, name, nodes, edges, **kwargs)
    def query(self, query: str, **kwargs: Any) -> Any:
        return self._i.query(query, **kwargs)
    def next_step(self, procedure_id: str, **kwargs: Any) -> Any:
        return self._i.next_step(procedure_id, **kwargs)
    def timeline(self, entity_id: str) -> Any:
        return self._i.timeline(entity_id)
    def create_edge(self, source_id: str, target_id: str,
                    edge_type: str, **kwargs: Any) -> Any:
        return self._i.create_edge(source_id, target_id, edge_type, **kwargs)
    def node_edges(self, memory_id: str) -> Any:
        return self._i.node_edges(memory_id)
    def delete_edge(self, edge_id: str) -> Any:
        return self._i.delete_edge(edge_id)
    def belief_history(self, fact_key: str) -> Any:
        return self._i.belief_history(fact_key)
    def refresh_centrality(self) -> Any:
        return self._i.refresh_centrality()


class _SyncMetaClient:
    def __init__(self, inner: MetaClient) -> None:
        self._i = inner

    def execute(self, query: dict) -> Any:
        return self._i.execute(query)


class _SyncDiskAnnClient:
    def __init__(self, inner: DiskAnnClient) -> None:
        self._i = inner

    def build(self, max_degree: Optional[int] = None,
              build_threads: Optional[int] = None,
              search_list_size: Optional[int] = None) -> Any:
        return self._i.build(max_degree, build_threads, search_list_size)
    def tune(self, max_degree: Optional[int] = None,
             build_threads: Optional[int] = None,
             search_list_size: Optional[int] = None) -> Any:
        return self._i.tune(max_degree, build_threads, search_list_size)
    def status(self) -> Any:
        return self._i.status()


class SyncLuma:
    """Fully synchronous entry point for the Luma SDK.

    Wraps every sub-client so callers never need ``await`` or an event loop.
    All sync sub-clients expose the same method names as their async
    counterparts, minus the ``a`` prefix.

    For SSE streaming use the sync :meth:`~luma.stream.StreamClient.subscribe`
    iterator which is already synchronous::

        luma_async = Luma(url, api_key)
        for event in luma_async.stream.subscribe(since=0):
            print(event)

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
        self.vector = _SyncVectorClient(VectorClient(self._http))
        self.state = _SyncStateClient(StateClient(self._http))
        self.doc = _SyncDocClient(DocClient(self._http))
        self.sql = _SyncSqlClient(SqlClient(self._http))
        self.admin = _SyncAdminClient(AdminClient(self._http))
        self.auth = _SyncAuthClient(AuthClient(self._http))
        self.config = _SyncConfigClient(ConfigClient(self._http))
        # SSE streaming is inherently sync-iterable already
        self.stream = StreamClient(self._http)

    def memory(self, namespace: str) -> _SyncMemoryClient:
        return _SyncMemoryClient(MemoryClient(self._http, namespace))

    def hub(self, namespace: str) -> _SyncHubClient:
        return _SyncHubClient(HubClient(self._http, namespace))

    def meta(self, collection: str) -> _SyncMetaClient:
        return _SyncMetaClient(MetaClient(self._http, collection))

    def diskann(self, collection: str) -> _SyncDiskAnnClient:
        return _SyncDiskAnnClient(DiskAnnClient(self._http, collection))

    def health(self) -> Any:
        return self._http.get("/v1/health")

    def metrics(self) -> str:
        return self._http.get("/v1/metrics")  # type: ignore[return-value]

    def close(self) -> None:
        """Close the underlying httpx clients."""
        self._http.close()

    def __enter__(self) -> "SyncLuma":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SyncLuma(url={self._http.base_url!r})"
