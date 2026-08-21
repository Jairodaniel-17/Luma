"""
search.py — Text search engine (``/search``, ``/search/ingest``).

This is the standalone ``src/search`` engine, not the hybrid RAG hub. The
difference matters when choosing between them:

- **Here**, :meth:`SearchClient.ingest` takes a document with a
  **caller-supplied vector**. Nothing is embedded server-side and nothing is
  chunked.
- **The hub** (``luma.hub(ns).ingest``) takes text, chunks it, embeds it with
  the configured provider, and supports SQL pre-filtering.

Reach for the hub unless you are managing vectors yourself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._http import Http


class SearchClient:
    """Standalone text search engine."""

    def __init__(self, http: Http):
        self._http = http

    def ingest(
        self,
        id: int,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Index one document.

        ``id`` is a 32-bit integer here, unlike the string ids used elsewhere in
        the API — this engine keys its own index by numeric id.
        """
        return self._http.post("/search/ingest", _document(id, vector, content, metadata))

    async def aingest(
        self,
        id: int,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await self._http.apost(
            "/search/ingest", _document(id, vector, content, metadata)
        )

    def query(self, query: str, top_k: int = 10, **options: Any) -> Any:
        """Search by text. ``query`` is capped at 1024 characters server-side."""
        body: Dict[str, Any] = {"query": query, "top_k": top_k}
        body.update(options)
        return self._http.post("/search", body)

    async def aquery(self, query: str, top_k: int = 10, **options: Any) -> Any:
        body: Dict[str, Any] = {"query": query, "top_k": top_k}
        body.update(options)
        return await self._http.apost("/search", body)


def _document(
    id: int, vector: List[float], content: str, metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"id": id, "vector": vector, "content": content}
    if metadata is not None:
        doc["metadata"] = metadata
    return {"document": doc}
