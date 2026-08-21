from typing import Dict, List, Optional

from ._http import Http


class VectorClient:
    """Core vector operations — collections, CRUD, search, and analytics."""

    def __init__(self, http: Http):
        self._http = http

    # ── collections ──────────────────────────────────────────────────────────

    def list(self) -> dict:
        return self._http.get("/v1/vector")

    async def alist(self) -> dict:
        return await self._http.aget("/v1/vector")

    def get(self, collection: str) -> dict:
        return self._http.get(f"/v1/vector/{collection}")

    async def aget(self, collection: str) -> dict:
        return await self._http.aget(f"/v1/vector/{collection}")

    def create(self, collection: str, dim: int, metric: str = "cosine") -> dict:
        return self._http.post(f"/v1/vector/{collection}", {"dim": dim, "metric": metric})

    async def acreate(self, collection: str, dim: int, metric: str = "cosine") -> dict:
        return await self._http.apost(f"/v1/vector/{collection}", {"dim": dim, "metric": metric})

    def drop(self, collection: str) -> dict:
        """Delete an entire collection: in-memory index, on-disk data and the
        ownership row. Irreversible — there is no tombstone to restore from."""
        return self._http.delete(f"/v1/vector/{collection}")

    async def adrop(self, collection: str) -> dict:
        return await self._http.adelete(f"/v1/vector/{collection}")

    # ── write ────────────────────────────────────────────────────────────────

    def add(self, collection: str, id: str, vector: List[float],
            meta: Optional[Dict] = None) -> dict:
        return self._http.post(f"/v1/vector/{collection}/add", _vec_body(id, vector, meta))

    async def aadd(self, collection: str, id: str, vector: List[float],
                   meta: Optional[Dict] = None) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/add", _vec_body(id, vector, meta))

    def upsert(self, collection: str, id: str, vector: List[float],
               meta: Optional[Dict] = None) -> dict:
        return self._http.post(f"/v1/vector/{collection}/upsert", _vec_body(id, vector, meta))

    async def aupsert(self, collection: str, id: str, vector: List[float],
                      meta: Optional[Dict] = None) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/upsert", _vec_body(id, vector, meta))

    def upsert_batch(self, collection: str, items: List[Dict]) -> dict:
        return self._http.post(f"/v1/vector/{collection}/upsert_batch", {"items": items})

    async def aupsert_batch(self, collection: str, items: List[Dict]) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/upsert_batch", {"items": items})

    def update(self, collection: str, id: str, vector: Optional[List[float]] = None,
               meta: Optional[Dict] = None) -> dict:
        body: Dict = {"id": id}
        if vector is not None:
            body["vector"] = vector
        if meta is not None:
            body["meta"] = meta
        return self._http.post(f"/v1/vector/{collection}/update", body)

    async def aupdate(self, collection: str, id: str, vector: Optional[List[float]] = None,
                      meta: Optional[Dict] = None) -> dict:
        body: Dict = {"id": id}
        if vector is not None:
            body["vector"] = vector
        if meta is not None:
            body["meta"] = meta
        return await self._http.apost(f"/v1/vector/{collection}/update", body)

    def delete(self, collection: str, id: str) -> dict:
        return self._http.post(f"/v1/vector/{collection}/delete", {"id": id})

    async def adelete(self, collection: str, id: str) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/delete", {"id": id})

    def delete_batch(self, collection: str, ids: List[str]) -> dict:
        return self._http.post(f"/v1/vector/{collection}/delete_batch", {"ids": ids})

    async def adelete_batch(self, collection: str, ids: List[str]) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/delete_batch", {"ids": ids})

    # ── read ─────────────────────────────────────────────────────────────────

    def get_by_id(self, collection: str, id: str) -> dict:
        return self._http.get(f"/v1/vector/{collection}/get", params={"id": id})

    async def aget_by_id(self, collection: str, id: str) -> dict:
        return await self._http.aget(f"/v1/vector/{collection}/get", params={"id": id})

    def search(self, collection: str, vector: List[float], k: int,
               filters: Optional[Dict] = None, include_meta: bool = False) -> dict:
        return self._http.post(f"/v1/vector/{collection}/search", _search_body(vector, k, filters, include_meta))

    async def asearch(self, collection: str, vector: List[float], k: int,
                      filters: Optional[Dict] = None, include_meta: bool = False) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/search", _search_body(vector, k, filters, include_meta))

    def search_batch(self, collection: str, queries: List[Dict]) -> dict:
        """Execute up to 100 search queries in parallel via rayon. One result set per query."""
        return self._http.post(f"/v1/vector/{collection}/search_batch", {"queries": queries})

    async def asearch_batch(self, collection: str, queries: List[Dict]) -> dict:
        return await self._http.apost(f"/v1/vector/{collection}/search_batch", {"queries": queries})

    def scroll(self, collection: str, cursor: Optional[str] = None,
               limit: int = 100, include_vectors: bool = False) -> dict:
        """Cursor-based page scan. Pass `next_cursor` from the response to fetch the next page."""
        params: Dict = {"limit": limit, "include_vectors": include_vectors}
        if cursor:
            params["cursor"] = cursor
        return self._http.get(f"/v1/vector/{collection}/scroll", params=params)

    async def ascroll(self, collection: str, cursor: Optional[str] = None,
                      limit: int = 100, include_vectors: bool = False) -> dict:
        params: Dict = {"limit": limit, "include_vectors": include_vectors}
        if cursor:
            params["cursor"] = cursor
        return await self._http.aget(f"/v1/vector/{collection}/scroll", params=params)

    def rerank(self, collection: str, ids: List[str],
               query_text: Optional[str] = None,
               query_vector: Optional[List[float]] = None) -> dict:
        """Re-score stored vectors for the given IDs and return them sorted by cosine desc."""
        body: Dict = {"ids": ids}
        if query_text is not None:
            body["query_text"] = query_text
        if query_vector is not None:
            body["query_vector"] = query_vector
        return self._http.post(f"/v1/vector/{collection}/rerank", body)

    async def arerank(self, collection: str, ids: List[str],
                      query_text: Optional[str] = None,
                      query_vector: Optional[List[float]] = None) -> dict:
        body: Dict = {"ids": ids}
        if query_text is not None:
            body["query_text"] = query_text
        if query_vector is not None:
            body["query_vector"] = query_vector
        return await self._http.apost(f"/v1/vector/{collection}/rerank", body)

    def aggregate(self, collection: str, group_by: str,
                  filter: Optional[Dict] = None, limit: int = 100) -> dict:
        """Group vectors by a metadata field and count per bucket. Uses keyword index fast path."""
        body: Dict = {"group_by": group_by, "limit": limit}
        if filter:
            body["filter"] = filter
        return self._http.post(f"/v1/vector/{collection}/aggregate", body)

    async def aaggregate(self, collection: str, group_by: str,
                         filter: Optional[Dict] = None, limit: int = 100) -> dict:
        body: Dict = {"group_by": group_by, "limit": limit}
        if filter:
            body["filter"] = filter
        return await self._http.apost(f"/v1/vector/{collection}/aggregate", body)


def _vec_body(id: str, vector: List[float], meta: Optional[Dict]) -> Dict:
    body: Dict = {"id": id, "vector": vector}
    if meta is not None:
        body["meta"] = meta
    return body


def _search_body(vector: List[float], k: int,
                 filters: Optional[Dict], include_meta: bool) -> Dict:
    body: Dict = {"vector": vector, "k": k, "include_meta": include_meta}
    if filters:
        body["filters"] = filters
    return body
