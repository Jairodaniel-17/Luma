import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from luma import Luma
from luma.exceptions import LumaNotFoundError


class LumaVectorStore(VectorStore):
    def __init__(self, url: str, api_key: str, collection: str,
                 embedding: Embeddings, dim: int, metric: str = "cosine"):
        self._client = Luma(url, api_key)
        self._collection = collection
        self._embedding = embedding
        self._dim = dim
        self._metric = metric
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.vector.get(self._collection)
        except LumaNotFoundError:
            self._client.vector.create(self._collection, dim=self._dim, metric=self._metric)

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ── relevance score ───────────────────────────────────────────────────────
    # Luma returns cosine in [-1, 1]; normalize to [0, 1] for relevance API.

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._metric == "cosine":
            return lambda s: (s + 1.0) / 2.0
        return lambda s: s

    # ── add_texts ─────────────────────────────────────────────────────────────

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        if not texts:
            return []
        vectors = self._embedding.embed_documents(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        items = [
            {"id": ids[i], "vector": vectors[i], "meta": {**metadatas[i], "_text": texts[i]}}
            for i in range(len(texts))
        ]
        self._client.vector.upsert_batch(self._collection, items)
        return ids

    async def aadd_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                         ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        if not texts:
            return []
        vectors = await self._embedding.aembed_documents(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        items = [
            {"id": ids[i], "vector": vectors[i], "meta": {**metadatas[i], "_text": texts[i]}}
            for i in range(len(texts))
        ]
        await self._client.vector.aupsert_batch(self._collection, items)
        return ids

    # ── get_by_ids ────────────────────────────────────────────────────────────

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        docs = []
        for id_ in ids:
            try:
                rec = self._client.vector.get_by_id(self._collection, id_)
                meta = dict(rec.get("meta") or {})
                text = meta.pop("_text", "")
                docs.append(Document(id=id_, page_content=text, metadata=meta))
            except LumaNotFoundError:
                pass
        return docs

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        docs = []
        for id_ in ids:
            try:
                rec = await self._client.vector.aget_by_id(self._collection, id_)
                meta = dict(rec.get("meta") or {})
                text = meta.pop("_text", "")
                docs.append(Document(id=id_, page_content=text, metadata=meta))
            except LumaNotFoundError:
                pass
        return docs

    # ── similarity_search ─────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4,
                          filter: Optional[Dict] = None, **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

    async def asimilarity_search(self, query: str, k: int = 4,
                                 filter: Optional[Dict] = None, **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in
                await self.asimilarity_search_with_score(query, k=k, filter=filter)]

    def similarity_search_with_score(self, query: str, k: int = 4,
                                     filter: Optional[Dict] = None,
                                     **kwargs: Any) -> List[Tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(vector, k=k, filter=filter)

    async def asimilarity_search_with_score(self, query: str, k: int = 4,
                                            filter: Optional[Dict] = None,
                                            **kwargs: Any) -> List[Tuple[Document, float]]:
        vector = await self._embedding.aembed_query(query)
        return await self.asimilarity_search_by_vector_with_score(vector, k=k, filter=filter)

    # ── similarity_search_by_vector ───────────────────────────────────────────

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4,
                                    filter: Optional[Dict] = None,
                                    **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in
                self.similarity_search_by_vector_with_score(embedding, k=k, filter=filter)]

    async def asimilarity_search_by_vector(self, embedding: List[float], k: int = 4,
                                           filter: Optional[Dict] = None,
                                           **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in
                await self.asimilarity_search_by_vector_with_score(embedding, k=k, filter=filter)]

    def similarity_search_by_vector_with_score(self, embedding: List[float], k: int = 4,
                                               filter: Optional[Dict] = None
                                               ) -> List[Tuple[Document, float]]:
        resp = self._client.vector.search(
            self._collection, vector=embedding, k=k, filters=filter, include_meta=True
        )
        return _hits_to_docs(resp.get("hits", []))

    async def asimilarity_search_by_vector_with_score(self, embedding: List[float], k: int = 4,
                                                      filter: Optional[Dict] = None
                                                      ) -> List[Tuple[Document, float]]:
        resp = await self._client.vector.asearch(
            self._collection, vector=embedding, k=k, filters=filter, include_meta=True
        )
        return _hits_to_docs(resp.get("hits", []))

    # ── MMR ───────────────────────────────────────────────────────────────────

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20,
                                      lambda_mult: float = 0.5,
                                      filter: Optional[Dict] = None,
                                      **kwargs: Any) -> List[Document]:
        vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            vector, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    async def amax_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20,
                                             lambda_mult: float = 0.5,
                                             filter: Optional[Dict] = None,
                                             **kwargs: Any) -> List[Document]:
        vector = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            vector, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    def max_marginal_relevance_search_by_vector(self, embedding: List[float], k: int = 4,
                                                fetch_k: int = 20, lambda_mult: float = 0.5,
                                                filter: Optional[Dict] = None,
                                                **kwargs: Any) -> List[Document]:
        resp = self._client.vector.search(
            self._collection, vector=embedding, k=fetch_k, filters=filter, include_meta=True
        )
        hits = resp.get("hits", [])
        return _mmr_from_hits(embedding, hits, k, lambda_mult, self._embedding)

    async def amax_marginal_relevance_search_by_vector(self, embedding: List[float], k: int = 4,
                                                       fetch_k: int = 20, lambda_mult: float = 0.5,
                                                       filter: Optional[Dict] = None,
                                                       **kwargs: Any) -> List[Document]:
        resp = await self._client.vector.asearch(
            self._collection, vector=embedding, k=fetch_k, filters=filter, include_meta=True
        )
        hits = resp.get("hits", [])
        candidate_embs = await self._embedding.aembed_documents(
            [h.get("meta", {}).get("_text", "") for h in hits]
        )
        indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            candidate_embs,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [_hit_to_doc(hits[i]) for i in indices]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        self._client.vector.delete_batch(self._collection, ids)
        return True

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        await self._client.vector.adelete_batch(self._collection, ids)
        return True

    # ── from_texts ────────────────────────────────────────────────────────────

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings,
                   metadatas: Optional[List[Dict]] = None,
                   ids: Optional[List[str]] = None,
                   **kwargs: Any) -> "LumaVectorStore":
        store = cls(
            url=kwargs["url"],
            api_key=kwargs["api_key"],
            collection=kwargs["collection"],
            embedding=embedding,
            dim=kwargs["dim"],
            metric=kwargs.get("metric", "cosine"),
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_texts(cls, texts: List[str], embedding: Embeddings,
                          metadatas: Optional[List[Dict]] = None,
                          ids: Optional[List[str]] = None,
                          **kwargs: Any) -> "LumaVectorStore":
        store = cls(
            url=kwargs["url"],
            api_key=kwargs["api_key"],
            collection=kwargs["collection"],
            embedding=embedding,
            dim=kwargs["dim"],
            metric=kwargs.get("metric", "cosine"),
        )
        await store.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def __repr__(self) -> str:
        return f"LumaVectorStore(collection={self._collection!r})"


# ── helpers ───────────────────────────────────────────────────────────────────


def _hit_to_doc(hit: dict) -> Document:
    meta = dict(hit.get("meta") or {})
    text = meta.pop("_text", "")
    return Document(page_content=text, metadata=meta)


def _hits_to_docs(hits: list) -> List[Tuple[Document, float]]:
    return [(_hit_to_doc(h), h["score"]) for h in hits]


def _mmr_from_hits(query_emb: List[float], hits: list, k: int,
                   lambda_mult: float, embedding: Embeddings) -> List[Document]:
    if not hits:
        return []
    texts = [h.get("meta", {}).get("_text", "") for h in hits]
    candidate_embs = embedding.embed_documents(texts)
    indices = maximal_marginal_relevance(
        np.array(query_emb, dtype=np.float32),
        candidate_embs,
        lambda_mult=lambda_mult,
        k=k,
    )
    return [_hit_to_doc(hits[i]) for i in indices]
