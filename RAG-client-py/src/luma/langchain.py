from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, Dict

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "Could not import langchain-core. "
        "Please install it with `pip install langchain-core` or `pip install rustkissvdb[langchain]`."
    )

from .client import Client
from .models import VectorRecord, Distance, CollectionConfig

class LumaVectorStore(VectorStore):
    """
    Luma (powered by RustKissVDB) vector store integration for LangChain.
    """

    def __init__(
        self,
        client: Client,
        collection_name: str,
        embedding: Embeddings,
        content_payload_key: str = "page_content",
        metadata_payload_key: str = "metadata",
        distance: Distance = Distance.COSINE,
    ):
        self.client = client
        self.collection_name = collection_name
        self._embedding = embedding
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.distance = distance

        # Ensure collection exists (optional, but good for DX)
        try:
            self.client.create_collection(
                collection_name,
                CollectionConfig(dim=1536, metric=distance) # Dimension is unknown here without embedding check
            )
        except Exception:
            pass

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        vectors = self._embedding.embed_documents(texts)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        records = []
        for text, vector, meta, id_ in zip(texts, vectors, metadatas, ids):
            payload = meta.copy()
            payload[self.content_payload_key] = text
            
            records.append(VectorRecord(
                id=id_,
                vector=vector,
                metadata=payload
            ))
            
        self.client.upsert(self.collection_name, records)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        vector = self._embedding.embed_query(query)
        results = self.client.search(
            collection=self.collection_name,
            vector=vector,
            k=k,
            filter=filter,
            include_metadata=True
        )
        
        docs = []
        for res in results:
            payload = res.metadata or {}
            content = payload.pop(self.content_payload_key, "")
            docs.append(Document(page_content=content, metadata=payload))
            
        return docs

    @classmethod
    def from_texts(
        cls: Type[LumaVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[Client] = None,
        collection_name: str = "langchain",
        **kwargs: Any,
    ) -> LumaVectorStore:
        if client is None:
            client = Client(**kwargs.get("client_kwargs", {}))
            
        if texts:
            sample_embed = embedding.embed_query(texts[0])
            dim = len(sample_embed)
            try:
                client.create_collection(
                    collection_name, 
                    CollectionConfig(dim=dim, metric=kwargs.get("distance", Distance.COSINE))
                )
            except Exception:
                pass

        vectorstore = cls(client, collection_name, embedding, **kwargs)
        vectorstore.add_texts(texts, metadatas)
        return vectorstore
