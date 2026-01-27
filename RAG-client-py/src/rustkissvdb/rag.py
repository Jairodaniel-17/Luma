import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from .types import RAGGeneration, SearchResult


class RAGClient:
    """
    High-level client for RAG operations.
    Handles embedding generation, context retrieval, and LLM answer synthesis.
    """

    def __init__(
        self,
        vector_api,  # Type: VectorAPI
        openai_client: Optional[OpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o",
    ):
        self.vector_api = vector_api

        # Initialize OpenAI client (tries env vars if not provided)
        self.openai = openai_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL")
        self.llm_model = llm_model or os.getenv("LLM_MODEL")

        self._cached_dim: Optional[int] = None

    def embed(self, text: Union[str, List[str]]) -> List[float]:
        """Generates embeddings for a single text or list of texts."""
        if isinstance(text, str):
            text = [text]

        # Clean text
        text = [t.replace("\n", " ") for t in text]

        response = self.openai.embeddings.create(input=text, model=self.embedding_model)

        # Return single vector if input was single string
        if len(text) == 1:
            return response.data[0].embedding
        else:
            return [item.embedding for item in response.data]

    def get_model_dimension(self) -> int:
        """
        Determines the output dimension of the configured embedding model
        by generating a dummy embedding.
        """
        if self._cached_dim is not None:
            return self._cached_dim

        dummy_vector = self.embed("test_dimension_probe")
        self._cached_dim = len(dummy_vector)
        return self._cached_dim

    def initialize_collection(self, collection: str, metric: str = "cosine") -> int:
        """
        Ensures a collection exists with the correct dimension for the current model.
        If the collection doesn't exist, it detects the dimension and creates it.
        Returns the dimension of the collection.
        """
        # 1. Check if collection exists
        try:
            info = self.vector_api.get_collection(collection)
            if info and "dim" in info:
                return info["dim"]
        except Exception:
            # Assume it doesn't exist (404), proceed to create
            pass

        # 2. Detect dimension
        dim = self.get_model_dimension()

        # 3. Create collection (handle race condition or existence)
        try:
            self.vector_api.create_collection(collection, dim=dim, metric=metric)
        except Exception as e:
            msg = str(e).lower()
            if "already_exists" in msg or "409" in msg:
                # It exists now, so we are good.
                return dim
            raise e
            
        return dim

    def ingest_text(
        self, 
        collection: str, 
        text: str, 
        metadata: Dict[str, Any] = {}, 
        chunk_size: int = 1000,
        id_prefix: str = "doc"
    ) -> int:
        """
        Chunks text, generates embeddings, and upserts to the database.
        Returns the number of chunks ingested.
        """
        metadata = metadata or {}

        # 1. Naive chunking (can be improved)
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        items = []
        for idx, chunk in enumerate(chunks):
            # Generate ID
            doc_id = f"{id_prefix}_{idx}"

            # Embed
            vector = self.embed(chunk)

            # Prepare metadata
            meta = metadata.copy()
            meta["text_snippet"] = chunk[:200] + "..."  # Store preview
            meta["full_text"] = chunk  # Store full text for RAG context

            items.append({"id": doc_id, "vector": vector, "meta": meta})

        # 2. Batch Upsert
        self.vector_api.upsert_batch(collection, items)
        return len(items)

    def chat(
        self,
        collection: str,
        query: str,
        k: int = 5,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer the user's question.",
        filters: Optional[Dict] = None,
    ) -> RAGGeneration:
        """
        Performs a full RAG cycle:
        1. Embeds the query.
        2. Retrieves relevant chunks from the DB.
        3. Constructs a prompt with context.
        4. Generates an answer using the LLM.
        """
        # 1. Retrieve Context
        query_vector = self.embed(query)

        search_result = self.vector_api.search(
            collection=collection,
            vector=query_vector,
            k=k,
            include_meta=True,
            filters=filters,
        )

        hits = search_result.get("hits", [])

        # 2. Build Context String
        context_parts = []
        source_docs = []

        for hit in hits:
            meta = hit.get("meta") or {}
            text = meta.get("full_text") or meta.get("text_snippet") or str(meta)
            context_parts.append(f"-- Source (ID: {hit['id']}):\n{text}")

            # Map to typed result
            source_docs.append(SearchResult(id=hit["id"], score=hit["score"], meta=meta, content=text))

        context_str = "\n\n".join(context_parts)

        # 3. Prompt LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{query}"},
        ]

        completion = self.openai.chat.completions.create(model=self.llm_model, messages=messages)

        answer = completion.choices[0].message.content

        return RAGGeneration(answer=answer, sources=source_docs, usage=completion.usage.model_dump())
