import os
from typing import Any, Dict, List, Optional

import httpx

from .doc import DocAPI
from .errors import LumaError
from .models import CollectionConfig, OperationInfo, SearchResult, VectorRecord
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


class LumaClient:
    """
    Cliente principal para Luma (powered by RustKissVDB).

    Integra operaciones de base de datos (Vector, State, SQL) y capacidades de RAG.

    Soporta:
    - Ollama (local)
    - OpenAI-compatible APIs (Novita, OpenAI, etc.)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        # --- RAG / LLM ---
        openai_key: Optional[str] = None,
        rag_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        # --- Ollama ---
        use_ollama: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_embedding_model: str = "embeddinggemma:300m",
        ollama_llm_model: str = "llama3.2:latest",
        timeout: float = 60.0,
        port: Optional[int] = None,
        host: Optional[str] = None,
    ) -> None:
        # -------------------------
        # Resolver URL del backend
        # -------------------------
        target_url = url or base_url
        if not target_url and host:
            p = port or 1234
            target_url = f"http://{host}:{p}"

        self._base_url = (target_url or os.getenv("KISS_VDB_URL", "http://localhost:1234")).rstrip("/")

        self._api_key = api_key or os.getenv("KISS_VDB_KEY", "dev")

        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

        # -------------------------
        # APIs Low Level
        # -------------------------
        self.state = StateAPI(self)
        self.vector = VectorAPI(self)
        self.doc = DocAPI(self)
        self.sql = SqlAPI(self)
        self.stream = StreamAPI(self)

        # -------------------------
        # API RAG (High Level)
        # -------------------------
        if use_ollama:
            # Ollama explícito (modo legacy / local)
            self.rag = RAGClient(
                vector_api=self.vector,
                use_ollama=True,
                ollama_base_url=ollama_base_url,
                ollama_embedding_model=ollama_embedding_model,
                llm_model=ollama_llm_model,
            )
        else:
            # OpenAI-compatible (Novita por defecto)
            self.rag = RAGClient(
                vector_api=self.vector,
                openai_key=openai_key or os.getenv("OPENAI_API_KEY"),
                base_url=rag_base_url or os.getenv("RAG_BASE_URL", "https://api.novita.ai/openai"),
                embedding_model=embedding_model or os.getenv("EMBEDDING_MODEL", "baai/bge-m3"),
                llm_model=llm_model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
                use_ollama=False,
            )

    # ------------------------------------------------------------------
    # High Level API (Generic)
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        config: CollectionConfig,
    ) -> bool:
        self.vector.create_collection(
            collection=name,
            dim=config.dim,
            metric=config.metric.value,
        )
        return True

    def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> OperationInfo:
        items = [{"id": str(r.id), "vector": r.vector, "meta": r.metadata} for r in records]
        self.vector.upsert_batch(collection=collection, items=items)
        return OperationInfo()

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        res = self.vector.search(
            collection=collection,
            vector=vector,
            k=k,
            include_meta=include_metadata,
            filters=filter,
        )

        results = []
        for h in res.get("hits", []):
            results.append(
                SearchResult(
                    id=h["id"],
                    score=h["score"],
                    metadata=h.get("meta"),
                    vector=h.get("vector"),
                )
            )
        return results

    # ------------------------------------------------------------------
    # RAG helpers
    # ------------------------------------------------------------------

    def create_rag_collection(
        self,
        name: str,
        metric: str = "cosine",
    ) -> int:
        return self.rag.initialize_collection(name, metric)

    def ingest_document(
        self,
        collection: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        overlap: int = 100,
        id_prefix: str = "doc",
    ) -> int:
        return self.rag.ingest_text(
            collection=collection,
            text=text,
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
            id_prefix=id_prefix,
        )

    def ask(
        self,
        collection: str,
        question: str,
        k: int = 5,
        system_prompt: str = ("You are a helpful assistant. Use the provided context to answer the user's question."),
        filters: Optional[Dict] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Any:
        return self.rag.chat(
            collection=collection,
            query=question,
            k=k,
            system_prompt=system_prompt,
            filters=filters,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LumaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

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
            raise LumaError(f"Connection error: {e}")

        if resp.status_code >= 400:
            raise LumaError(self._error_message(resp))

        if resp.headers.get("content-type", "").startswith("application/json"):
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
