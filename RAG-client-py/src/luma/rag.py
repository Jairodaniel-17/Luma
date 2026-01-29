from typing import Any, Dict, List, Optional

from openai import OpenAI

from .errors import LumaError
from .models import VectorRecord


class RAGClient:
    def __init__(
        self,
        vector_api,
        openai_key: Optional[str] = None,
        embedding_model: str = "baai/bge-m3",
        llm_model: str = "openai/gpt-oss-120b",
        base_url: str = "https://api.novita.ai/openai",
        use_ollama: bool = False,
        **_,
    ):
        if use_ollama:
            raise LumaError("Ollama no soportado en este RAGClient (Novita only)")

        if not openai_key:
            raise LumaError("OPENAI / NOVITA API key requerida")

        self.vector = vector_api
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.client = OpenAI(
            api_key=openai_key,
            base_url=base_url,
        )

        self._embedding_dim: Optional[int] = None

    # ---------------------------
    # Embeddings
    # ---------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [d.embedding for d in res.data]

    # ---------------------------
    # Init collection
    # ---------------------------

    def initialize_collection(self, name: str, metric: str) -> int:
        if self._embedding_dim is None:
            vec = self.embed(["dimension probe"])[0]
            self._embedding_dim = len(vec)

        self.vector.create_collection(
            collection=name,
            dim=self._embedding_dim,
            metric=metric,
        )
        return self._embedding_dim

    # ---------------------------
    # Ingest
    # ---------------------------

    def ingest_text(
        self,
        collection: str,
        text: str,
        metadata: Optional[Dict[str, Any]],
        chunk_size: int,
        overlap: int,
        id_prefix: str,
    ) -> int:
        chunks = self._chunk(text, chunk_size, overlap)
        embeddings = self.embed(chunks)

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            records.append(
                VectorRecord(
                    id=f"{id_prefix}_{i}",
                    vector=vector,
                    metadata={
                        **(metadata or {}),
                        "content": chunk,
                        "chunk": i,
                    },
                )
            )

        self.vector.upsert_batch(
            collection=collection,
            items=[{"id": r.id, "vector": r.vector, "meta": r.metadata} for r in records],
        )

        return len(records)

    # ---------------------------
    # Chat RAG
    # ---------------------------

    def chat(
        self,
        collection: str,
        query: str,
        k: int,
        system_prompt: str,
        filters: Optional[Dict],
        temperature: float,
        max_tokens: Optional[int],
    ):
        q_vector = self.embed([query])[0]

        res = self.vector.search(
            collection=collection,
            vector=q_vector,
            k=k,
            include_meta=True,
            filters=filters,
        )

        hits = res.get("hits", [])
        if not hits:
            return {
                "answer": "No se encontró información relevante en la base de conocimiento.",
                "sources": [],
            }

        context_blocks = []
        sources = []

        for h in hits:
            meta = h.get("meta", {})
            content = meta.get("content", "")
            context_blocks.append(content)

            sources.append(
                {
                    "id": h["id"],
                    "score": h["score"],
                    "meta": meta,
                }
            )

        prompt = self._build_prompt(query, context_blocks)

        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": completion.choices[0].message.content,
            "sources": sources,
            "usage": completion.usage.model_dump() if completion.usage else None,
        }

    # ---------------------------
    # Utils
    # ---------------------------

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk = words[i : i + size]
            chunks.append(" ".join(chunk))
            i += size - overlap

        return chunks

    @staticmethod
    def _build_prompt(question: str, contexts: List[str]) -> str:
        joined = "\n\n".join(f"[Contexto {i + 1}]\n{c}" for i, c in enumerate(contexts))
        return f"""
Responde usando SOLO la información del contexto.
Si no está en el contexto, di que no se encuentra.

{joined}

Pregunta:
{question}

Respuesta:
""".strip()
