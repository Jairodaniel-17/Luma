import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from rustkissvdb import Client as RustClient, RustKissVDBError

# ---------------------------
#  RustKissVDB minimal client
# ---------------------------

class RustKissVDBClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0):
        port = int(os.getenv("PORT_RUST_KISS_VDB", "9917"))
        resolved = (base_url or f"http://localhost:{port}").rstrip("/")
        api_key = os.getenv("RUSTKISS_API_KEY")
        self._client = RustClient(base_url=resolved, api_key=api_key, timeout=timeout)

    def health(self) -> bool:
        self._client.request("GET", "/v1/health")
        return True

    def state_put(self, key: str, value: Any) -> Dict[str, Any]:
        return self._client.state.put(key, value=value)

    def vector_create(self, collection: str, dim: int, metric: str = "cosine") -> None:
        try:
            self._client.vector.create_collection(collection, dim=dim, metric=metric)
        except RustKissVDBError as exc:
            if "already_exists" not in str(exc):
                raise

    def vector_upsert(self, collection: str, _id: str, vector: List[float], meta: Any) -> None:
        self._client.vector.upsert(
            collection,
            vector_id=_id,
            vector=vector,
            meta=meta,
        )


# ---------------------------
#  Ollama embeddings
# ---------------------------

class OllamaEmbeddings:
    def __init__(self, model: str = "embeddinggemma:300m", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=120)

    def embed(self, text: str) -> List[float]:
        r = self.http.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text})
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError(f"Ollama embeddings sin 'embedding': {data}")
        return [float(x) for x in vec]


# ---------------------------
#  PDF -> chunks
# ---------------------------

def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Chunking simple por caracteres con overlap.
    """
    text = normalize_ws(text)
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def read_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for idx in range(len(doc)):
        page = doc[idx]
        txt = page.get_text("text")
        pages.append({"page": idx + 1, "text": txt})
    doc.close()
    return pages


def main():
    PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\jairo\Downloads\JDMT_EXAME FINAL DE PORTUGUÊS.pdf")

    COLLECTION = os.getenv("VDB_COLLECTION", "docs_portugues_exam").replace("-", "_")
    METRIC = os.getenv("VDB_METRIC", "cosine")
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:300m")

    vdb = RustKissVDBClient()
    vdb.health()

    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    pages = read_pdf_pages(PDF_PATH)

    # Unimos páginas y chunkamos manteniendo page_range aproximado
    all_chunks = []
    for p in pages:
        p_chunks = chunk_text(p["text"], chunk_chars=1200, overlap=150)
        for c in p_chunks:
            all_chunks.append({"page": p["page"], "text": c})

    if not all_chunks:
        raise SystemExit("No se extrajo texto del PDF (¿es escaneado sin texto?).")

    # Detectar dim real con un embed de prueba
    dim = len(embedder.embed("dim_probe"))
    vdb.vector_create(COLLECTION, dim=dim, metric=METRIC)

    # Guardar “manifest” en state (opcional, útil para inspección)
    manifest_key = f"docs:{COLLECTION}:manifest"
    vdb.state_put(manifest_key, {
        "pdf_path": PDF_PATH,
        "collection": COLLECTION,
        "embed_model": EMBED_MODEL,
        "chunks": len(all_chunks),
    })

    # Upsert chunks
    for i, ch in enumerate(all_chunks, start=1):
        text = ch["text"]
        vec = embedder.embed(text)
        chunk_id = f"chunk_{i:05d}_{uuid.uuid4().hex[:8]}"
        meta = {
            "source": "pdf",
            "pdf_path": PDF_PATH,
            "page": ch["page"],
            "chunk_index": i,
            "text": text,
        }
        vdb.vector_upsert(COLLECTION, chunk_id, vec, meta=meta)

        if i % 20 == 0:
            print(f"Ingestados {i}/{len(all_chunks)}...")

    print(f"✅ Listo. Collection='{COLLECTION}' chunks={len(all_chunks)} dim={dim}")
    print(f"Manifest en state: {manifest_key}")


if __name__ == "__main__":
    main()
