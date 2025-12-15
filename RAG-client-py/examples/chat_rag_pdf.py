from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from dotenv import find_dotenv, load_dotenv
from rustkissvdb import Client as RustClient, RustKissVDBError

# =================================================
# Load .env
# =================================================
load_dotenv(find_dotenv())


# =================================================
# Defaults (fallback)
# =================================================
DEFAULT_COLLECTION = "docs_portugues_exam"
DEFAULT_TOPK = 6
DEFAULT_MAX_CTX_CHARS = 5500
DEFAULT_HISTORY_WINDOW = 10

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


# =================================================
# Resolved config (single source of truth)
# =================================================
VDB_COLLECTION = os.getenv("VDB_COLLECTION", DEFAULT_COLLECTION).replace("-", "_")

PORT_RUST_KISS_VDB = int(os.getenv("PORT_RUST_KISS_VDB", "9917"))
VDB_BASE_URL = os.getenv("VDB_BASE_URL", f"http://localhost:{PORT_RUST_KISS_VDB}").rstrip("/")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:300m")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
RUSTKISS_API_KEY = os.getenv("RUSTKISS_API_KEY")

RAG_TOPK = int(os.getenv("RAG_TOPK", str(DEFAULT_TOPK)))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", str(DEFAULT_MAX_CTX_CHARS)))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", str(DEFAULT_HISTORY_WINDOW)))

CHAT_SESSION_ID = os.getenv("CHAT_SESSION_ID")  # opcional


# =================================================
# Helpers
# =================================================
def now_ts() -> int:
    return int(time.time())


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "...") if len(s) > n else s


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# =================================================
# RustKissVDB Client
# =================================================
@dataclass
class StateItem:
    key: str
    value: Any
    revision: int
    expires_at_ms: Optional[int] = None


class RustKissVDBClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0):
        if not base_url:
            raise ValueError("base_url requerido para RustKissVDBClient")
        self._client = RustClient(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)

    def health(self) -> None:
        self._client.request("GET", "/v1/health")

    # ---- state ----
    def state_list(self, prefix: str = "", limit: int = 100) -> List[StateItem]:
        data = self._client.state.list(prefix=prefix, limit=int(limit))
        return [
            StateItem(
                key=it["key"],
                value=it.get("value"),
                revision=int(it["revision"]),
                expires_at_ms=it.get("expires_at_ms"),
            )
            for it in data
        ]

    def state_get(self, key: str) -> StateItem:
        it = self._client.state.get(key)
        return StateItem(
            key=it["key"],
            value=it.get("value"),
            revision=int(it["revision"]),
            expires_at_ms=it.get("expires_at_ms"),
        )

    def state_put(self, key: str, value: Any, if_revision: Optional[int] = None) -> Dict[str, Any]:
        return self._client.state.put(key, value=value, if_revision=if_revision)

    def state_delete(self, key: str) -> bool:
        return bool(self._client.state.delete(key))

    # ---- vector ----
    def vector_search(
        self, collection: str, vector: List[float], k: int = 5, include_meta: bool = True
    ) -> Dict[str, Any]:
        return self._client.vector.search(
            collection,
            vector,
            k=int(k),
            include_meta=bool(include_meta),
        )


# =================================================
# Ollama clients
# =================================================
class OllamaEmbeddings:
    def __init__(self, model: str, base_url: str):
        if not base_url:
            raise ValueError("base_url requerido para OllamaEmbeddings")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=120)

    def embed(self, text: str) -> List[float]:
        r = self.http.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError(f"Ollama embeddings sin 'embedding': {data}")
        return [float(x) for x in vec]


class OllamaChat:
    def __init__(self, model: str, base_url: str):
        if not base_url:
            raise ValueError("base_url requerido para OllamaChat")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=240)

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        r = self.http.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "")


# =================================================
# Context building + sources
# =================================================
@dataclass
class SourceHit:
    page: Optional[int]
    score: float
    text: str
    chunk_index: Optional[int] = None
    pdf_path: Optional[str] = None


def hits_to_sources(hits: List[Dict[str, Any]]) -> List[SourceHit]:
    out: List[SourceHit] = []
    for h in hits:
        meta = h.get("meta") or {}
        out.append(
            SourceHit(
                page=meta.get("page"),
                score=float(h.get("score", 0.0)),
                text=str(meta.get("text") or ""),
                chunk_index=meta.get("chunk_index"),
                pdf_path=meta.get("pdf_path"),
            )
        )
    return out


def build_context(sources: List[SourceHit], max_chars: int) -> str:
    parts = []
    used = 0
    for s in sources:
        header = f"[p.{s.page} | score={s.score:.4f}]" if s.page else f"[score={s.score:.4f}]"
        block = f"{header}\n{s.text}".strip()
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def format_sources_footer(sources: List[SourceHit], max_items: int = 6) -> str:
    pages: List[int] = []
    for s in sources:
        if s.page is not None and s.page not in pages:
            pages.append(s.page)
    pages_str = ", ".join([f"p.{p}" for p in pages[:max_items]]) if pages else "(sin páginas)"
    return f"Fuentes usadas: {pages_str}"


# =================================================
# Persistent chat memory in VDB state
# =================================================
class ChatMemory:
    """
    Guarda historial persistente en:
      chat:{session_id}:history
    Estructura:
      {"messages":[{"role","content","ts"}...]}
    """

    def __init__(self, vdb: RustKissVDBClient, session_id: str):
        self.vdb = vdb
        self.session_id = session_id
        self.key = f"chat:{session_id}:history"

        # init if missing
        try:
            self.vdb.state_get(self.key)
        except Exception:
            self.vdb.state_put(self.key, {"messages": []})

    def load(self) -> List[Dict[str, str]]:
        it = self.vdb.state_get(self.key)
        data = it.value or {}
        msgs = data.get("messages", [])
        out: List[Dict[str, str]] = []
        for m in msgs:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": str(m["role"]), "content": str(m["content"])})
        return out

    def append(self, role: str, content: str) -> None:
        it = self.vdb.state_get(self.key)
        data = it.value or {"messages": []}
        data.setdefault("messages", [])
        data["messages"].append({"role": role, "content": content, "ts": now_ts()})
        self.vdb.state_put(self.key, data, if_revision=it.revision)

    def clear(self) -> None:
        for _ in range(5):
            it = self.vdb.state_get(self.key)
            try:
                self.vdb.state_put(self.key, {"messages": []}, if_revision=it.revision)
                return
            except RustKissVDBError:
                time.sleep(0.05)
        raise RuntimeError("No pude limpiar historial (CAS contention).")


# =================================================
# Manifest
# =================================================
def load_manifest(vdb: RustKissVDBClient, collection: str) -> Dict[str, Any]:
    key = f"docs:{collection}:manifest"
    it = vdb.state_get(key)
    return it.value if isinstance(it.value, dict) else {}


# =================================================
# Main
# =================================================
def main():
    collection = VDB_COLLECTION
    rag_topk = clamp_int(RAG_TOPK, 1, 50)
    max_ctx_chars = clamp_int(MAX_CTX_CHARS, 800, 20000)
    history_window = clamp_int(HISTORY_WINDOW, 2, 50)

    session_id = CHAT_SESSION_ID or uuid.uuid4().hex[:12]

    vdb = RustKissVDBClient(base_url=VDB_BASE_URL, api_key=RUSTKISS_API_KEY)
    vdb.health()

    emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    llm = OllamaChat(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)

    mem = ChatMemory(vdb=vdb, session_id=session_id)

    # manifest check
    try:
        manifest = load_manifest(vdb, collection)
        if manifest:
            print(
                f"📌 Manifest: chunks={manifest.get('chunks')} "
                f"embed_model={manifest.get('embed_model')} pdf={manifest.get('pdf_path')}"
            )
    except Exception:
        docs = vdb.state_list(prefix="docs:", limit=50)
        print(f"❌ No encuentro docs:{collection}:manifest. Disponibles:")
        for d in docs:
            print(" -", d.key)
        return

    # dim check (fail-fast)
    try:
        dim = len(emb.embed("dim_probe"))
        expected_dim = manifest.get("dim") if isinstance(manifest, dict) else None
        if expected_dim and int(expected_dim) != dim:
            print(f"❌ DIM mismatch: embedder dim={dim} != manifest dim={expected_dim}")
            return
    except Exception as e:
        print(f"❌ Embeddings fallaron: {e}")
        return

    print("✅ Chat RAG listo.")
    print(f"   vdb={VDB_BASE_URL} | collection='{collection}' | topk={rag_topk} | ctx_chars={max_ctx_chars}")
    print(f"   ollama={OLLAMA_BASE_URL} | embed='{OLLAMA_EMBED_MODEL}' | chat='{OLLAMA_CHAT_MODEL}'")
    print(f"   session={session_id}")
    print("Comandos:")
    print("  /exit, /clear, /history, /stats, /session, /sources")
    print("  /set topk N, /set chars N, /set window N\n")

    last_sources: List[SourceHit] = []
    manifest_cached = manifest

    while True:
        q = input("tú> ").strip()
        if not q:
            continue

        # commands
        if q.lower() in ("/exit", "exit", "quit"):
            break

        if q.lower() == "/clear":
            mem.clear()
            print("🧼 historial (persistente) limpiado.\n")
            continue

        if q.lower() == "/history":
            h = mem.load()
            for m in h[-30:]:
                print(f"{m['role']}: {m['content']}")
            print()
            continue

        if q.lower() == "/stats":
            h = mem.load()
            print(f"session={session_id} messages={len(h)} topk={rag_topk} ctx_chars={max_ctx_chars} window={history_window}")
            if manifest_cached:
                print("manifest:", short(safe_json(manifest_cached), 250))
            print()
            continue

        if q.lower() == "/session":
            print(f"CHAT_SESSION_ID={session_id}")
            print(f"state_key=chat:{session_id}:history\n")
            continue

        if q.lower() == "/sources":
            if not last_sources:
                print("(aún no hay fuentes)\n")
            else:
                for i, s in enumerate(last_sources[:10], start=1):
                    p = f"p.{s.page}" if s.page else "p.?"
                    print(f"{i}. {p} score={s.score:.4f} | {short(s.text, 140)}")
                print()
            continue

        if q.lower().startswith("/set "):
            parts = q.split()
            if len(parts) == 3:
                key, val = parts[1].lower(), parts[2]
                try:
                    n = int(val)
                    if key == "topk":
                        rag_topk = clamp_int(n, 1, 50)
                        print(f"ok topk={rag_topk}\n")
                        continue
                    if key == "chars":
                        max_ctx_chars = clamp_int(n, 800, 20000)
                        print(f"ok ctx_chars={max_ctx_chars}\n")
                        continue
                    if key == "window":
                        history_window = clamp_int(n, 2, 50)
                        print(f"ok window={history_window}\n")
                        continue
                except Exception:
                    pass
            print("uso: /set topk N | /set chars N | /set window N\n")
            continue

        # normal flow
        mem.append("user", q)

        try:
            qvec = emb.embed(q)
            res = vdb.vector_search(collection, qvec, k=rag_topk, include_meta=True)
        except RustKissVDBError as e:
            if "HTTP 404" in str(e):
                print(f"❌ colección '{collection}' no existe (¿VDB_COLLECTION correcto?).\n")
                continue
            print(f"❌ VDB error: {e}\n")
            continue
        except Exception as e:
            print(f"❌ Embeddings/retrieval falló: {e}\n")
            continue

        hits = res.get("hits", [])
        sources = hits_to_sources(hits)
        last_sources = sources

        ctx = build_context(sources, max_chars=max_ctx_chars)

        system = (
            "Eres un asistente en español.\n"
            "Reglas:\n"
            "1) Usa el CONTEXTO cuando sea relevante.\n"
            "2) Si el contexto NO alcanza, dilo explícitamente (no inventes).\n"
            "3) Cuando cites, indica página (p.X).\n\n"
            f"CONTEXTO:\n{ctx if ctx else '(vacío)'}\n"
        )

        history = mem.load()
        window = history[-history_window:]

        try:
            ans = llm.chat(system=system, messages=window)
        except Exception as e:
            ans = f"[error] Ollama chat falló: {e}"

        print(f"bot> {ans}")
        print(f"{format_sources_footer(sources)}\n")

        mem.append("assistant", ans)


if __name__ == "__main__":
    main()
