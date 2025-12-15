# SDK_PYTHON.md

## Instalación local

```bash
cd RAG-client-py
pip install -e .
# o
uv pip install -e .
```

Requisitos mínimos: Python 3.11, `httpx`, `python-dotenv`. Las demos (`examples/`) opcionalmente usan `pymupdf` (instala el extra `pip install -e .[rag]`).

## Carga de configuración

```python
from rustkissvdb import Client, Config

cfg = Config.from_env()  # lee .env y variables RUSTKISS_*
with Client(cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout) as client:
    client.health()
```

Variables soportadas:

| Variable            | Ejemplo                         |
|---------------------|---------------------------------|
| `RUSTKISS_URL`      | `http://127.0.0.1:9917`         |
| `RUSTKISS_API_KEY`  | `super-secret`                  |
| `RUSTKISS_TIMEOUT`  | `60` (segundos)                 |

## APIs expuestas

```python
from rustkissvdb import Client

client = Client("http://localhost:9917")

# State Store
client.state.put("foo", {"value": 1})
item = client.state.get("foo")
client.state.batch_put([
    {"key": "bulk:1", "value": {"x": 1}},
    {"key": "bulk:2", "value": {"x": 2}},
])

# Vector Store
client.vector.create_collection("docs", dim=768, metric="cosine")
client.vector.upsert("docs", vector_id="doc1", vector=[0.1, 0.2], meta={"title": "demo"})
hits = client.vector.search("docs", [0.1, 0.2], k=3, include_meta=True)

# DocStore
client.doc.put("tickets", "tk_1", {"title": "Bug #1", "severity": "high"})
client.doc.find("tickets", filter={"severity": "high"})

# SQL (SQLite embebido)
client.sql.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)")
client.sql.query("SELECT * FROM notes WHERE id = ?", params=[1])

# SSE stream
for event in client.stream.events(since=0, types="state_updated"):
    print(event)
```

## Ejemplos incluidos

- `examples/chat_rag_pdf.py`: chat RAG en consola usando RustKissVDB + Ollama (embeddings + chat).
- `examples/ingest_pdf_to_vdb.py`: ingesta de PDF (chunks + embeddings).

Ambos ejemplos consumen el SDK y cargan `VDB_BASE_URL`, `OLLAMA_*` y `RUSTKISS_API_KEY` desde `.env`.
