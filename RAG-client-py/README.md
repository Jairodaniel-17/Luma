# RAG-client-py

SDK Python oficial para RustKissVDB + ejemplos de referencia (chat RAG + ingestión de PDF).

## Instalación

```bash
uv pip install -e .
# o
pip install -e .
```

Extras para las demos (`PyMuPDF`):

```bash
pip install -e .[rag]
```

## Configuración

Copia `.env.example` → `.env` y ajusta:

```
RUSTKISS_URL=http://127.0.0.1:9917
RUSTKISS_API_KEY=dev
VDB_COLLECTION=docs_demo
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=embeddinggemma:300m
OLLAMA_CHAT_MODEL=gemma3:4b
```

El SDK también expone `Config.from_env()` para cargar estas variables automáticamente.

## Uso básico

```python
from rustkissvdb import Client, Config

cfg = Config.from_env()
with Client(cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout) as client:
    client.state.put("foo", {"value": 1})
    hits = client.vector.search("docs", [0.1, 0.2], k=3, include_meta=True)
    client.doc.put("tickets", "tk_1", {"title": "Bug"})
    client.sql.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)")
```

## Ejemplos

- `examples/chat_rag_pdf.py`: CLI de chat que almacena memoria en RustKissVDB y usa Ollama para embeddings/chat.
- `examples/ingest_pdf_to_vdb.py`: chunk + embed de un PDF y lo sube a la colección vectorial.

Ambos se apoyan en el SDK (`rustkissvdb.Client`) y usan las variables `.env`.

## Scripts legacy

Los scripts originales fueron migrados a `examples/`. El paquete exporta:

- `Client` (`client.state`, `client.vector`, `client.doc`, `client.sql`, `client.stream`)
- `Config` (`Config.from_env()`)
- `RustKissVDBError`

Consulta `docs/SDK_PYTHON.md` para más detalles.
