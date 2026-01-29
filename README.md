# Luma

**Luma** is a high-performance, developer-friendly Vector Database powered by **RustKissVDB**.

It is designed to be lightweight, fast, and easy to use, offering a familiar API for developers coming from other vector databases like Qdrant, while maintaining the raw speed and simplicity of the underlying RustKissVDB engine.

## Features

- **Blazing Fast**: Written in Rust for maximum performance.
- **Developer Friendly**: Python SDK with intuitive API (`luma-vdb`).
- **Standardized**: Uses standard JSON APIs and generic vector/collection concepts.
- **Docker Ready**: Deploy in seconds with `docker-compose`.
- **LangChain Integration**: Built-in support for LangChain RAG workflows.

## Getting Started

### 1. Run with Docker

```bash
docker-compose up -d
```

The server will start on port `9917`.

### 2. Install Python SDK

```bash
pip install luma-vdb
```

### 3. Usage Example

```python
from luma import Client, CollectionConfig, Distance, VectorRecord

client = Client(url="http://localhost:9917")

# Create Collection
client.create_collection("my_docs", CollectionConfig(dim=1536, metric=Distance.COSINE))

# Upsert
client.upsert("my_docs", [
    VectorRecord(id="1", vector=[0.1, ...], metadata={"title": "Hello World"})
])

# Search
results = client.search("my_docs", vector=[0.1, ...], k=3)
```

## "Powered by RustKissVDB"

Luma builds upon the solid foundation of RustKissVDB, extending it with:
- Improved API ergonomics.
- Better tooling and SDKs.
- Clearer documentation and naming conventions.

## License

MIT
