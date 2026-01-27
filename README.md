# 🧠 rust-kiss-vdb

**Low-memory Exact & Approximate Vector Database with Intelligent Grouped Search for RAG**

> *Because RAG without collapsing is just noise.*

---

## 🚀 What is `rust-kiss-vdb`?

`rust-kiss-vdb` is a **minimalist, high-performance vector database** written in **Rust**. It provides both exact and approximate (ANN) search, with a core focus on **RAG-grade result grouping** to deliver clean, contextually relevant results.

It's designed for scenarios where:

*   FAISS is too primitive
*   Qdrant is too heavy
*   Milvus is overkill
*   You need deterministic, high-quality results for your RAG pipeline
*   **You actually care about result quality, not just speed**

---

## 🔥 Core Features

*   **Vector Search**: High-performance exact and approximate (HNSW) vector search.
*   **Intelligent Grouping**: First-class support for collapsing search results by a specific metadata field (e.g., `document_id`), ensuring contextually diverse and meaningful results for RAG.
*   **Low Memory Footprint**: Optimized for low-resource environments, running comfortably with just a few MB of RAM.
*   **HTTP API**: A simple, powerful API for easy integration into any stack.
*   **Python Client**: An ergonomic Python client (`RAG-client-py`) for seamless interaction from your Python applications.
*   **CLI**: A command-line interface for database administration and maintenance tasks.
*   **Persistent & Embeddable**: Built on `redb` for persistent storage, with a straightforward on-disk format.

---

## 📦 Getting Started

### Building from Source

1.  **Install Rust**: If you don't have it, get it from [rust-lang.org](https://www.rust-lang.org/).
2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/rust-kiss-vdb.git
    cd rust-kiss-vdb
    ```
3.  **Build the Project**:
    ```bash
    cargo build --release
    ```
    The binary will be located at `target/release/rust-kiss-vdb`.

### Running the Server

You can run the server directly:

```bash
./target/release/rust-kiss-vdb
```

By default, it will run on `127.0.0.1:8000` and store data in a temporary directory.

### Configuration

Configuration is managed via command-line arguments and environment variables. For a complete list of all available options and their corresponding environment variables, please see the detailed configuration guide:

➡️ **[Full Configuration Guide](./docs/CLI.md)**

### API Guide

The server exposes a simple RESTful API for creating collections, managing vectors, and performing searches. For a complete guide with `curl` examples and details on request/response formats, please see the API documentation:

➡️ **[Complete API Guide](./docs/API.md)**


---

## 🧬 Philosophy

> **This is not a “database for everything”.**
> This is a **precision instrument** for high-quality RAG.

If you want:

*   Speed at any cost → Use a pure ANN library
*   Big, distributed clusters → Use Milvus
*   Enterprise lock-in → Use a proprietary vendor

If you want:

*   **Clean, grouped RAG results**
*   **Explainable and deterministic output**
*   **Low resource usage and simple deployment**
*   **Real control over your search logic**

👉 `rust-kiss-vdb` is for you.

---

## 🛣️ Roadmap

*   [x] Exact & HNSW vector search
*   [x] Metadata filters
*   [x] Grouped / collapsed search
*   [x] Streaming append storage
*   [x] Python Client
*   [ ] On-disk ANN index (experimental `DiskAnn` support)
*   [ ] Hybrid lexical + vector scoring
*   [ ] gRPC interface

---

## ⚖️ Comparison with Other Vector Databases

### 🟦 Qdrant

| Aspect                 | Qdrant             | rust-kiss-vdb                      |
| ---------------------- | ------------------ | ---------------------------------- |
| Grouping               | Basic / shallow    | ✅ Native, deterministic & RAG-focused |
| Memory                 | Heavy              | ✅ Extremely Low (~1.5 MB idle)      |
| Search                 | ANN-first          | Both Exact & ANN (HNSW)            |
| RAG Quality            | Medium             | ✅ High                            |
| Operational Complexity | High               | Low (single binary)                |

### 🟩 Milvus

| Aspect       | Milvus     | rust-kiss-vdb     |
| ------------ | ---------- | ----------------- |
| Deployment   | Kubernetes | Single binary     |
| Memory       | Very high  | Extremely low     |
| Search       | ANN-focused| Both Exact & ANN  |
| RAG Focus    | No         | Yes               |
| Use Case     | Big Data   | Precision RAG     |