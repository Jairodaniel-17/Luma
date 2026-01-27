# 🧠 rust-kiss-vdb

**Low-memory Exact Vector Database with Intelligent Grouped Search for RAG**

> *Because RAG without collapsing is just noise.*

---

## 🚀 What is `rust-kiss-vdb`?

`rust-kiss-vdb` is a **minimalist, high-performance vector database** written in **Rust**, designed for:

* **Exact vector search**
* **Ultra-low RAM usage**
* **Deterministic results**
* **First-class support for RAG-grade grouping / collapsing**

It targets scenarios where:

* FAISS is too primitive
* Qdrant is too heavy
* Milvus is overkill
* Oracle DB is not designed for embeddings
* **You actually care about result quality, not just speed**

---

## 🎯 Core Design Goals

| Goal                    | Description                                     |
| ----------------------- | ----------------------------------------------- |
| 🧠 RAG-first            | Designed around how RAG *should* work           |
| 🧮 Exact search         | No approximations, no HNSW surprises            |
| 🪶 Low memory           | Works in **~1.5 MB RAM**                        |
| 🧩 Intelligent grouping | Collapse noisy chunks into meaningful documents |
| 🔒 Deterministic        | Same query → same results                       |
| 🧰 Simple API           | No magic, no hidden heuristics                  |

---

## 🔥 Killer Feature: Grouped / Collapsed Search (Priority #1)

### Why this matters

> **RAG without collapsing = garbage output**

Most vector DBs return:

* 10 chunks
* From the same document
* With almost identical embeddings

That destroys:

* Context diversity
* Answer quality
* Trust in the system

---

### ✅ How `rust-kiss-vdb` solves this

We introduce **first-class grouping** at query time.

```json
{
  "query": "how to configure OAuth",
  "top_k": 20,
  "group_by": "document_id",
  "group_limit": 1,
  "filters": {
    "language": "en",
    "status": "published"
  }
}
```

---

### 🧠 Grouping semantics

| Concept           | Behavior                                 |
| ----------------- | ---------------------------------------- |
| `group_by`        | `document_id` or `group_id`              |
| Group score       | **max score of all chunks in the group** |
| Returned metadata | Metadata of the **best chunk**           |
| Content           | Full chunk content (not just metadata)   |
| Result count      | Controlled by `top_k` *after grouping*   |

---

### 🧩 What you get

Instead of this ❌

| Rank | Chunk    | Document |
| ---- | -------- | -------- |
| 1    | chunk_42 | doc_A    |
| 2    | chunk_43 | doc_A    |
| 3    | chunk_44 | doc_A    |

You get this ✅

| Rank | Document | Best Chunk |
| ---- | -------- | ---------- |
| 1    | doc_A    | chunk_42   |
| 2    | doc_B    | chunk_7    |
| 3    | doc_C    | chunk_19   |

**That is RAG-ready output.**

---

## 📦 Data Model

### Stored chunk

```rust
struct StoredChunk {
    embedding: Vec<f32>,
    content: String,
    metadata: {
        document_id: String,
        group_id: Option<String>,
        file_name: String,
        processed_at: DateTime,
        tags: HashMap<String, String>
    }
}
```

---

## 🔍 Search Capabilities

### Supported features

* ✅ Exact cosine similarity
* ✅ Optional metadata filters
* ✅ Optional grouping / collapsing
* ✅ Top-K control
* ✅ Content + metadata retrieval
* ✅ Streaming index (AppendLog)
* ❌ No ANN (by design)

---

## 🧪 Example Search Response

```json
{
  "results": [
    {
      "score": 0.9123,
      "document_id": "auth_guide_v2",
      "content": "OAuth tokens must be refreshed using...",
      "metadata": {
        "file_name": "auth.md",
        "processed_at": "2026-01-25T14:33:00Z"
      }
    }
  ]
}
```

---

## 🧠 Why Exact Search?

Because:

* You can’t debug ANN
* You can’t explain ANN
* You can’t trust ANN for small / medium corpora

If your dataset fits in memory → **exact search wins**.

---

## ⚖️ Comparison with Other Vector Databases

### 🟦 Qdrant

| Aspect                 | Qdrant             | rust-kiss-vdb            |
| ---------------------- | ------------------ | ------------------------ |
| Grouping               | ⚠️ Basic / shallow | ✅ Native & deterministic |
| Memory                 | ❌ Heavy            | ✅ ~1.5 MB                |
| ANN bias               | Yes                | No                       |
| RAG quality            | ⚠️ Medium          | ✅ High                   |
| Operational complexity | High               | Low                      |

---

### 🟩 Milvus

| Aspect       | Milvus     | rust-kiss-vdb     |
| ------------ | ---------- | ----------------- |
| Deployment   | Kubernetes | Single binary     |
| Memory       | Very high  | Extremely low     |
| Exact search | Limited    | First-class       |
| RAG focus    | No         | Yes               |
| Use case     | Big data   | Precision systems |

---

### 🟥 Oracle DB (Vector Search)

| Aspect       | Oracle         | rust-kiss-vdb |
| ------------ | -------------- | ------------- |
| License      | 💰 Paid        | 🆓 Open       |
| Purpose      | General DB     | Vector-native |
| Grouping     | SQL workaround | Native        |
| Cost         | Very high      | Zero          |
| Dev velocity | Slow           | Fast          |

---

## 🧬 Philosophy

> **This is not a “database for everything”.**
> This is a **precision instrument**.

If you want:

* Speed at any cost → ANN
* Big clusters → Milvus
* Enterprise lock-in → Oracle

If you want:

* Clean RAG
* Explainable results
* Low resources
* Real control

👉 `rust-kiss-vdb`

---

## 🛣️ Roadmap

* [x] Exact vector search
* [x] Metadata filters
* [x] Grouped / collapsed search
* [x] Streaming append storage
* [ ] Hybrid lexical + vector scoring
* [ ] Pluggable embedding backends
* [ ] On-disk mmap index
* [ ] gRPC interface

---

## 🧠 Final Thought

> **Vector search is easy.
> Good RAG is not.**

`rust-kiss-vdb` is built for the second one.
