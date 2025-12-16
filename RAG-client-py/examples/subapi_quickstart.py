from __future__ import annotations

import os
import random
from typing import Any, Dict, List

from rustkissvdb import Config, RustKissVDBError


def title(text: str) -> None:
    print(f"\n=== {text} ===")


def demo_state(client) -> None:
    title("State Store")
    key = "sdk:demo:state"
    client.state.put(key, {"messages": 1})
    item = client.state.get(key)
    print("GET ->", item)
    items = client.state.list(prefix="sdk:demo", limit=10)
    print("LIST ->", items)


def demo_doc(client) -> None:
    title("DocStore")
    doc_id = "ticket_1"
    client.doc.put("tickets", doc_id, {"title": "Bug 1", "severity": "high"})
    print("GET ->", client.doc.get("tickets", doc_id))
    found = client.doc.find("tickets", {"severity": "high"})
    print("FIND ->", found)


def ensure_vector_collection(client, collection: str, dim: int) -> None:
    try:
        client.vector.create_collection(collection, dim=dim, metric="cosine")
        print(f"Collection '{collection}' created.")
    except RustKissVDBError as exc:
        if "already exists" not in str(exc):
            raise


def demo_vector(client, collection: str) -> None:
    title("Vector Store")
    ensure_vector_collection(client, collection, dim=3)
    vec = [round(random.random(), 3) for _ in range(3)]
    client.vector.upsert(
        collection,
        vector_id="demo_vec",
        vector=vec,
        meta={"tag": "sdk_demo"},
    )
    hits = client.vector.search(collection, vec, k=2, include_meta=True)
    print("SEARCH ->", hits)


def demo_sql(client) -> None:
    title("SQLite API")
    client.sql.exec("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)", params=[])
    client.sql.exec("INSERT INTO notes(body) VALUES (?)", params=["demo"])
    rows = client.sql.query("SELECT id, body FROM notes ORDER BY id DESC LIMIT 1", params=[])
    print("QUERY ->", rows)


def main() -> None:
    cfg = Config.from_env()
    collection = os.getenv("VDB_COLLECTION", "sdk_quickstart").replace("-", "_")
    with cfg.create_client() as client:
        demo_state(client)
        demo_doc(client)
        demo_vector(client, collection)
        demo_sql(client)


if __name__ == "__main__":
    main()
