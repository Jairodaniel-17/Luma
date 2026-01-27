import os
from rustkissvdb import Client
from rich.console import Console
from rich.markdown import Markdown

# Pretty printing
console = Console()

def main():
    # 1. Initialize Client (Auto-loads from .env)
    client = Client()
    
    COLLECTION = "knowledge-base"
    
    # 2. Setup Collection (Auto-detect dimension!)
    console.print(f"[bold blue]Initializing collection '{COLLECTION}'...[/bold blue]")
    try:
        # No need to specify dim=1536 anymore!
        dim = client.rag.initialize_collection(COLLECTION, metric="cosine")
        console.print(f"[green]Collection ready with dimension: {dim}[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing collection: {e}[/red]")
        return

    # 3. Ingest Data (Automatic Chunking & Embedding)
    long_text = """
    RustKissVDB is a high-performance, low-memory vector database written in Rust.
    It is designed specifically for RAG (Retrieval-Augmented Generation) applications.
    Unlike general-purpose vector stores like Qdrant or Milvus, RustKissVDB focuses on 
    grouping and collapsing search results to provide diverse context to LLMs.
    
    Key features include:
    - Real-time SSE streaming for event subscriptions.
    - Built-in support for optimistic locking on state management.
    - A specialized high-level search API that mocks embedding generation for testing.
    - Compatibility with OpenAI-compatible embedding models via its Python SDK.
    
    The architecture uses a Write-Ahead Log (WAL) for durability and supports 
    both in-memory and disk-based operation.
    """
    
    console.print("[bold blue]Ingesting document...[/bold blue]")
    chunks = client.rag.ingest_text(
        collection=COLLECTION, 
        text=long_text, 
        metadata={"source": "manual", "topic": "database_intro"}
    )
    console.print(f"[green]Successfully ingested {chunks} chunks![/green]")

    # 4. RAG Chat
    question = "What makes RustKissVDB different from Milvus?"
    console.print(f"\n[bold purple]Question:[/bold purple] {question}")
    
    response = client.rag.chat(
        collection=COLLECTION,
        query=question,
        k=3
    )
    
    console.print("\n[bold purple]Answer:[/bold purple]")
    console.print(Markdown(response.answer))
    
    console.print("\n[dim]Sources used:[/dim]")
    for src in response.sources:
        console.print(f"- {src.meta.get('text_snippet')} (Score: {src.score:.4f})")

if __name__ == "__main__":
    main()