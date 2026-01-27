import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from rustkissvdb import Client

# Pretty printing
console = Console()

def normalize_text(text: str) -> str:
    """Cleans up text formatting."""
    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def semantic_chunking(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chunks text preserving paragraph structure where possible.
    1. Splits by double newlines (paragraphs).
    2. Merges paragraphs until max_chunk_size is reached.
    3. If a single paragraph is too large, splits by sentence/chars (fallback).
    4. Adds overlap between chunks.
    """
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds max size
        if len(current_chunk) + len(para) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                # Take last 'overlap' chars, but try to keep it clean (not critical for simple implementation)
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                # The paragraph itself is huge, we must split it hard
                # (Simple fallback: just add it, or use fixed size splitter. 
                # For this demo, we'll just add it to avoid complexity, 
                # assuming paragraphs aren't massive)
                chunks.append(para.strip())
                current_chunk = "" 
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def main():
    console.print("[bold cyan]🤖 RustKissVDB + Ollama PDF RAG Demo[/bold cyan]")

    # --- Configuration ---
    PDF_PATH = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "2026-01-23-174258-PROYECTOS-INBOUND_docs_bundle.pdf"
    )
    
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"
    
    # Models
    CHAT_MODEL = "nemotron-3-nano:30b-cloud"
    EMBED_MODEL = "embeddinggemma:300m"
    
    COLLECTION = "PdfProyectoInbound"

    # 1. Initialize Client
    client = Client(
        base_url="http://localhost:1234",
        api_key="dev",
    )

    # Configure RAG for Ollama
    client.rag.openai.base_url = OLLAMA_BASE_URL
    client.rag.openai.api_key = OLLAMA_API_KEY
    client.rag.llm_model = CHAT_MODEL
    client.rag.embedding_model = EMBED_MODEL

    # 2. Setup Collection
    with console.status(f"[bold blue]Initializing collection '{COLLECTION}'...[/bold blue]"):
        try:
            dim = client.rag.initialize_collection(COLLECTION, metric="cosine")
        except Exception as e:
            console.print(f"[red]Collection error: {e}[/red]")
            return
    console.print(f"[green]Collection ready (dim: {dim})[/green]")

    # 3. Ingest PDF Data
    if not os.path.exists(PDF_PATH):
        console.print(f"[red]PDF not found at: {PDF_PATH}[/red]")
        return

    # Extract text and prepare chunks first
    console.print(f"[bold blue]Reading PDF: {os.path.basename(PDF_PATH)}...[/bold blue]")
    try:
        doc = fitz.open(PDF_PATH)
        total_pages = len(doc)
    except Exception as e:
        console.print(f"[red]Error opening PDF: {e}[/red]")
        return

    all_chunks_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Phase 1: Processing Text
        task_extract = progress.add_task("[cyan]Processing pages...", total=total_pages)
        
        for i in range(total_pages):
            page = doc.load_page(i)
            text = page.get_text()
            page_num = i + 1
            
            # Semantic chunking
            page_chunks = semantic_chunking(text)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                all_chunks_data.append({
                    "text": chunk_text,
                    "meta": {
                        "source": os.path.basename(PDF_PATH),
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "text_snippet": chunk_text[:200] + "...",
                        "full_text": chunk_text
                    },
                    "id": f"p{page_num}_c{chunk_idx}"
                })
            
            progress.advance(task_extract)

        # Phase 2: Embedding & Ingestion (Batched)
        BATCH_SIZE = 10  # Adjust based on memory/model speed
        total_chunks = len(all_chunks_data)
        task_ingest = progress.add_task(f"[green]Ingesting {total_chunks} chunks...", total=total_chunks)

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = all_chunks_data[i : i + BATCH_SIZE]
            
            # Prepare batch for embedding
            texts_to_embed = [item["text"] for item in batch]
            
            try:
                # Generate embeddings in batch
                embeddings = client.rag.embed(texts_to_embed)
                
                # If single vector returned (shouldn't happen with list input but safety check)
                if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], float):
                    embeddings = [embeddings]

                # Prepare upsert items
                upsert_items = []
                for idx, item in enumerate(batch):
                    upsert_items.append({
                        "id": item["id"],
                        "vector": embeddings[idx],
                        "meta": item["meta"]
                    })
                
                # Batch Upsert
                client.rag.vector_api.upsert_batch(COLLECTION, upsert_items)
                
            except Exception as e:
                console.print(f"[red]Error ingesting batch: {e}[/red]")
            
            progress.advance(task_ingest, advance=len(batch))

    console.print(f"[green]Successfully ingested {total_chunks} chunks![/green]")

    # 4. Interactive Chat Loop
    console.print("\n[bold yellow]✨ Chat ready! Type 'exit' or 'quit' to stop.[/bold yellow]")
    
    while True:
        question = Prompt.ask("\n[bold purple]Question[/bold purple]")
        
        if question.lower() in ("exit", "quit"):
            break
            
        if not question.strip():
            continue

        console.print(f"[dim]Thinking with {CHAT_MODEL}...[/dim]")
        try:
            response = client.rag.chat(collection=COLLECTION, query=question, k=3)
            
            console.print("\n[bold purple]Answer:[/bold purple]")
            console.print(Markdown(response.answer))
            
            console.print("\n[dim]Sources used:[/dim]")
            for src in response.sources:
                page_info = f"Page {src.meta.get('page', '?')}"
                # Show a bit more context in the source snippet if possible
                snippet = src.meta.get('full_text', '')[:150].replace('\n', ' ')
                console.print(f"- {page_info}: {snippet}... (Score: {src.score:.4f})")
                
        except Exception as e:
            console.print(f"[red]Error during chat: {e}[/red]")

    console.print("[bold cyan]Goodbye![/bold cyan]")

if __name__ == "__main__":
    main()
