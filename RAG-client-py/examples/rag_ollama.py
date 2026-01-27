from rich.console import Console
from rich.markdown import Markdown

from rustkissvdb import Client

# Pretty printing
console = Console()


def main():
    console.print("[bold cyan]🤖 RustKissVDB + Ollama RAG Demo[/bold cyan]")

    # --- Configuration for Ollama ---
    # Ollama usually runs on port 11434.
    # We point to the /v1 endpoint for OpenAI compatibility.
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"  # Not used by Ollama, but required by the SDK

    # Models pulled in Ollama (Make sure you ran `ollama pull llama3` and `ollama pull nomic-embed-text`)
    CHAT_MODEL = "nemotron-3-nano:30b-cloud"
    EMBED_MODEL = "embeddinggemma:300m"

    # 1. Initialize Client
    # We pass the OpenAI configuration explicitly here to override .env if needed
    client = Client(
        base_url="http://localhost:1234",  # RustKissVDB URL
        api_key="dev",  # RustKissVDB Key
    )

    # Configure the RAG subsystem to talk to Ollama
    client.rag.openai.base_url = OLLAMA_BASE_URL
    client.rag.openai.api_key = OLLAMA_API_KEY
    client.rag.llm_model = CHAT_MODEL
    client.rag.embedding_model = EMBED_MODEL

    COLLECTION = "ollama-knowledge"

    # 2. Setup Collection (Auto-detect dimension!)
    console.print(f"[bold blue]Initializing collection '{COLLECTION}' using '{EMBED_MODEL}'...[/bold blue]")
    try:
        # MAGIC: We don't need to know that nomic-embed-text is 768d!
        dim = client.rag.initialize_collection(COLLECTION, metric="cosine")
        console.print(f"[green]Collection ready with detected dimension: {dim}[/green]")
    except Exception as e:
        console.print(f"[red]Collection error: {e}[/red]")
        return

    # 3. Ingest Data locally using Ollama
    text_data = """
    Ollama allows you to run open-source large language models, such as Llama 3, locally.
    RustKissVDB is a lightweight vector database perfect for local RAG because it uses very little RAM.
    Combining Ollama + RustKissVDB creates a 100% private, local RAG stack without sending data to the cloud.

    The 'nomic-embed-text' model is a popular choice for embeddings in Ollama, typically outputting 768 dimensions.
    """

    console.print("[bold blue]Ingesting document...[/bold blue]")
    chunks = client.rag.ingest_text(collection=COLLECTION, text=text_data, metadata={"source": "ollama_demo"})
    console.print(f"[green]Successfully ingested {chunks} chunks![/green]")

    # 4. RAG Chat
    question = "¿Como se combina Ollama y RustKissVDB? (responde en español)"
    console.print(f"\n[bold purple]Question:[/bold purple] {question}")

    console.print(f"[dim]Thinking with {CHAT_MODEL}...[/dim]")
    response = client.rag.chat(collection=COLLECTION, query=question, k=2)

    console.print("\n[bold purple]Answer:[/bold purple]")
    console.print(Markdown(response.answer))

    console.print("\n[dim]Sources used:[/dim]")
    for src in response.sources:
        console.print(f"- {src.meta.get('text_snippet')} (Score: {src.score:.4f})")


if __name__ == "__main__":
    main()
