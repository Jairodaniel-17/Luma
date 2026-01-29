import asyncio

from luma.client import LumaClient


async def main():
    try:
        # Con Ollama
        client = LumaClient(
            base_url="http://0.0.0.0:1234",
            api_key="dev",
            use_ollama=True,
            ollama_embedding_model="embeddinggemma:300m",
            ollama_llm_model="nemotron-3-nano:30b-cloud",
        )

        print("Cliente Luma inicializado con Ollama")

        # Crear colección automáticamente
        dimension = client.create_rag_collection("mi_coleccion")
        print(f"Colección creada con dimensión: {dimension}")

        # Ingresar documento
        text = """
        Este es un documento de prueba sobre inteligencia artificial.
        La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana
        por parte de máquinas, especialmente sistemas informáticos.

        Los principales tipos de IA incluyen:
        1. IA débil o estrecha: Diseñada para tareas específicas
        2. IA fuerte o general: Puede realizar cualquier tarea intelectual humana
        3. Superinteligencia: Excede la inteligencia humana en todos los aspectos

        Machine Learning es un subcampo de la IA que se centra en el desarrollo de
        algoritmos que permiten a las computadoras aprender de los datos.
        """

        chunks = client.ingest_document(
            collection="mi_coleccion", text=text, metadata={"author": "yo", "source": "documento", "tema": "IA"}
        )

        print(f"Documento ingresado en {chunks} chunks")

        # Hacer pregunta
        print("\nRealizando consulta RAG...")
        respuesta = client.ask(
            collection="mi_coleccion",
            question="¿Qué es la inteligencia artificial y qué tipos existen?",
            k=3,
            temperature=0.1,
        )

        print("\n" + "=" * 50)
        print("RESPUESTA:")
        print("=" * 50)
        print(respuesta.answer)
        print("\n" + "=" * 50)
        print(f"Fuentes encontradas: {len(respuesta.sources)}")

        for i, source in enumerate(respuesta.sources):
            print(f"\n--- Fuente {i + 1} ---")
            print(f"ID: {source.id}")
            print(f"Score: {source.score:.4f}")
            if source.meta:
                print(f"Metadatos: {source.meta.get('tema', 'N/A')}")
            if source.content:
                preview = source.content[:150] + "..." if len(source.content) > 150 else source.content
                print(f"Contenido: {preview}")

        if respuesta.usage:
            print(f"\n📊 Uso de tokens: {respuesta.usage}")

        # Hacer otra pregunta
        print("\n" + "=" * 50)
        print("Segunda consulta...")
        print("=" * 50)

        respuesta2 = client.ask(collection="mi_coleccion", question="¿Qué es Machine Learning?", k=2)

        print("\nRESPUESTA 2:")
        print(respuesta2.answer)
        print(f"\nFuentes: {len(respuesta2.sources)}")

        # Cerrar cliente
        client.close()
        print("\nPrueba completada exitosamente")

    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Para pruebas síncronas, puedes usar directamente:
    # main()
    asyncio.run(main())
