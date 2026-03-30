from langchain_chroma import Chroma
from langchain_core.documents import Document


def init_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_function,
) -> Chroma:
    """Create or load a Chroma vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def index_documents(vector_store: Chroma, documents: list[Document]) -> None:
    """Add documents to the vector store."""
    print(f"Indexing {len(documents)} document chunks...")
    vector_store.add_documents(documents)
    print("Chroma index built and persisted.")
