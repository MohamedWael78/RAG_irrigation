"""
05_create_chroma_store.py – Step 5: Create / update ChromaDB vector store.

Creates the initial store from chunks.json and also provides
add_documents_to_store() for incrementally adding image description
chunks at runtime (without rebuilding the entire store).
"""

import os
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

import importlib
embed_module = importlib.import_module("vector_representation")

CHUNKS_FILE = os.path.join("data", "chunks", "chunks.json")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "aquamind_irrigation"


def load_chunks_from_file() -> list[dict]:
    if not os.path.isfile(CHUNKS_FILE):
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run 01→02→03 first.")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in chunks
    ]


def create_chroma_store(
    documents: list[Document] | None = None,
    embedding_model=None,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    if embedding_model is None:
        embedding_model = embed_module.get_embedding_model()
    if documents is None:
        chunks = load_chunks_from_file()
        documents = chunks_to_documents(chunks)
    print(f"🔄 Creating ChromaDB with {len(documents)} documents ...")
    vs = Chroma.from_documents(
        documents=documents, embedding=embedding_model,
        persist_directory=persist_directory, collection_name=collection_name,
    )
    count = vs._collection.count()
    print(f"✅ ChromaDB created: {count} vectors in '{collection_name}'")
    return vs


def load_existing_chroma_store(
    embedding_model=None, persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma | None:
    if not os.path.isdir(persist_directory):
        return None
    if embedding_model is None:
        embedding_model = embed_module.get_embedding_model()
    try:
        vs = Chroma(
            collection_name=collection_name, embedding_function=embedding_model,
            persist_directory=persist_directory,
        )
        count = vs._collection.count()
        if count == 0:
            return None
        print(f"✅ Loaded ChromaDB: {count} vectors")
        return vs
    except Exception as e:
        print(f"⚠️  Load failed: {e}")
        return None


def add_documents_to_store(
    documents: list[Document],
    embedding_model=None,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Add additional documents (e.g., image descriptions) to existing store.

    Used by streamlit_app.py to add vision-model-described images
    without rebuilding the entire vector store.
    """
    if embedding_model is None:
        embedding_model = embed_module.get_embedding_model()

    vs = load_existing_chroma_store(
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if vs is None:
        # Store doesn't exist yet – create it with these documents
        return create_chroma_store(
            documents=documents, embedding_model=embedding_model,
            persist_directory=persist_directory, collection_name=collection_name,
        )

    # Add to existing store
    vs.add_documents(documents)
    count = vs._collection.count()
    print(f"✅ Added {len(documents)} documents. Total: {count}")
    return vs


if __name__ == "__main__":
    if not os.path.isfile(CHUNKS_FILE):
        print("⚠️  Running pipeline 01→02→03 ...")
        m01 = importlib.import_module("documents")
        m02 = importlib.import_module("preprocessing")
        m03 = importlib.import_module("chunking")
        m01.create_documents()
        m02.preprocess_all_documents()
        m03.chunk_all_documents()
    vs = create_chroma_store()
    print(f"\n🎉 ChromaDB ready at {CHROMA_DIR}/")