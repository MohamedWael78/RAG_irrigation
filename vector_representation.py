"""
04_vector_representation.py – Step 4: Configure the embedding model.

Defines the embedding model used across the pipeline for vector representation.
Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) – a lightweight,
high-quality model that runs locally without any API key.
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a configured HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    model = get_embedding_model()
    embedding = model.embed_query("Drip irrigation delivers water directly to the root zone.")
    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
    print(f"   Vector dimension: {len(embedding)}")