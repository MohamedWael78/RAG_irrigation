"""
06_retrieve_context.py – Step 6: Hybrid retrieval with BM25 + Vector + RRF + FlashRank reranking.

Full pipeline:
  1. BM25 keyword search (rank_bm25) → top candidates
  2. ChromaDB vector similarity search → top candidates
  3. Reciprocal Rank Fusion (RRF) merges both result sets
  4. FlashRank cross-encoder reranking (ms-marco-MiniLM-L-12-v2) → final top-k

Graceful fallbacks: if BM25 or FlashRank unavailable, pure vector search is used.

The main search function hybrid_search_with_reranking() is called by the
search_knowledge_base tool in 07_prompting.py.
"""

import os
import re
import importlib

from langchain_core.documents import Document

embed_module = importlib.import_module("vector_representation")
store_module = importlib.import_module("create_chroma_store")
chunk_module = importlib.import_module("chunking")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "aquamind_irrigation"
DEFAULT_K = 5
RRF_K = 60  # RRF constant from literature (Cormack et al.)

# ── Optional dependencies with graceful fallback ──────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not installed. BM25 keyword search disabled. "
          "Install with: pip install rank-bm25")

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️  flashrank not installed. Cross-encoder reranking disabled. "
          "Install with: pip install flashrank")

# ── Module-level caches (persist across Streamlit reruns) ──────────────────
_cached_bm25 = None
_cached_chunks = None
_cached_reranker = None
_cached_vector_store = None


# ═══════════════════════════════════════════════════════════════════════════
# BM25 UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 indexing and querying.

    Lowercases, removes punctuation (keeps numbers for irrigation data like
    '0.27', '1.15', '120-200'), splits on whitespace, and filters tokens
    shorter than 2 characters.
    """
    text = text.lower()
    # Keep numbers and decimal points; remove other punctuation
    text = re.sub(r"[^\w\s.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def _build_bm25_index():
    """Build or retrieve cached BM25 index from chunks.

    Returns (BM25Okapi_index, chunks_list).
    """
    global _cached_bm25, _cached_chunks

    if _cached_bm25 is not None and _cached_chunks is not None:
        return _cached_bm25, _cached_chunks

    chunks = chunk_module.load_chunks()
    if not chunks:
        # Auto-build pipeline if chunks don't exist
        print("⚠️  No chunks found. Running full pipeline ...")
        m01 = importlib.import_module("01_documents")
        m02 = importlib.import_module("02_preprocessing")
        m01.create_documents()
        m02.preprocess_all_documents()
        chunk_module.chunk_all_documents()
        chunks = chunk_module.load_chunks()

    _cached_chunks = chunks

    if BM25_AVAILABLE and chunks:
        tokenized_corpus = [_tokenize(c["content"]) for c in chunks]
        _cached_bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index built: {len(chunks)} documents indexed")
    else:
        _cached_bm25 = None

    return _cached_bm25, _cached_chunks


# ═══════════════════════════════════════════════════════════════════════════
# FLASHRANK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _load_reranker():
    """Load or retrieve cached FlashRank cross-encoder reranker.

    Uses ms-marco-MiniLM-L-12-v2 – a fast, lightweight model (12M params)
    that runs locally without any API key. Model weights are cached in
    .flashrank_cache/ after first download.
    """
    global _cached_reranker

    if _cached_reranker is not None:
        return _cached_reranker

    if not FLASHRANK_AVAILABLE:
        return None

    try:
        _cached_reranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=".flashrank_cache",
        )
        print("✅ FlashRank reranker loaded: ms-marco-MiniLM-L-12-v2")
    except Exception as e:
        print(f"⚠️  FlashRank model load failed: {e}")
        _cached_reranker = None

    return _cached_reranker


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════

def get_vector_store():
    """Get or create the ChromaDB vector store. Cached at module level."""
    global _cached_vector_store

    if _cached_vector_store is not None:
        try:
            count = _cached_vector_store._collection.count()
            if count > 0:
                return _cached_vector_store
        except Exception:
            _cached_vector_store = None

    embedding_model = embed_module.get_embedding_model()
    vs = store_module.load_existing_chroma_store(
        embedding_model=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    if vs is None:
        print("⚠️  ChromaDB not found. Building from pipeline ...")
        vs = store_module.create_chroma_store(embedding_model=embedding_model)

    _cached_vector_store = vs
    return vs


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SEARCH METHODS
# ═══════════════════════════════════════════════════════════════════════════

def _bm25_search(query: str, k: int = 15) -> list[tuple]:
    """BM25 keyword search over chunk corpus.

    Returns list of (rank, chunk_dict, bm25_score) tuples,
    sorted by BM25 score descending.
    """
    bm25, chunks = _build_bm25_index()

    if bm25 is None or not chunks:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append((rank, chunks[idx], float(scores[idx])))

    return results


def _vector_search(query: str, k: int = 15) -> list[tuple]:
    """ChromaDB vector similarity search with distance scores.

    Returns list of (rank, Document, distance) tuples.
    Lower distance = higher relevance.
    """
    vs = get_vector_store()
    docs_with_scores = vs.similarity_search_with_score(query, k=k)

    results = []
    for rank, (doc, dist) in enumerate(docs_with_scores):
        results.append((rank, doc, float(dist)))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION (RRF)
# ═══════════════════════════════════════════════════════════════════════════

def _rrf_fuse(
    bm25_results: list[tuple],
    vector_results: list[tuple],
    k: int = RRF_K,
) -> list[dict]:
    """Merge BM25 and vector results using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ  1 / (k + rank_i(d))
    across all input result lists. Documents appearing in both lists
    receive additive scores, naturally boosting consensus results.

    Args:
        bm25_results:  list of (rank, chunk_dict, bm25_score)
        vector_results: list of (rank, Document, distance)
        k:              RRF constant (default 60 from Cormack et al.)

    Returns list of dicts sorted by fused RRF score descending:
        {"content": str, "metadata": dict, "rrf_score": float,
         "bm25_rank": int|None, "vector_rank": int|None}
    """
    fused = {}

    # ── BM25 contributions ──
    for rank, chunk_dict, bm25_score in bm25_results:
        content = chunk_dict["content"]
        key = content[:200]  # dedup key (first 200 chars)

        rrf_contribution = 1.0 / (k + rank + 1)

        if key not in fused:
            fused[key] = {
                "content": content,
                "metadata": chunk_dict.get("metadata", {}),
                "rrf_score": 0.0,
                "bm25_rank": rank + 1,
                "vector_rank": None,
                "bm25_score": bm25_score,
                "vector_distance": None,
            }
        fused[key]["rrf_score"] += rrf_contribution

    # ── Vector contributions ──
    for rank, doc, dist in vector_results:
        content = doc.page_content
        key = content[:200]

        rrf_contribution = 1.0 / (k + rank + 1)

        if key not in fused:
            fused[key] = {
                "content": content,
                "metadata": doc.metadata if isinstance(doc.metadata, dict) else {},
                "rrf_score": 0.0,
                "bm25_rank": None,
                "vector_rank": rank + 1,
                "bm25_score": None,
                "vector_distance": dist,
            }
        else:
            fused[key]["vector_rank"] = rank + 1
            fused[key]["vector_distance"] = dist
        fused[key]["rrf_score"] += rrf_contribution

    # ── Sort by fused score ──
    sorted_results = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return sorted_results


# ═══════════════════════════════════════════════════════════════════════════
# FLASHRANK CROSS-ENCODER RERANKING
# ═══════════════════════════════════════════════════════════════════════════

def _rerank(query: str, candidates: list[dict], k: int = DEFAULT_K) -> list[Document]:
    """Rerank RRF-fused candidates using FlashRank cross-encoder.

    The cross-encoder jointly encodes (query, passage) pairs and produces
    a relevance score that is more accurate than BM25 or vector similarity
    alone, especially for:
      - Distinguishing relevant vs irrelevant passages with shared keywords
      - Understanding semantic nuance that bi-encoders miss
      - Handling domain-specific terminology (irrigation, Kc, ET0)

    If FlashRank is unavailable, falls back to top-k RRF results.

    Args:
        query:      user query string
        candidates: RRF-fused candidate list (sorted by rrf_score)
        k:          number of final results to return

    Returns list of LangChain Document objects with metadata.
    """
    reranker = _load_reranker()

    if reranker is None or not candidates:
        # ── Fallback: return top-k RRF results ──
        top_candidates = candidates[:k]
        return [
            Document(
                page_content=c["content"],
                metadata=c.get("metadata", {}),
            )
            for c in top_candidates
        ]

    # ── Prepare passages for FlashRank ──
    # Rerank more candidates than final k (standard practice: 3x over-retrieval)
    rerank_pool = candidates[:min(k * 3, len(candidates))]

    passages = [
        {
            "id": i,
            "text": c["content"],
            "meta": c.get("metadata", {}),
        }
        for i, c in enumerate(rerank_pool)
    ]

    try:
        rerank_request = RerankRequest(query=query, passages=passages)
        results = reranker.rerank(rerank_request)

        # results is sorted by cross-encoder relevance score
        final_docs = []
        for r in results[:k]:
            meta = r.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            # Ensure required metadata fields exist
            if "source" not in meta:
                meta["source"] = "unknown"
            if "page" not in meta:
                meta["page"] = "?"
            final_docs.append(
                Document(page_content=r["text"], metadata=meta)
            )

        return final_docs

    except Exception as e:
        print(f"⚠️  FlashRank reranking failed: {e}. Using RRF results.")
        top_candidates = candidates[:k]
        return [
            Document(
                page_content=c["content"],
                metadata=c.get("metadata", {}),
            )
            for c in top_candidates
        ]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SEARCH FUNCTION (used by search_knowledge_base tool)
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_search_with_reranking(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Full hybrid retrieval pipeline: BM25 + Vector → RRF → FlashRank reranking.

    Step 1: BM25 keyword search retrieves top (k×3) keyword matches
    Step 2: ChromaDB vector search retrieves top (k×3) semantic matches
    Step 3: Reciprocal Rank Fusion merges both result sets, boosting
            documents that appear in both (consensus = higher confidence)
    Step 4: FlashRank cross-encoder reranks fused candidates for
            final relevance, returning top-k Documents

    Fallback cascade:
      - No BM25 → pure vector + reranking
      - No FlashRank → RRF-fused top-k (still better than pure vector)
      - No BM25 + no FlashRank → pure vector similarity search

    Args:
        query: user query string
        k:     number of final documents to return

    Returns list of LangChain Document objects with metadata
    (source, page, chunk_id) for citation rendering.
    """
    overretrieve = k * 3  # retrieve more for fusion + reranking pool

    # ── Step 1: BM25 keyword search ──
    bm25_results = _bm25_search(query, k=overretrieve)

    # ── Step 2: Vector similarity search ──
    vector_results = _vector_search(query, k=overretrieve)

    # ── Step 3: RRF fusion ──
    if bm25_results and vector_results:
        # Full hybrid: both result sets available
        fused = _rrf_fuse(bm25_results, vector_results)
    elif vector_results:
        # BM25 unavailable: use vector results directly
        fused = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata if isinstance(doc.metadata, dict) else {},
                "rrf_score": 1.0 / (RRF_K + rank + 1),
            }
            for rank, doc, dist in vector_results
        ]
    elif bm25_results:
        # Vector unavailable: use BM25 results directly
        fused = [
            {
                "content": chunk_dict["content"],
                "metadata": chunk_dict.get("metadata", {}),
                "rrf_score": 1.0 / (RRF_K + rank + 1),
            }
            for rank, chunk_dict, score in bm25_results
        ]
    else:
        # Both unavailable: return empty
        return []

    # ── Step 4: FlashRank reranking ──
    final_docs = _rerank(query, fused, k=k)

    return final_docs


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLER SEARCH FUNCTIONS (fallbacks / alternatives)
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Pure vector similarity search (no hybrid, no reranking)."""
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)


def retrieve_context_mmr(query: str, k: int = DEFAULT_K, lambda_mult: float = 0.5) -> list[Document]:
    """MMR diversity search (vector only, no BM25, no reranking)."""
    vs = get_vector_store()
    return vs.max_marginal_relevance_search(query, k=k, lambda_mult=lambda_mult)


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_advanced_retriever(k: int = DEFAULT_K):
    """Return a callable that performs full hybrid search + reranking.

    Used by the search_knowledge_base tool in 07_prompting.py.
    Returns a function (not a LangChain Retriever) because the tool
    directly calls search and formats results with citation markers.
    """
    def search_fn(query: str) -> list[Document]:
        return hybrid_search_with_reranking(query, k=k)
    return search_fn


def setup_retriever(k: int = DEFAULT_K):
    """Return a LangChain-compatible MMR retriever (for simple use cases)."""
    vs = get_vector_store()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CITATION FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_retrieved_docs(docs: list[Document]) -> str:
    """Format retrieved documents with [n] citation markers.

    For PDF-derived documents, citations include source and page number:
      [1] (source: fao_manual.txt, page: 5): content ...

    This enables the LLM to cite sources and the UI to render citation cards.
    """
    if not docs:
        return "No relevant documents found in the knowledge base."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] (source: {source}, page: {page}): {doc.page_content}"
        )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID SEARCH + RERANKING TEST")
    print("=" * 60)

    queries = [
        "What is the recommended emitter spacing for sandy soil?",
        "Kc value for tomato at mid-season",
        "How to troubleshoot a Hunter PGV valve that won't open?",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")

        # Hybrid + reranked
        docs_hybrid = hybrid_search_with_reranking(q, k=3)
        print(f"   Hybrid + Reranked: {len(docs_hybrid)} docs")
        for d in docs_hybrid:
            src = d.metadata.get("source", "?")
            pg = d.metadata.get("page", "?")
            print(f"     → {src} p.{pg}: {d.page_content[:80]}...")

        # Pure vector (comparison)
        docs_vec = retrieve_context(q, k=3)
        print(f"   Pure Vector: {len(docs_vec)} docs")