"""
03_chunking.py – Step 3: Chunk processed documents for RAG.

Reads cleaned text from data/processed/, splits each document into overlapping
chunks using RecursiveCharacterTextSplitter. For PDF-derived documents with
--- PAGE N --- markers, tracks page numbers in chunk metadata for citation.

Saves chunks as JSON to data/chunks/chunks.json.

Run as:  python 03_chunking.py
"""

import os
import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DIR = os.path.join("data", "processed")
CHUNKS_DIR = os.path.join("data", "chunks")
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.json")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Chunking parameters – tuned for technical / table-heavy irrigation content
CHUNK_SIZE = 650
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Page-aware text splitting
# ---------------------------------------------------------------------------

def split_by_pages(text: str) -> list[tuple[int, str]]:
    """Split text into per-page sections using --- PAGE N --- markers.

    Returns list of (page_number, page_text) tuples. If no page markers
    are found, returns [(0, full_text)].
    """
    pages = re.split(r"--- PAGE (\d+) ---", text)
    if len(pages) < 3:  # no markers found (pattern: [pre, num, text, num, text, ...])
        return [(0, text)]

    result = []
    # pages[0] is text before first marker (often empty), then alternating num/text
    for i in range(1, len(pages) - 1, 2):
        page_num = int(pages[i])
        page_text = pages[i + 1].strip()
        if page_text:
            result.append((page_num, page_text))
    return result


def chunk_document(filepath: str, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Read a processed document, split it into chunks with page metadata.

    For PDF-derived documents with page markers, chunks inherit the page number
    of the page they primarily come from. For plain-text documents without
    markers, page is set to 0.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.basename(filepath)

    # Check if document has page markers
    has_pages = "--- PAGE" in text
    all_chunks = []
    
    chunk_id = 0
    if has_pages:
        # Split by pages first, then chunk within each page
        page_sections = split_by_pages(text)

        for page_num, page_text in page_sections:
            chunks = splitter.create_documents(
                texts=[page_text],
                metadatas=[{"source": source_name, "page": page_num}],
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk.page_content,
                    "metadata": {
                        "source": source_name,
                        "page": page_num,
                        "chunk_id": i,
                    },
                })
                chunk_id += 1
    else:
        # No page markers – chunk the whole document, page = 0
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source_name, "page": 0}],
        )
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk.page_content,
                "metadata": {
                    "source": source_name,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_id": i,
                },
            })
            chunk_id += 1

    print(f"✅ Chunked: {source_name} → {len(all_chunks)} chunks "
          f"(page-aware: {has_pages})")
    return all_chunks


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_all_documents() -> list[dict]:
    """Chunk every processed document and return all chunks combined."""
    processed_files = sorted(
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".txt")
    )
    if not processed_files:
        print("⚠️  No processed documents found. Run 02_preprocessing.py first.")
        return []

    splitter = get_text_splitter()
    all_chunks = []
    for filepath in processed_files:
        chunks = chunk_document(filepath, splitter)
        all_chunks.extend(chunks)

    # Save to JSON
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Total chunks: {len(all_chunks)} saved to {CHUNKS_FILE}")
    return all_chunks


def load_chunks() -> list[dict]:
    """Load previously saved chunks from the JSON file."""
    if not os.path.isfile(CHUNKS_FILE):
        print("⚠️  chunks.json not found. Run 03_chunking.py first.")
        return []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = chunk_all_documents()
    print(f"\n📝 {len(chunks)} chunks ready for embedding")