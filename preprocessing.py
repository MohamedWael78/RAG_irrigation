"""
02_preprocessing.py – Step 2: Preprocess raw documents.

Reads text files from data/raw/ (extracted from PDFs or sample documents),
applies cleaning specific to PDF-extracted text (fix broken lines, remove
page footers, rejoin hyphenated words), and saves to data/processed/.

Preserves --- PAGE N --- markers for downstream chunking with page metadata.

Run as:  python 02_preprocessing.py
"""

import os
import re

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """Apply cleaning transformations optimized for PDF-extracted text.

    Key PDF artifacts handled:
      - Broken lines within a paragraph (mid-word line breaks)
      - Hyphenated word splits across lines
      - Repeated page numbers/footers
      - Excessive whitespace
      - Unicode normalization
    Preserves --- PAGE N --- markers for page-aware chunking.
    """
    # 1. Normalize Unicode dashes and bullets
    text = text.replace("\u2013", "-")   # en-dash → hyphen
    text = text.replace("\u2014", "-")   # em-dash → hyphen
    text = text.replace("\u2022", "*")   # bullet → asterisk
    text = text.replace("\u2019", "'")   # right single quote
    text = text.replace("\u2018", "'")   # left single quote

    # 2. Remove common PDF page footers (standalone page numbers like "42" or "Page 42")
    #    but NOT our --- PAGE N --- markers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Page\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # 3. Rejoin hyphenated words split across lines:
    #    "irri-\n  gation" → "irrigation"
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)

    # 4. Fix broken lines within paragraphs:
    #    A line that starts with lowercase after a line ending without period
    #    is likely a mid-paragraph break from PDF extraction.
    text = re.sub(
        r"([a-z,;:])\n\s+([a-z])",
        r"\1 \2",
        text,
    )

    # 5. Normalize horizontal whitespace (tabs, multiple spaces → single space)
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Remove trailing whitespace per line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]

    # 7. Remove purely decorative separator lines (---, ===, ***)
    #    but NOT our page markers (--- PAGE N ---)
    cleaned_lines = []
    for line in lines:
        if re.match(r"^[=\-*]{3,}$", line.strip()) and "PAGE" not in line:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 8. Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 9. Remove any double-blank lines right before a page marker
    text = re.sub(r"\n{2,}(--- PAGE)", r"\n\n--- PAGE", text)

    # 10. Strip leading/trailing whitespace
    text = text.strip()

    # 11. Final cleanup: re-collapse any excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def preprocess_file(raw_path: str, processed_path: str) -> str:
    """Read a raw document, clean it, and save the processed version."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_pdf_text(raw_text)

    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✅ Preprocessed: {os.path.basename(raw_path)} → {os.path.basename(processed_path)}  "
          f"({len(raw_text)} → {len(cleaned)} chars)")
    return processed_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess_all_documents() -> list[str]:
    """Preprocess every raw .txt document and return processed file paths."""
    raw_files = sorted(
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".txt")
    )
    if not raw_files:
        print("⚠️  No raw documents found. Run 01_documents.py first.")
        return []

    processed_paths = []
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        processed_path = os.path.join(PROCESSED_DIR, filename)
        preprocess_file(raw_path, processed_path)
        processed_paths.append(processed_path)

    return processed_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    paths = preprocess_all_documents()
    print(f"\n📝 {len(paths)} processed documents saved in {PROCESSED_DIR}/")
    for p in paths:
        print(f"   → {p}")