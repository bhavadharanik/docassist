"""
CHUNKER — Step 1 of RAG Pipeline
=================================
What it does: Loads a PDF and splits it into smaller text chunks.

Why chunk?
- LLMs have token limits (can't send entire document)
- Smaller chunks = more precise retrieval
- Overlap ensures important info at chunk boundaries isn't lost

Interview explanation:
"We chunk documents into ~500 character pieces with 100 char overlap.
This balances retrieval precision with context preservation.
Too small = loses context. Too large = retrieval becomes imprecise."
"""

from PyPDF2 import PdfReader


def load_pdf(file) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text
        chunk_size: Characters per chunk (500 is sweet spot)
        overlap: Characters of overlap between chunks (prevents info loss at boundaries)

    Returns:
        List of dicts: [{text, chunk_id, start_char, end_char}, ...]

    Why overlap?
        If an important sentence spans two chunks, overlap ensures
        both chunks contain it. Like a sliding window.
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_piece = text[start:end]

        # Skip empty chunks
        if chunk_text_piece.strip():
            chunks.append({
                "text": chunk_text_piece.strip(),
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": min(end, len(text)),
            })
            chunk_id += 1

        # Move forward by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

    return chunks
