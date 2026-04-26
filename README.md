# DocAssist — RAG-Powered Document Q&A

Upload a PDF, ask questions, get grounded answers with source citations.

Built to demonstrate a complete **Retrieval Augmented Generation (RAG)** pipeline — from document ingestion to cited answer generation.

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌───────────────────┐     ┌───────────┐
│  PDF Upload  │────▶│  Chunker │────▶│  Embedder (SBERT) │────▶│   FAISS   │
│   (PyPDF2)   │     │ 500 char │     │ all-MiniLM-L6-v2  │     │  Index    │
└─────────────┘     │ 100 over │     └───────────────────┘     └─────┬─────┘
                    └──────────┘                                     │
                                                                     │
┌─────────────┐     ┌───────────────┐     ┌──────────────┐          │
│   Answer +   │◀───│  Generator    │◀───│  Retriever    │◀─────────┘
│  Citations   │     │ Gemini 2.0   │     │  Top-K + threshold      │
└─────────────┘     └───────────────┘     └──────────────┘
                                                ▲
                                                │
                                          ┌─────┴─────┐
                                          │   Query    │
                                          │  Embedding │
                                          └───────────┘
```

## Pipeline Steps

| Step | Module | What it does | Technology |
|------|--------|-------------|------------|
| 1 | `chunker.py` | Extracts PDF text, splits into overlapping chunks | PyPDF2, sliding window (500 chars, 100 overlap) |
| 2 | `embedder.py` | Converts text chunks to dense vectors | sentence-transformers (all-MiniLM-L6-v2) |
| 3 | `retriever.py` | Finds most similar chunks to query | FAISS cosine similarity, top-K with threshold |
| 4 | `generator.py` | Generates answer grounded in retrieved context | Google Gemini 2.0 Flash |
| 5 | `evaluator.py` | Measures retrieval quality (precision, MRR) | Custom evaluation framework |
| UI | `app.py` | Interactive chat interface | Streamlit |

## Key Design Decisions

- **Chunking strategy**: 500-char chunks with 100-char overlap. Overlap prevents information loss at chunk boundaries — like a sliding window.
- **Embedding model**: `all-MiniLM-L6-v2` — lightweight (80MB), fast on CPU, good semantic similarity for English text.
- **FAISS IndexFlatIP**: Inner product on L2-normalized vectors = cosine similarity. Exact search, no approximation — suitable for document-scale (not web-scale).
- **Relevance threshold**: Filters out low-similarity chunks before sending to LLM. Prevents hallucination from irrelevant context.
- **Citation enforcement**: System prompt requires `[Chunk X]` references, making answers verifiable and traceable.

## Setup

```bash
# Clone
git clone https://github.com/bhavadharanik/docassist.git
cd docassist

# Virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download embedding model (one-time, ~80MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='model_cache')"

# Set up API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

## Evaluation

Test retrieval quality with the evaluation module:

```bash
python evaluator.py your_document.pdf
```

Returns precision (% of retrieved chunks that are relevant) and MRR (how high the first relevant chunk ranks).

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Generation**: Google Gemini 2.0 Flash
- **PDF Parsing**: PyPDF2
- **UI**: Streamlit
- **Language**: Python
