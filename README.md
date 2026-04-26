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
│  Citations   │     │ Ollama   │     │  Top-K + threshold      │
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
| 4 | `generator.py` | Generates answer grounded in retrieved context | Ollama (gemma3:4b, local) |
| 5 | `evaluator.py` | Evaluates retrieval + generation quality | Precision, Recall, F1, MRR, NDCG, LLM-as-Judge |
| UI | `app.py` | Chat + Evaluation dashboard | Streamlit (tabbed layout) |

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

# Install and start Ollama (https://ollama.ai)
ollama pull gemma3:4b

# Run
streamlit run app.py
```

## Evaluation

The evaluation module measures both **retrieval quality** and **generation quality**.

### Retrieval Metrics
| Metric | What it measures | Analogy |
|--------|-----------------|---------|
| **Precision@K** | % of retrieved chunks that are relevant | Grep results — how many are useful? |
| **Recall** | % of all relevant chunks that were found | Did your search miss any important files? |
| **F1 Score** | Harmonic mean of precision and recall | Single number balancing both |
| **MRR** | How high the first relevant result ranks | Google search — is the answer on page 1? |
| **NDCG** | Are relevant results ranked at the top? | Penalizes good results at bad positions |

### Generation Metrics (LLM-as-Judge)
| Metric | What it measures | Score |
|--------|-----------------|-------|
| **Faithfulness** | Is the answer grounded in retrieved context? | 1-5 (5 = no hallucination) |
| **Answer Relevance** | Does the answer address the question? | 1-5 (5 = directly answers) |

### Usage

```bash
# Retrieval evaluation only
python evaluator.py your_document.pdf

# Full evaluation (retrieval + generation with LLM-as-judge)
python evaluator.py your_document.pdf --full

# Parameter sweep (find optimal chunk_size, overlap, top_k)
python evaluator.py your_document.pdf --sweep
```

### Parameter Sweep

Compares 24+ configurations of chunk_size × overlap × top_k, ranked by F1 score. Helps find the optimal RAG setup for your specific documents.

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Generation**: Ollama (gemma3:4b, local)
- **PDF Parsing**: PyPDF2
- **UI**: Streamlit
- **Language**: Python
