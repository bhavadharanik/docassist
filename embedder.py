"""
EMBEDDER — Step 2 of RAG Pipeline
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cache")


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_PATH, device="cpu", local_files_only=True)


def embed_chunks(chunks: list[dict], model) -> tuple:
    import faiss
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings


def embed_query(query: str, model) -> np.ndarray:
    import faiss
    embedding = model.encode([query], show_progress_bar=False)
    embedding = np.array(embedding, dtype="float32")
    faiss.normalize_L2(embedding)
    return embedding
