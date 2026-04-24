"""
RETRIEVER — Step 3 of RAG Pipeline
"""

from embedder import embed_query


def retrieve(query: str, index, chunks: list[dict], model, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    query_embedding = embed_query(query, model)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < threshold:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "chunk_id": chunks[idx]["chunk_id"],
            "score": float(score),
        })
    return results
