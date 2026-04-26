"""
EVALUATOR — RAG Pipeline Quality Metrics
=========================================
Measures how well the retrieval step is working.

Why evaluate RAG?
- Retrieval is the bottleneck — if wrong chunks are retrieved, the LLM can't help
- These metrics tell you if your chunking strategy and embedding model are working

Interview explanation:
"I built an evaluation module that measures retrieval precision and MRR.
Precision tells me what % of retrieved chunks are actually relevant.
MRR tells me how high the first relevant chunk ranks in results.
This helped me tune chunk_size, overlap, and top_k parameters."
"""

from chunker import load_pdf, chunk_text
from embedder import get_model, embed_chunks
from retriever import retrieve


def evaluate_retrieval(
    pdf_path: str,
    test_cases: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5,
    threshold: float = 0.1,
) -> dict:
    """
    Evaluate retrieval quality on test cases.

    Args:
        pdf_path: Path to the PDF file
        test_cases: List of {"query": str, "expected_keywords": [str]}
            expected_keywords = words that SHOULD appear in retrieved chunks

    Returns:
        dict with per-query and aggregate metrics
    """
    model = get_model()

    with open(pdf_path, "rb") as f:
        text = load_pdf(f)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    index, _ = embed_chunks(chunks, model)

    results = []
    for case in test_cases:
        query = case["query"]
        keywords = [kw.lower() for kw in case["expected_keywords"]]

        retrieved = retrieve(query, index, chunks, model, top_k=top_k, threshold=threshold)

        # Precision: what fraction of retrieved chunks contain expected keywords?
        relevant_count = 0
        first_relevant_rank = None
        for i, chunk in enumerate(retrieved):
            chunk_lower = chunk["text"].lower()
            if any(kw in chunk_lower for kw in keywords):
                relevant_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = i + 1

        precision = relevant_count / len(retrieved) if retrieved else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        results.append({
            "query": query,
            "chunks_retrieved": len(retrieved),
            "relevant_chunks": relevant_count,
            "precision": round(precision, 3),
            "mrr": round(mrr, 3),
            "top_scores": [round(c["score"], 3) for c in retrieved[:3]],
        })

    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    avg_mrr = sum(r["mrr"] for r in results) / len(results) if results else 0

    return {
        "per_query": results,
        "aggregate": {
            "avg_precision": round(avg_precision, 3),
            "avg_mrr": round(avg_mrr, 3),
            "total_queries": len(results),
        },
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "threshold": threshold,
        },
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <pdf_path>")
        print("\nExample test cases are built in for demo purposes.")
        sys.exit(1)

    # Demo test cases — replace with your own for your documents
    demo_cases = [
        {
            "query": "What is retrieval augmented generation?",
            "expected_keywords": ["retrieval", "augmented", "generation", "rag"],
        },
        {
            "query": "How does embedding work?",
            "expected_keywords": ["embedding", "vector", "encode", "representation"],
        },
        {
            "query": "What is a transformer?",
            "expected_keywords": ["transformer", "attention", "self-attention"],
        },
    ]

    results = evaluate_retrieval(sys.argv[1], demo_cases)
    print(json.dumps(results, indent=2))
