"""
EVALUATOR — RAG Pipeline Evaluation
=====================================
Think of this like monitoring for your ML system.
In microservices, you monitor latency, error rates, throughput.
In RAG, you monitor retrieval quality and generation faithfulness.

Three levels of evaluation:
1. RETRIEVAL — Did we fetch the right chunks? (Precision, Recall, MRR, NDCG)
2. GENERATION — Is the answer faithful to the context? Does it answer the question?
3. PARAMETER TUNING — Which chunk_size / overlap / top_k works best?

Key concept: LLM-as-Judge
    Using an LLM to evaluate another LLM's output.
    Like code review — the reviewer (judge LLM) checks the author's (generator LLM) work.
    Not perfect, but scalable. Human eval is better but expensive.
"""

import json
import math
from chunker import load_pdf, chunk_text
from embedder import get_model, embed_chunks
from retriever import retrieve


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================
# These measure: "Did we find the right chunks?"
# Analogy: You're searching a codebase. Precision = what % of search results
# were actually relevant. Recall = what % of all relevant files did you find.
# =============================================================================

def precision_at_k(retrieved_chunks: list[dict], expected_keywords: list[str]) -> float:
    """
    Precision@K: What fraction of retrieved chunks are relevant?

    High precision = few irrelevant results (low noise)
    Low precision = lots of junk mixed in

    Analogy: You grep for "database" and get 5 results.
    3 are about databases, 2 are about "database" in a comment.
    Precision = 3/5 = 0.6
    """
    if not retrieved_chunks:
        return 0.0
    keywords = [kw.lower() for kw in expected_keywords]
    relevant = sum(
        1 for chunk in retrieved_chunks
        if any(kw in chunk["text"].lower() for kw in keywords)
    )
    return relevant / len(retrieved_chunks)


def recall(retrieved_chunks: list[dict], all_chunks: list[dict], expected_keywords: list[str]) -> float:
    """
    Recall: What fraction of ALL relevant chunks did we retrieve?

    High recall = we found most relevant info (few misses)
    Low recall = we missed important chunks

    Analogy: There are 10 files with the bug. Your search found 7.
    Recall = 7/10 = 0.7. You missed 3 files.

    Precision vs Recall trade-off:
    - High top_k → better recall (find more), worse precision (more noise)
    - Low top_k → better precision (less noise), worse recall (miss things)
    - This is the same trade-off as sensitivity vs specificity in medical tests
    """
    keywords = [kw.lower() for kw in expected_keywords]

    # Total relevant chunks in entire document
    total_relevant = sum(
        1 for chunk in all_chunks
        if any(kw in chunk["text"].lower() for kw in keywords)
    )
    if total_relevant == 0:
        return 0.0

    # Relevant chunks we actually retrieved
    retrieved_relevant = sum(
        1 for chunk in retrieved_chunks
        if any(kw in chunk["text"].lower() for kw in keywords)
    )
    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(retrieved_chunks: list[dict], expected_keywords: list[str]) -> float:
    """
    MRR: How high does the first relevant result rank?

    MRR = 1/rank_of_first_relevant_result
    - First result relevant → MRR = 1.0
    - Second result relevant → MRR = 0.5
    - Third result relevant → MRR = 0.33

    Analogy: Google search. If the answer is the first link, great (MRR=1).
    If you have to scroll to result #5, that's bad (MRR=0.2).
    """
    keywords = [kw.lower() for kw in expected_keywords]
    for i, chunk in enumerate(retrieved_chunks):
        if any(kw in chunk["text"].lower() for kw in keywords):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_chunks: list[dict], expected_keywords: list[str]) -> float:
    """
    NDCG (Normalized Discounted Cumulative Gain): Are relevant results ranked higher?

    Unlike precision (binary: relevant or not), NDCG cares about ORDER.
    A relevant chunk at position 1 is worth more than at position 5.

    The "discounted" part: relevance is divided by log2(rank+1).
    Position 1: relevance / log2(2) = relevance / 1.0
    Position 2: relevance / log2(3) = relevance / 1.58
    Position 5: relevance / log2(6) = relevance / 2.58

    Analogy: In a ranked search, having the right answer first matters more
    than having it buried on page 2. NDCG penalizes good results at bad ranks.

    NDCG = DCG / ideal_DCG (normalized to 0-1 range)
    """
    keywords = [kw.lower() for kw in expected_keywords]

    # Binary relevance for each retrieved chunk
    relevances = [
        1.0 if any(kw in chunk["text"].lower() for kw in keywords) else 0.0
        for chunk in retrieved_chunks
    ]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    # Ideal DCG: all relevant results at the top
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# =============================================================================
# GENERATION EVALUATION (LLM-as-Judge)
# =============================================================================
# These measure: "Is the generated answer good?"
# Uses Gemini as a judge to score the generator's output.
#
# Why LLM-as-Judge?
#   - Human evaluation is the gold standard but doesn't scale
#   - LLM judges correlate ~80% with human judgments (good enough for dev)
#   - Same pattern used by RAGAS, DeepEval, and other eval frameworks
#
# Limitations:
#   - Judge LLM has its own biases (prefers verbose answers, etc.)
#   - Can't catch subtle factual errors it doesn't know about
#   - Best practice: use a stronger model as judge than as generator
# =============================================================================

def _call_ollama(prompt: str) -> str:
    """Helper to call Ollama for LLM-as-judge evaluations."""
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def evaluate_faithfulness(query: str, context: str, answer: str) -> dict:
    """
    Faithfulness: Is the answer grounded in the retrieved context?

    Score 1-5:
    5 = Every claim in the answer is supported by the context
    1 = The answer makes claims not found in the context (hallucination)

    This is the most critical RAG metric. The whole point of RAG is to
    ground LLM outputs in real data. If faithfulness is low, your RAG
    is not working — the LLM is hallucinating despite having context.

    Analogy: Code review. Is every line of code justified by the requirements?
    Or did the developer add features nobody asked for (hallucination)?
    """
    prompt = f"""You are evaluating a RAG system's answer for faithfulness.

Faithfulness = Is the answer supported by the provided context?

CONTEXT (retrieved chunks):
{context}

QUESTION: {query}

ANSWER: {answer}

Score the answer from 1-5:
5 = Every claim is directly supported by the context
4 = Most claims supported, minor unsupported details
3 = Mix of supported and unsupported claims
2 = Significant claims not in context
1 = Answer mostly makes things up / ignores context

Respond in this exact JSON format only, no other text:
{{"score": <1-5>, "reasoning": "<one sentence explanation>"}}"""

    try:
        text = _call_ollama(prompt).strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        return {"score": 0, "reasoning": str(e), "error": True}


def evaluate_relevance(query: str, answer: str) -> dict:
    """
    Answer Relevance: Does the answer actually address the question?

    Score 1-5:
    5 = Directly and completely answers the question
    1 = Answer is about something completely different

    A RAG system can be faithful (grounded in context) but irrelevant
    (retrieved wrong chunks, so the answer is about the wrong topic).

    Analogy: You ask "how do I deploy to prod?" and the system answers
    with perfectly accurate info about... database migrations. Faithful
    to the context, but not what you asked.
    """
    prompt = f"""You are evaluating if an answer is relevant to the question asked.

QUESTION: {query}

ANSWER: {answer}

Score from 1-5:
5 = Directly and completely answers the question
4 = Mostly answers but misses minor aspects
3 = Partially relevant, addresses some aspects
2 = Tangentially related but doesn't answer the question
1 = Completely irrelevant to the question

Respond in this exact JSON format only, no other text:
{{"score": <1-5>, "reasoning": "<one sentence explanation>"}}"""

    try:
        text = _call_ollama(prompt).strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        return {"score": 0, "reasoning": str(e), "error": True}


# =============================================================================
# FULL EVALUATION PIPELINE
# =============================================================================

def evaluate_retrieval(
    pdf_path: str,
    test_cases: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5,
    threshold: float = 0.1,
) -> dict:
    """
    Evaluate retrieval quality across all metrics.

    test_cases format:
    [
        {
            "query": "What is attention?",
            "expected_keywords": ["attention", "self-attention", "transformer"]
        }
    ]
    """
    model = get_model()

    with open(pdf_path, "rb") as f:
        text = load_pdf(f)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    index, _ = embed_chunks(chunks, model)

    results = []
    for case in test_cases:
        query = case["query"]
        keywords = case["expected_keywords"]

        retrieved = retrieve(query, index, chunks, model, top_k=top_k, threshold=threshold)

        p = precision_at_k(retrieved, keywords)
        r = recall(retrieved, chunks, keywords)
        mrr = mean_reciprocal_rank(retrieved, keywords)
        ndcg = ndcg_at_k(retrieved, keywords)

        # F1 score: harmonic mean of precision and recall
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results.append({
            "query": query,
            "chunks_retrieved": len(retrieved),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "mrr": round(mrr, 3),
            "ndcg": round(ndcg, 3),
            "top_scores": [round(c["score"], 3) for c in retrieved[:3]],
        })

    n = len(results) or 1
    return {
        "per_query": results,
        "aggregate": {
            "avg_precision": round(sum(r["precision"] for r in results) / n, 3),
            "avg_recall": round(sum(r["recall"] for r in results) / n, 3),
            "avg_f1": round(sum(r["f1"] for r in results) / n, 3),
            "avg_mrr": round(sum(r["mrr"] for r in results) / n, 3),
            "avg_ndcg": round(sum(r["ndcg"] for r in results) / n, 3),
            "total_queries": len(results),
        },
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "threshold": threshold,
        },
    }


def evaluate_end_to_end(
    pdf_path: str,
    test_cases: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5,
    threshold: float = 0.1,
) -> dict:
    """
    Full RAG evaluation: retrieval metrics + generation quality.

    test_cases format:
    [
        {
            "query": "What is attention?",
            "expected_keywords": ["attention", "self-attention"]
        }
    ]
    """
    from generator import generate_answer

    model = get_model()

    with open(pdf_path, "rb") as f:
        text = load_pdf(f)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    index, _ = embed_chunks(chunks, model)

    results = []
    for case in test_cases:
        query = case["query"]
        keywords = case["expected_keywords"]

        retrieved = retrieve(query, index, chunks, model, top_k=top_k, threshold=threshold)
        response = generate_answer(query, retrieved)

        # Retrieval metrics
        p = precision_at_k(retrieved, keywords)
        r = recall(retrieved, chunks, keywords)
        mrr = mean_reciprocal_rank(retrieved, keywords)
        ndcg = ndcg_at_k(retrieved, keywords)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Generation metrics (LLM-as-judge)
        context_text = "\n".join(c["text"] for c in retrieved)
        faithfulness = evaluate_faithfulness(query, context_text, response["answer"])
        relevance = evaluate_relevance(query, response["answer"])

        results.append({
            "query": query,
            "answer": response["answer"][:200] + "..." if len(response["answer"]) > 200 else response["answer"],
            "retrieval": {
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f1, 3),
                "mrr": round(mrr, 3),
                "ndcg": round(ndcg, 3),
            },
            "generation": {
                "faithfulness": faithfulness.get("score", 0),
                "faithfulness_reason": faithfulness.get("reasoning", ""),
                "relevance": relevance.get("score", 0),
                "relevance_reason": relevance.get("reasoning", ""),
            },
        })

    n = len(results) or 1
    return {
        "per_query": results,
        "aggregate": {
            "avg_precision": round(sum(r["retrieval"]["precision"] for r in results) / n, 3),
            "avg_recall": round(sum(r["retrieval"]["recall"] for r in results) / n, 3),
            "avg_f1": round(sum(r["retrieval"]["f1"] for r in results) / n, 3),
            "avg_mrr": round(sum(r["retrieval"]["mrr"] for r in results) / n, 3),
            "avg_ndcg": round(sum(r["retrieval"]["ndcg"] for r in results) / n, 3),
            "avg_faithfulness": round(sum(r["generation"]["faithfulness"] for r in results) / n, 1),
            "avg_relevance": round(sum(r["generation"]["relevance"] for r in results) / n, 1),
        },
    }


def parameter_sweep(
    pdf_path: str,
    test_cases: list[dict],
    chunk_sizes: list[int] = None,
    overlaps: list[int] = None,
    top_ks: list[int] = None,
) -> list[dict]:
    """
    Compare different RAG configurations to find the best one.

    This is hyperparameter tuning for RAG.

    Analogy: Like load testing your API with different configurations.
    You try different thread pool sizes, connection limits, etc.
    Here you try different chunk sizes, overlaps, and top_k values.

    Common findings:
    - Chunk size 300-500 usually works best for Q&A
    - Overlap of 20-30% of chunk size prevents info loss
    - top_k 3-5 balances precision and recall
    """
    if chunk_sizes is None:
        chunk_sizes = [300, 500, 800]
    if overlaps is None:
        overlaps = [50, 100, 150]
    if top_ks is None:
        top_ks = [3, 5, 7]

    sweep_results = []

    for cs in chunk_sizes:
        for ol in overlaps:
            if ol >= cs:
                continue  # overlap must be less than chunk size
            for k in top_ks:
                result = evaluate_retrieval(
                    pdf_path, test_cases,
                    chunk_size=cs, overlap=ol, top_k=k,
                )
                sweep_results.append({
                    "chunk_size": cs,
                    "overlap": ol,
                    "top_k": k,
                    "avg_f1": result["aggregate"]["avg_f1"],
                    "avg_precision": result["aggregate"]["avg_precision"],
                    "avg_recall": result["aggregate"]["avg_recall"],
                    "avg_ndcg": result["aggregate"]["avg_ndcg"],
                })

    # Sort by F1 score (best balance of precision and recall)
    sweep_results.sort(key=lambda x: x["avg_f1"], reverse=True)
    return sweep_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluator.py <pdf_path>              # Retrieval eval")
        print("  python evaluator.py <pdf_path> --full       # Full eval (retrieval + generation)")
        print("  python evaluator.py <pdf_path> --sweep      # Parameter sweep")
        sys.exit(1)

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

    mode = sys.argv[2] if len(sys.argv) > 2 else "--retrieval"

    if mode == "--full":
        print("Running full evaluation (retrieval + generation)...")
        results = evaluate_end_to_end(sys.argv[1], demo_cases)
    elif mode == "--sweep":
        print("Running parameter sweep...")
        results = parameter_sweep(sys.argv[1], demo_cases)
    else:
        print("Running retrieval evaluation...")
        results = evaluate_retrieval(sys.argv[1], demo_cases)

    print(json.dumps(results, indent=2))
