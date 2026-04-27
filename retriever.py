"""
RETRIEVER — Step 3 of RAG Pipeline
====================================
Two retrieval strategies, combined:

1. SEMANTIC SEARCH (what we already had)
   - Embeds query → finds similar vectors in FAISS
   - Understands MEANING: "car" matches "automobile"
   - Weakness: misses exact keyword matches

2. KEYWORD SEARCH (BM25 — new)
   - Counts word frequency, rewards rare words
   - Finds EXACT TERMS: "NDCG" matches "NDCG"
   - Weakness: doesn't understand synonyms

3. HYBRID = combine both with weighted scores
   - Production systems (Google, Perplexity) all do this
   - Called "Reciprocal Rank Fusion" (RRF) when merging ranked lists

Interview explanation:
"Pure semantic search misses exact keyword matches, and pure keyword search
doesn't understand synonyms. I implemented hybrid retrieval using BM25 for
keyword matching and FAISS for semantic similarity, fused with Reciprocal
Rank Fusion. This improved recall without sacrificing precision."

=== BM25 EXPLAINED (for interview) ===

BM25 = "Best Matching 25" — the algorithm behind Elasticsearch, Solr, etc.

Think of it like this:
- TF (Term Frequency): How many times does the word appear in THIS chunk?
  More occurrences = more relevant. But with diminishing returns.
  (Like code reviews: first mention of a bug is important, 10th mention
   doesn't add 10x value)

- IDF (Inverse Document Frequency): How RARE is this word across ALL chunks?
  Rare words are more informative than common ones.
  "the" appears everywhere = low IDF = not useful for search
  "NDCG" appears in 2 chunks = high IDF = very useful for search
  (Like grep: searching for "func" matches everything. Searching for
   "calculateNDCG" finds exactly what you need)

- BM25 combines TF and IDF with length normalization:
  score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * len/avglen))
  k1 = how much TF matters (default 1.5)
  b = how much length normalization matters (default 0.75)

  Don't memorize the formula. Know: "BM25 rewards chunks where the search
  term appears frequently AND the term is rare across all chunks."
"""

import math
from collections import Counter
from embedder import embed_query


# =============================================================================
# SEMANTIC RETRIEVER (FAISS — already existed)
# =============================================================================

def retrieve_semantic(query: str, index, chunks: list[dict], model, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """
    Semantic search using FAISS cosine similarity.
    Finds chunks with similar MEANING to the query.
    """
    query_embedding = embed_query(query, model)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "chunk_id": chunks[idx]["chunk_id"],
            "score": float(score),
            "method": "semantic",
        })
    return results


# =============================================================================
# BM25 KEYWORD RETRIEVER (new)
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Production systems use spaCy/NLTK."""
    return [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split() if len(w.strip(".,!?;:()[]{}\"'")) > 1]


class BM25:
    """
    BM25 scoring for keyword-based retrieval.

    Analogy: Like grep, but smarter.
    - grep: does "NDCG" appear? yes/no
    - BM25: does "NDCG" appear? how many times? how rare is "NDCG"
      across all chunks? how long is this chunk vs average?

    The result is a relevance SCORE, not just a binary match.
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation. Higher = TF matters more.
        self.b = b     # Length normalization. 0 = ignore length. 1 = full normalization.
        self.chunks = chunks
        self.n_chunks = len(chunks)

        # Tokenize all chunks
        self.chunk_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        self.chunk_lengths = [len(tokens) for tokens in self.chunk_tokens]
        self.avg_length = sum(self.chunk_lengths) / self.n_chunks if self.n_chunks else 1

        # Count how many chunks contain each word (for IDF)
        self.doc_freqs = Counter()
        for tokens in self.chunk_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

    def _idf(self, term: str) -> float:
        """
        Inverse Document Frequency: how rare is this term?

        IDF = log((N - df + 0.5) / (df + 0.5))
        N = total chunks, df = chunks containing this term

        "the" → df=100, IDF≈0 (useless for search)
        "NDCG" → df=2, IDF≈3.5 (very useful for search)
        """
        df = self.doc_freqs.get(term, 0)
        return math.log((self.n_chunks - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> list[tuple[int, float]]:
        """Score all chunks against the query. Returns [(chunk_idx, score), ...]"""
        query_tokens = _tokenize(query)
        scores = []

        for idx, chunk_tokens in enumerate(self.chunk_tokens):
            chunk_score = 0.0
            tf_counter = Counter(chunk_tokens)
            chunk_len = self.chunk_lengths[idx]

            for term in query_tokens:
                if term not in tf_counter:
                    continue

                tf = tf_counter[term]
                idf = self._idf(term)

                # BM25 formula: IDF * (TF * (k1+1)) / (TF + k1 * (1 - b + b * len/avglen))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * chunk_len / self.avg_length)
                chunk_score += idf * numerator / denominator

            scores.append((idx, chunk_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def retrieve_keyword(query: str, chunks: list[dict], bm25: BM25, top_k: int = 5) -> list[dict]:
    """
    Keyword search using BM25.
    Finds chunks containing the query's exact terms, weighted by rarity.
    """
    scores = bm25.score(query)
    results = []
    for idx, score in scores[:top_k]:
        if score <= 0:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "chunk_id": chunks[idx]["chunk_id"],
            "score": float(score),
            "method": "keyword",
        })
    return results


# =============================================================================
# HYBRID RETRIEVER (Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    index,
    chunks: list[dict],
    model,
    bm25: BM25,
    top_k: int = 5,
    threshold: float = 0.3,
    semantic_weight: float = 0.6,
) -> list[dict]:
    """
    Hybrid retrieval: combine semantic + keyword search.

    Uses Reciprocal Rank Fusion (RRF):
    - Each method produces a ranked list
    - RRF score = sum of 1/(k + rank) for each method
    - k=60 is standard (prevents top-1 result from dominating)

    Why RRF instead of just averaging scores?
    - Semantic scores (cosine sim) are 0-1
    - BM25 scores can be 0-50+
    - Can't directly compare. RRF uses RANKS instead of scores.

    Analogy: Two code reviewers rank PRs by priority. You combine their
    rankings, not their raw scores (one uses 1-10, other uses 1-100).

    semantic_weight: 0.6 = trust semantic more. Tune this based on your
    evaluation results.
    """
    # Get results from both methods
    semantic_results = retrieve_semantic(query, index, chunks, model, top_k=top_k * 2, threshold=threshold)
    keyword_results = retrieve_keyword(query, chunks, bm25, top_k=top_k * 2)

    # RRF fusion constant
    k = 60

    # Build RRF scores per chunk_id
    rrf_scores = {}
    chunk_map = {}

    keyword_weight = 1.0 - semantic_weight

    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + semantic_weight * (1.0 / (k + rank + 1))
        chunk_map[cid] = result

    for rank, result in enumerate(keyword_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + keyword_weight * (1.0 / (k + rank + 1))
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Sort by RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, rrf_score in sorted_chunks[:top_k]:
        result = chunk_map[cid]
        results.append({
            "text": result["text"],
            "chunk_id": cid,
            "score": float(rrf_score),
            "method": "hybrid",
        })
    return results


# =============================================================================
# DEFAULT RETRIEVE FUNCTION (backward compatible)
# =============================================================================

def retrieve(query: str, index, chunks: list[dict], model, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """Default retriever — uses semantic search for backward compatibility."""
    return retrieve_semantic(query, index, chunks, model, top_k=top_k, threshold=threshold)
