"""
GENERATOR — Step 4 of RAG Pipeline
Uses Ollama (local LLM) for grounded answer generation.
No API key needed. Runs completely offline.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question
based ONLY on the provided context chunks. Follow these rules strictly:

1. Only use information from the provided context chunks
2. Cite your sources using [Chunk X] format after each claim
3. If the answer is not in the context, say "I cannot find this information in the provided documents"
4. Be concise and direct
5. Do not make up or infer information beyond what's explicitly stated"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "chunks_used": 0,
            "model": MODEL,
        }

    context = "\n\n".join(
        f"[Chunk {chunk['chunk_id']}] (relevance: {chunk['score']:.2f}):\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["response"]
    except Exception as e:
        answer = f"Error calling Ollama: {str(e)}"

    return {
        "answer": answer,
        "chunks_used": len(retrieved_chunks),
        "model": MODEL,
    }
