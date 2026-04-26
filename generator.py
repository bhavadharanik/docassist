"""
GENERATOR — Step 4 of RAG Pipeline
Uses Google Gemini for grounded answer generation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question
based ONLY on the provided context chunks. Follow these rules strictly:

1. Only use information from the provided context chunks
2. Cite your sources using [Chunk X] format after each claim
3. If the answer is not in the context, say "I cannot find this information in the provided documents"
4. Be concise and direct
5. Do not make up or infer information beyond what's explicitly stated"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    import google.generativeai as genai

    model_name = "gemini-2.0-flash"

    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "chunks_used": 0,
            "model": model_name,
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY not set in .env file.",
            "chunks_used": 0,
            "model": model_name,
        }

    genai.configure(api_key=api_key)

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
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2),
        )
        answer = response.text
    except Exception as e:
        answer = f"Error calling Gemini: {str(e)}"

    return {
        "answer": answer,
        "chunks_used": len(retrieved_chunks),
        "model": model_name,
    }
