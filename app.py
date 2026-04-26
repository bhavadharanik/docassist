"""
DocAssist — RAG-Powered Document Q&A
Run: streamlit run app.py
"""

import streamlit as st
from chunker import load_pdf, chunk_text
from embedder import get_model, embed_chunks
from retriever import retrieve
from generator import generate_answer
from evaluator import (
    precision_at_k, recall, mean_reciprocal_rank, ndcg_at_k,
    evaluate_faithfulness, evaluate_relevance,
)

st.set_page_config(page_title="DocAssist", page_icon="📄", layout="wide")

# Cache the model so it only loads once across all reruns
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return get_model()

model = load_embedding_model()

st.title("📄 DocAssist")
st.caption("RAG-powered document Q&A — Upload a PDF, ask questions, get grounded answers with citations")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    st.divider()
    st.header("Settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 200, 100, 25)
    top_k = st.slider("Top-K results", 1, 10, 5)
    threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.1, 0.05)
    st.divider()
    st.markdown("""
    **RAG Pipeline:**
    ```
    PDF → Chunk → Embed → FAISS
    Query → Embed → Search
    Top-K → Gemini 2.0 → Answer
    ```
    """)

if uploaded_file:
    if "chunks" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing document..."):
            text = load_pdf(uploaded_file)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            index, embeddings = embed_chunks(chunks, model)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name
            st.session_state.doc_text = text

        st.success(f"✅ Ready! {len(chunks)} chunks indexed from {uploaded_file.name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Document", st.session_state.file_name)
    col2.metric("Chunks", len(st.session_state.chunks))
    col3.metric("Characters", f"{len(st.session_state.doc_text):,}")

    st.divider()

    # Two tabs: Chat and Evaluate
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluate"])

    # =========================================================================
    # CHAT TAB
    # =========================================================================
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Source Chunks"):
                        for src in msg["sources"]:
                            st.markdown(f"**Chunk {src['chunk_id']}** (score: {src['score']:.2f})")
                            st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                            st.divider()

        if query := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching and generating..."):
                    results = retrieve(
                        query=query,
                        index=st.session_state.index,
                        chunks=st.session_state.chunks,
                        model=model,
                        top_k=top_k,
                        threshold=threshold,
                    )
                    response = generate_answer(query, results)
                    st.markdown(response["answer"])

                    if results:
                        with st.expander("📚 Source Chunks"):
                            for src in results:
                                st.markdown(f"**Chunk {src['chunk_id']}** (score: {src['score']:.2f})")
                                st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                                st.divider()

                    st.caption(f"Model: {response['model']} | Chunks used: {response['chunks_used']} | Top-K: {top_k}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": results,
            })

    # =========================================================================
    # EVALUATION TAB
    # =========================================================================
    with tab_eval:
        st.subheader("RAG Pipeline Evaluation")
        st.markdown("""
        Test how well the retrieval and generation are working.
        Enter a query and keywords that **should** appear in the retrieved chunks.
        """)

        eval_query = st.text_input("Test query", placeholder="e.g., What is machine learning?")
        eval_keywords = st.text_input(
            "Expected keywords (comma-separated)",
            placeholder="e.g., machine learning, algorithm, training, model",
        )

        if st.button("Run Evaluation", type="primary") and eval_query and eval_keywords:
            keywords = [kw.strip() for kw in eval_keywords.split(",") if kw.strip()]

            with st.spinner("Evaluating retrieval..."):
                retrieved = retrieve(
                    query=eval_query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    model=model,
                    top_k=top_k,
                    threshold=threshold,
                )

                # Retrieval metrics
                p = precision_at_k(retrieved, keywords)
                r = recall(retrieved, st.session_state.chunks, keywords)
                mrr = mean_reciprocal_rank(retrieved, keywords)
                ndcg = ndcg_at_k(retrieved, keywords)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            st.markdown("### Retrieval Metrics")
            st.markdown("""
            *These measure: did we fetch the right chunks?*
            """)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Precision", f"{p:.2f}", help="What % of retrieved chunks are relevant")
            m2.metric("Recall", f"{r:.2f}", help="What % of all relevant chunks were found")
            m3.metric("F1 Score", f"{f1:.2f}", help="Balance of precision and recall")
            m4.metric("MRR", f"{mrr:.2f}", help="How high the first relevant result ranks")
            m5.metric("NDCG", f"{ndcg:.2f}", help="Are relevant results ranked higher?")

            # Show retrieved chunks
            if retrieved:
                st.markdown("### Retrieved Chunks")
                for i, chunk in enumerate(retrieved):
                    is_relevant = any(kw.lower() in chunk["text"].lower() for kw in keywords)
                    status = "✅ Relevant" if is_relevant else "❌ Not relevant"
                    with st.expander(f"Chunk {chunk['chunk_id']} — score: {chunk['score']:.3f} — {status}"):
                        st.text(chunk["text"])

            # Generation evaluation (LLM-as-judge)
            st.markdown("### Generation Evaluation")
            st.markdown("*Using Gemini as a judge to score the generated answer (LLM-as-Judge pattern)*")

            with st.spinner("Generating answer and evaluating..."):
                response = generate_answer(eval_query, retrieved)
                st.markdown(f"**Answer:** {response['answer']}")

                context_text = "\n".join(c["text"] for c in retrieved)
                faithfulness = evaluate_faithfulness(eval_query, context_text, response["answer"])
                relevance = evaluate_relevance(eval_query, response["answer"])

            g1, g2 = st.columns(2)
            faith_score = faithfulness.get("score", 0)
            rel_score = relevance.get("score", 0)

            g1.metric(
                "Faithfulness",
                f"{faith_score}/5",
                help="Is the answer grounded in context? (5=fully grounded, 1=hallucinating)",
            )
            g2.metric(
                "Relevance",
                f"{rel_score}/5",
                help="Does the answer address the question? (5=perfectly relevant, 1=off-topic)",
            )

            if faithfulness.get("reasoning"):
                st.caption(f"Faithfulness: {faithfulness['reasoning']}")
            if relevance.get("reasoning"):
                st.caption(f"Relevance: {relevance['reasoning']}")

        # Parameter sweep section
        st.divider()
        st.subheader("Parameter Sweep")
        st.markdown("""
        Compare different chunking configurations to find the optimal setup.
        This is **hyperparameter tuning for RAG** — like load testing with different configs.
        """)

        sweep_query = st.text_input("Sweep query", placeholder="e.g., What is deep learning?", key="sweep_q")
        sweep_keywords = st.text_input(
            "Expected keywords",
            placeholder="e.g., deep learning, neural network, layers",
            key="sweep_kw",
        )

        if st.button("Run Parameter Sweep") and sweep_query and sweep_keywords:
            keywords = [kw.strip() for kw in sweep_keywords.split(",") if kw.strip()]
            test_case = [{"query": sweep_query, "expected_keywords": keywords}]

            configs = []
            for cs in [300, 500, 800]:
                for ol in [50, 100, 150]:
                    if ol >= cs:
                        continue
                    for k in [3, 5, 7]:
                        configs.append((cs, ol, k))

            progress = st.progress(0)
            sweep_results = []

            for i, (cs, ol, k) in enumerate(configs):
                progress.progress((i + 1) / len(configs))

                chunks_sweep = chunk_text(st.session_state.doc_text, chunk_size=cs, overlap=ol)
                index_sweep, _ = embed_chunks(chunks_sweep, model)
                retrieved_sweep = retrieve(
                    sweep_query, index_sweep, chunks_sweep, model, top_k=k, threshold=threshold,
                )

                p = precision_at_k(retrieved_sweep, keywords)
                r = recall(retrieved_sweep, chunks_sweep, keywords)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                ndcg_val = ndcg_at_k(retrieved_sweep, keywords)

                sweep_results.append({
                    "Chunk Size": cs,
                    "Overlap": ol,
                    "Top-K": k,
                    "Precision": round(p, 3),
                    "Recall": round(r, 3),
                    "F1": round(f1, 3),
                    "NDCG": round(ndcg_val, 3),
                })

            progress.empty()

            # Sort by F1
            sweep_results.sort(key=lambda x: x["F1"], reverse=True)
            st.dataframe(sweep_results, use_container_width=True)

            if sweep_results:
                best = sweep_results[0]
                st.success(
                    f"Best config: chunk_size={best['Chunk Size']}, "
                    f"overlap={best['Overlap']}, top_k={best['Top-K']} "
                    f"(F1={best['F1']}, Precision={best['Precision']}, Recall={best['Recall']})"
                )

else:
    st.info("👈 Upload a PDF document to get started")
    st.markdown("""
    ### How it works
    | Step | Component | Technology |
    |------|-----------|------------|
    | 1. Upload | PDF text extraction | PyPDF2 |
    | 2. Chunk | Split into overlapping pieces | Custom chunker (500 chars, 100 overlap) |
    | 3. Embed | Convert text to vectors | sentence-transformers (all-MiniLM-L6-v2) |
    | 4. Store | Index vectors for search | FAISS (Facebook AI Similarity Search) |
    | 5. Query | Embed question with same model | sentence-transformers |
    | 6. Retrieve | Find similar chunks | FAISS cosine similarity, top-K |
    | 7. Generate | LLM answers with context | Google Gemini 2.0 Flash |
    | 8. Cite | Reference source chunks | Citation enforcement in prompt |
    """)
