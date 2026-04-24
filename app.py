"""
DocAssist — RAG-Powered Document Q&A
Run: streamlit run app.py
"""

import streamlit as st
from chunker import load_pdf, chunk_text
from embedder import get_model, embed_chunks
from retriever import retrieve
from generator import generate_answer

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
    Question → Embed → Search
    Top-K → Gemini → Answer
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
    | 7. Generate | LLM answers with context | Google Gemini 1.5 Flash |
    | 8. Cite | Reference source chunks | Citation enforcement in prompt |
    """)
