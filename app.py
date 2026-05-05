"""Streamlit chat UI for the Healthcare RAG Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_assistant.generator import answer_question
from rag_assistant.retriever import _reranking_enabled

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=20, value=5)
    use_cache = st.checkbox("Use answer cache", value=True)
    st.divider()
    reranking = _reranking_enabled()
    st.markdown(
        f"**Reranking:** {'✅ Cohere enabled' if reranking else '⬜ Vector-only (no COHERE_API_KEY)'}"
    )
    st.divider()
    st.markdown(
        "Ask any question about the indexed healthcare policy documents. "
        "Answers are grounded in the documents and include page citations."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main header ---
st.title("🏥 Healthcare RAG Assistant")
st.caption("Grounded answers with source citations from your indexed policy documents.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}** — page {src['page']}")
        if msg.get("meta"):
            st.caption(msg["meta"])

# --- Input ---
if prompt := st.chat_input("Ask about healthcare policy..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                result = answer_question(prompt, top_k=top_k, use_cache=use_cache)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        st.markdown(result["answer"])

        sources = result.get("sources", [])
        if sources:
            with st.expander("📄 Sources", expanded=True):
                for src in sources:
                    st.markdown(f"- **{src['source']}** — page {src['page']}")

        meta_parts = []
        if result.get("from_cache"):
            meta_parts.append("served from cache")
        else:
            if result.get("latency_s") is not None:
                meta_parts.append(f"latency {result['latency_s']}s")
            if result.get("token_count") is not None:
                meta_parts.append(f"{result['token_count']} tokens")
        meta = " · ".join(meta_parts)
        if meta:
            st.caption(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
        "meta": meta,
    })
