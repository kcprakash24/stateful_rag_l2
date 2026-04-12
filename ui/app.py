import sys
import logging
import uuid
from pathlib import Path

# Suppress noise
logging.getLogger("langfuse").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from stateful_rag.ingestion.loader import load_pdf
from stateful_rag.ingestion.chunker import chunk_document
from stateful_rag.embeddings.embedder import embed_texts
from stateful_rag.vectorstore.pgvector_store import add_chunks, get_collection_stats
from stateful_rag.agent.graph import ask
from stateful_rag.config import get_settings

settings = get_settings()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatefulRAG",
    page_icon="🧠",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = {}  # keyed by user_id

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "current_user" not in st.session_state:
    st.session_state.current_user = "dave"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 StatefulRAG L2")
    st.caption("Stateful multi-user RAG with memory")

    st.divider()

    # User selector
    st.subheader("👤 Current User")
    user_options = ["dave", "mike", "other"]
    selected = st.selectbox(
        "Select user",
        options=user_options,
        index=user_options.index(st.session_state.current_user)
        if st.session_state.current_user in user_options else 0,
    )

    if selected == "other":
        custom_user = st.text_input("Enter username")
        if custom_user:
            st.session_state.current_user = custom_user.lower().strip()
    else:
        st.session_state.current_user = selected

    st.caption(f"Active: `{st.session_state.current_user}`")
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")

    st.divider()

    # Knowledge base stats
    st.subheader("📦 Knowledge Base")
    try:
        stats = get_collection_stats()
        st.metric("Total chunks", stats["total_chunks"])
        if stats["sources"]:
            for s in stats["sources"]:
                st.caption(f"📄 {s['source']} — {s['chunks']} chunks")
    except Exception as e:
        st.error(f"DB error: {e}")

    st.divider()

    # PDF uploader
    st.subheader("📄 Upload Paper")
    uploaded_file = st.file_uploader(
        "Upload a research PDF",
        type=["pdf"],
    )

    if uploaded_file:
        papers_dir = Path(__file__).parent.parent / "data" / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        save_path = papers_dir / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("⚙️ Ingest Paper", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    doc = load_pdf(save_path)
                    chunks = chunk_document(
                        doc,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                    texts = [c.text for c in chunks]
                    embeddings = embed_texts(texts)
                    inserted = add_chunks(chunks, embeddings)
                    st.success(f"✅ {inserted} new chunks ingested")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # Clear chat for current user
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages[st.session_state.current_user] = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
user_id = st.session_state.current_user
session_id = st.session_state.session_id

st.title(f"Chat — {user_id.capitalize()}")
st.caption("Each user has isolated memory. Switch users in the sidebar.")

# Init message list for this user if needed
if user_id not in st.session_state.messages:
    st.session_state.messages[user_id] = []

# Render chat history for current user
for message in st.session_state.messages[user_id]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            # Cache hit indicator
            if message.get("cache_hit"):
                st.caption("⚡ Cache hit")

            # Sources
            if message.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in message["sources"]:
                        st.markdown(f"**{s['chunk_id']}**")
                        st.caption(s.get("preview", "")[:200])
                        if s.get("similarity"):
                            st.caption(f"Similarity: {s['similarity']}")
                        st.divider()

# Chat input
if question := st.chat_input(
    f"Ask a question as {user_id.capitalize()}..."
):
    # Show user message
    st.session_state.messages[user_id].append({
        "role": "user",
        "content": question,
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                )

                st.markdown(result["answer"])

                if result["cache_hit"]:
                    st.caption("⚡ Cache hit")

                if result["sources"]:
                    with st.expander("📎 Sources", expanded=False):
                        for s in result["sources"]:
                            st.markdown(f"**{s['chunk_id']}**")
                            st.caption(s.get("preview", "")[:200])
                            if s.get("similarity"):
                                st.caption(f"Similarity: {s['similarity']}")
                            st.divider()

                # Save to session state
                st.session_state.messages[user_id].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "cache_hit": result["cache_hit"],
                })

            except Exception as e:
                st.error(f"Error: {e}")