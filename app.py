"""
app.py
------
Streamlit front-end for the PDF-Insight-AI conversational RAG assistant.

This file's job is ONLY to:
- render the chat UI
- own Streamlit-specific state (session_state, caching, widgets)
- glue together the `core/` modules (document processing, vector store,
  RAG pipeline)

All actual RAG / embedding / DB logic lives in `core/`, so this file stays
thin and easy to read top-to-bottom during a walkthrough. Run it with:

    streamlit run app.py
"""

import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from core.document_utils import SmartPDFProcessor
from core.rag_pipeline import (
    build_conversational_rag_chain,
    clear_session_history,
    stream_answer,
)
from core.vector_store import (
    clear_vector_store,
    get_embedding_model,
    get_retriever,
    get_vector_store,
    ingest_documents,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(page_title="PDF-Insight-AI", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# `st.cache_resource` makes sure the embedding model, DB connection, and
# LLM/chain are built ONCE per process and reused across every rerun and
# every user interaction. Without this, plain module-level code would be
# re-executed on EVERY interaction, since Streamlit re-runs the whole
# script top-to-bottom each time -- that would mean reloading the
# HuggingFace model and reopening a DB connection on every chat message.

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings):
    # The leading underscore on the parameter name tells Streamlit's
    # hasher to skip hashing this (unhashable) object -- it's only used
    # to build the store, not as part of the cache key.
    return get_vector_store(_embeddings)


@st.cache_resource(show_spinner=False)
def load_conversational_chain(_retriever):
    return build_conversational_rag_chain(_retriever)


def load_pipeline():
    """
    Build (or fetch from cache) every piece of the pipeline behind one
    call, so the rest of the app doesn't need to know the wiring order.
    """
    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever = get_retriever(vector_store, k=8)
    chain = load_conversational_chain(retriever)
    return vector_store, chain


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    # One random ID per browser session -- this is the key that keeps
    # different users' (or tabs') conversation histories separate inside
    # RunnableWithMessageHistory's session store (see core/rag_pipeline.py).
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


# ---------------------------------------------------------------------------
# Sidebar: document ingestion + controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Knowledge Base")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # Re-indexing the same PDF (or any PDF) without clearing first would
    # just append more copies of the same chunks into AstraDB -- vector
    # stores have no built-in de-duplication, so every "Process & Index"
    # click is a pure insert. Left unchecked over several uploads, that
    # means duplicate chunks crowding out genuinely different content
    # during retrieval. Defaulting this to checked keeps the knowledge
    # base clean for a single-document demo app; uncheck it only if you
    # deliberately want to build up a multi-document collection.
    clear_before_indexing = st.checkbox(
        "Clear existing knowledge base before indexing",
        value=True,
        help=(
            "Recommended: prevents duplicate chunks from piling up if you "
            "re-upload the same (or a different) PDF more than once."
        ),
    )

    if uploaded_file is not None and st.button("Process & Index PDF", use_container_width=True):
        try:
            vector_store_for_ingest = load_vector_store(load_embeddings())

            if clear_before_indexing:
                with st.spinner("Clearing existing knowledge base..."):
                    clear_vector_store(vector_store_for_ingest)

            with st.spinner("Chunking PDF and generating embeddings..."):
                # PyPDFLoader needs a real file path, so the uploaded
                # in-memory file is written to a temp file first.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                processor = SmartPDFProcessor()
                chunks = processor.process_pdf(tmp_path)
                os.unlink(tmp_path)

            if chunks:
                with st.spinner(f"Inserting {len(chunks)} chunks into AstraDB..."):
                    ingest_documents(vector_store_for_ingest, chunks)
                st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'.")
            else:
                st.warning("No extractable text found in this PDF.")

        except EnvironmentError as e:
            st.error(str(e))

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        clear_session_history(st.session_state.session_id)
        st.rerun()

    st.caption(
        "Conversational memory, streaming answers, and source citations "
        "are all backed by AstraDB + Groq (Llama-3.1-8B-Instant)."
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("📄 PDF-Insight-AI")
st.caption("Ask questions about your indexed PDF(s) — with memory, streaming, and citations.")

try:
    vector_store, conversational_rag_chain = load_pipeline()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()


def render_sources(sources) -> None:
    """Small helper so history redraws and live answers render sources identically."""
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page", "?")
            source_file = doc.metadata.get("source_file", "document")
            st.markdown(f"**[{i}] {source_file} — page {page}**")
            preview = doc.page_content[:400]
            st.caption(preview + ("..." if len(doc.page_content) > 400 else ""))


# Render chat history. Streamlit has no persistent DOM between
# interactions, so every rerun redraws the whole conversation from
# `st.session_state.messages`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

# Chat input
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    # 1) Show + store the user's message
    st.session_state.messages.append({"role": "user", "content": user_question, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2) Stream the assistant's answer
    with st.chat_message("assistant"):
        sources: list = []  # populated by stream_answer as a side effect
        full_answer = st.write_stream(
            stream_answer(
                conversational_rag_chain,
                user_question,
                st.session_state.session_id,
                sources,
            )
        )

        # De-duplicate retrieved chunks (defensive -- keeps the citations
        # expander clean even if a chunk shows up more than once).
        seen = set()
        unique_sources = []
        for doc in sources:
            key = (doc.metadata.get("source_file"), doc.metadata.get("page"), doc.page_content[:50])
            if key not in seen:
                seen.add(key)
                unique_sources.append(doc)

        if unique_sources:
            render_sources(unique_sources)

    # 3) Store the assistant's turn (with sources) so it redraws correctly
    #    on the next rerun.
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": unique_sources,
    })
