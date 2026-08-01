"""
core/vector_store.py
----------------------
Embedding model + AstraDB vector store setup.

Isolating this from the RAG chain logic means the embedding model and DB
connection are built through one shared entry point instead of being
scattered across the app. In `app.py`, this module's functions are wrapped
with `st.cache_resource` so the HuggingFace model is loaded and the DB
connection opened only ONCE per process -- not re-created on every single
Streamlit rerun (which happens on every user interaction).
"""

import os
from typing import List, Optional

from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Module-level constants (rather than hardcoded inline) so they're easy to
# spot, change, and reference from tests.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "astra_vector_langchain"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Build the sentence-transformer embedding model.

    all-MiniLM-L6-v2 is a good default for a portfolio RAG project: it's
    small (~80MB), runs fast on CPU, and produces 384-dim embeddings -- a
    solid accuracy/latency trade-off for demo-scale document sets.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store(embeddings: Optional[HuggingFaceEmbeddings] = None) -> AstraDBVectorStore:
    """
    Connect to the AstraDB collection used to store document embeddings.

    Credentials are read from environment variables (populated via
    `load_dotenv()` once at app startup) rather than hardcoded, so the
    same code works locally and in a deployed environment (e.g. Streamlit
    Community Cloud secrets, Docker env vars) without any changes.
    """
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_TOKEN")

    if not api_endpoint or not token:
        raise EnvironmentError(
            "Missing AstraDB credentials. Make sure ASTRA_DB_API_ENDPOINT "
            "and ASTRA_DB_TOKEN are set in your .env file."
        )

    if embeddings is None:
        embeddings = get_embedding_model()

    return AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=api_endpoint,
        collection_name=COLLECTION_NAME,
        token=token,
        namespace=None,
    )


def get_retriever(vector_store: AstraDBVectorStore, k: int = 8):
    """
    Wrap the vector store as a LangChain retriever using Maximal Marginal
    Relevance (MMR) search instead of plain similarity search.

    Plain similarity search can return several near-duplicate chunks if
    a book repeats a theme across pages -- MMR explicitly balances
    relevance against diversity, so the k chunks passed to the LLM cover
    more distinct angles on the question instead of overlapping content.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,          # final number of chunks returned
            "fetch_k": 20,   # candidate pool MMR selects from before diversifying
            "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance (0.5 balances both)
        },
    )

def ingest_documents(vector_store: AstraDBVectorStore, documents: List[Document]) -> List[str]:
    """
    Push a list of processed `Document` chunks into AstraDB and return the
    IDs AstraDB assigned to them.

    Kept as a thin wrapper (rather than calling `vector_store.add_documents`
    directly from the UI) so ingestion always goes through one place --
    useful later if you want to add logging, batching, or de-duplication.
    """
    if not documents:
        return []
    return vector_store.add_documents(documents)


def clear_vector_store(vector_store: AstraDBVectorStore) -> None:
    """Delete every document from the collection. Used by a 'reset KB' control."""
    vector_store.clear()
