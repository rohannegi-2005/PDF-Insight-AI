"""
config.py
----------
Centralized configuration for ResearchPilot.

Why centralize this?
----------------------
Model names, chunk sizes, and retrieval parameters used to live as
scattered module-level constants inside individual files
(EMBEDDING_MODEL_NAME in vector_store.py, LLM_MODEL_NAME in
rag_pipeline.py, etc.). That's fine for a single-file prototype, but as
this project grows into multiple agents and a LangGraph workflow,
several modules need to agree on the SAME values -- e.g. both the
chunker and the evaluation harness need to know the chunk size, and
both the retriever and the reranker need to know how many candidates to
pull. One shared config file is the standard fix: change a value once,
every module that imports config.py sees the new value.
"""

# ---------------------------------------------------------------------------
# Embedding model (used for the vector store AND semantic chunking --
# reusing one model for both jobs avoids downloading/loading a second one)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# "fixed"    -> RecursiveCharacterTextSplitter (character-count based)
# "semantic" -> SemanticChunker (meaning-shift based, see core/semantic_chunker.py)
CHUNKING_STRATEGY = "semantic"

FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 100

# Below this cosine similarity between consecutive sentences, start a
# new chunk (i.e. the topic probably shifted).
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
# Chunks smaller than this (in characters) get merged forward instead
# of standing alone as a tiny, low-context fragment.
SEMANTIC_MIN_CHUNK_CHARS = 300
# Chunks larger than this get force-cut even if similarity stays high,
# so no single chunk blows past a sane size for embedding/retrieval.
SEMANTIC_MAX_CHUNK_CHARS = 1500

# ---------------------------------------------------------------------------
# AstraDB
# ---------------------------------------------------------------------------
COLLECTION_NAME = "astra_vector_langchain"

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Retrieval (used starting Step 2+: hybrid search, MMR, reranking)
# ---------------------------------------------------------------------------
RETRIEVAL_K = 8
MMR_FETCH_K = 12
MMR_LAMBDA_MULT = 0.5

# ---------------------------------------------------------------------------
# Reranking (Step 4)
# ---------------------------------------------------------------------------
# Free, local cross-encoder from sentence-transformers -- no new
# dependency, same library already used for embeddings.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# How many fused hybrid candidates to feed INTO the reranker.
RERANK_CANDIDATE_K = 20
# How many chunks to keep AFTER reranking -- this is what actually
# reaches the LLM prompt, so keep it small and high-precision.
RERANK_TOP_N = 5