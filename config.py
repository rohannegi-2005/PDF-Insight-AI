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
# NOTE: if this model ever 404s with "model_not_found", check the live,
# authoritative list at https://console.groq.com/docs/models -- Groq's
# available models can change, and access can vary by account.
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0
# The Groq client retries rate-limit (429) and server (5xx) errors by itself,
# with waiting in between. 4 is more forgiving than the default of 2.
LLM_MAX_RETRIES = 4
# How many times we ask for a "fixed form" answer (see agents/structured.py).
STRUCTURED_OUTPUT_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Retrieval (used starting Step 2+: hybrid search, MMR, reranking)
# ---------------------------------------------------------------------------
# Final number of chunks for the plain dense-only chat chain.
RETRIEVAL_K = 8
# How many candidates EACH retriever (dense, sparse) gives to the fusion step.
CANDIDATE_POOL_SIZE = 12
# MMR looks at (k x this) nearest chunks, then keeps the k most diverse ones.
MMR_FETCH_MULTIPLIER = 2
# 0 = maximum diversity, 1 = maximum relevance.
MMR_LAMBDA_MULT = 0.5
# Reciprocal Rank Fusion smoothing constant (60 = the value from the RRF paper).
RRF_K = 60
# If AstraDB hiccups (503, timeout), try the dense search this many times
# before continuing with sparse results only.
DENSE_SEARCH_ATTEMPTS = 2

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

# ---------------------------------------------------------------------------
# Agentic loop (Step 7)
# ---------------------------------------------------------------------------
# Max number of times the graph will rewrite the query and search again
# after the Critic judges evidence insufficient. Without a hard cap, a
# genuinely unanswerable question (not covered by the document at all)
# would retry forever -- this guarantees the graph always terminates,
# answering with best-effort evidence if the budget runs out.
MAX_RETRIES = 2
# Evidence collected across retries is capped at this many chunks, so the
# answer prompt does not grow without limit.
MAX_EVIDENCE_CHUNKS = 8

# ---------------------------------------------------------------------------
# Multi-question research (Step 8)
# ---------------------------------------------------------------------------
# Max number of sub-questions a big question can be split into. Every
# sub-question runs the full retrieve -> critic -> retry loop, so each
# one costs several LLM calls. A cap keeps cost and rate-limit use sane.
MAX_SUB_QUESTIONS = 3