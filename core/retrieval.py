"""
core/retrieval.py
-------------------
Hybrid retrieval: dense + sparse search, merged with Reciprocal Rank
Fusion (RRF), then re-ordered by a cross-encoder reranker.

    query --> dense (MMR) ----+
                              +--> RRF fusion --> reranker --> final top-N
    query --> sparse (BM25) --+

Why RRF? Dense gives cosine scores (0.82), BM25 gives other numbers
(4.3). They cannot be compared. RRF ignores the scores and only uses the
RANK (1st, 2nd, ...): a chunk that ranks high in BOTH lists wins.

Failure handling: if AstraDB has a hiccup, dense search is retried; if it
still fails we continue with sparse results only (and log a warning)
instead of crashing the whole run.
"""

import logging
import time
from typing import Dict, List, Optional

from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

import config
from core.reranker import rerank
from core.sparse_store import SparseStore
from core.vector_store import get_retriever

logger = logging.getLogger(__name__)


def chunk_key(doc: Document) -> str:
    """
    A stable identity for a chunk. The same chunk coming back from dense
    AND from sparse search must be recognised as ONE chunk.
    Uses the chunk_id made in document_utils.py; falls back to
    file + page + first characters for chunks that do not have one.
    """
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    return f"{doc.metadata.get('source_file')}::{doc.metadata.get('page')}::{doc.page_content[:50]}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    rrf_k: int = config.RRF_K,
) -> List[Document]:
    """
    Merge several ranked lists into one. Each chunk gets 1/(rrf_k + rank)
    points from every list it appears in (rank starts at 1); points are
    added up; highest total first.
    """
    scores: Dict[str, float] = {}
    lookup: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = chunk_key(doc)
            lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [lookup[key] for key in ordered_keys]


def _safe_dense_search(
    vector_store: AstraDBVectorStore,
    query: str,
    pool_size: int,
) -> Optional[List[Document]]:
    """
    Dense search with a retry. Returns None (not an exception) if every
    attempt fails, so the caller can decide how to continue.
    """
    retriever = get_retriever(vector_store, k=pool_size)
    last_error: Optional[Exception] = None

    for attempt in range(config.DENSE_SEARCH_ATTEMPTS):
        try:
            return retriever.invoke(query)
        except Exception as error:  # AstraDB 503 / timeout / 400 ...
            last_error = error
            if attempt < config.DENSE_SEARCH_ATTEMPTS - 1:
                time.sleep(1)

    logger.warning("Dense search failed after retries: %s", last_error)
    return None


def hybrid_search(
    query: str,
    vector_store: AstraDBVectorStore,
    sparse_store: SparseStore,
    k: int = config.RETRIEVAL_K,
    candidate_pool_size: int = config.CANDIDATE_POOL_SIZE,
) -> List[Document]:
    """
    Dense (MMR) + sparse (BM25) search for one query, fused with RRF,
    returning the top-k chunks.

    `candidate_pool_size` is how many candidates EACH retriever gives to
    the fusion (more than k, so fusion can find real overlap).
    """
    sparse_ready = sparse_store.is_ready()

    dense_results = _safe_dense_search(vector_store, query, candidate_pool_size)
    if dense_results is None:
        if not sparse_ready:
            raise RuntimeError(
                "Dense search failed and there is no sparse index to fall back on."
            )
        logger.warning("Continuing with sparse (BM25) results only.")
        dense_results = []

    sparse_results: List[Document] = []
    if sparse_ready:
        sparse_results = sparse_store.search(query, k=candidate_pool_size)

    return reciprocal_rank_fusion([dense_results, sparse_results])[:k]


def hybrid_search_with_rerank(
    query: str,
    vector_store: AstraDBVectorStore,
    sparse_store: SparseStore,
    candidate_pool_size: int = config.CANDIDATE_POOL_SIZE,
    rerank_candidate_k: int = config.RERANK_CANDIDATE_K,
    final_top_n: int = config.RERANK_TOP_N,
) -> List[Document]:
    """
    The full retrieval pipeline in one call:
    dense + sparse -> RRF fusion -> cross-encoder rerank -> top-N chunks.
    Returns [] (an empty list) if nothing was found.
    """
    fused = hybrid_search(
        query,
        vector_store,
        sparse_store,
        k=rerank_candidate_k,
        candidate_pool_size=candidate_pool_size,
    )
    return rerank(query, fused, top_n=final_top_n)