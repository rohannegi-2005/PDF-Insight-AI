"""
core/reranker.py
------------------
Cross-encoder reranking.

Why do we need this AFTER hybrid fusion?
-------------------------------------------
Dense search and BM25 both score a query against a passage
INDEPENDENTLY -- the query gets embedded/tokenized on its own, the
passage gets embedded/tokenized on its own, and the two are compared
afterward (cosine similarity, or word overlap). Neither ever actually
looks at the query and the passage TOGETHER, side by side.

A cross-encoder does exactly that: it feeds the query and the passage
into the model AT THE SAME TIME, so it can directly judge "does this
passage answer this specific question" rather than "are these two
things generally similar." This is meaningfully more accurate.

The trade-off is cost: a cross-encoder score can't be precomputed the
way an embedding can, since it depends on the specific query -- so
running it over an entire document collection for every question would
be far too slow. That's why it's used as a RERANKING step on a small
shortlist (the output of hybrid_search), never as the primary retriever
over a whole collection. Cheap methods narrow the field; the expensive,
accurate method picks the final winners.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 -- free, small, runs
locally, and ships as part of the sentence-transformers library already
installed for embeddings. No new dependency, no paid API.
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config

# Loaded once per process and reused -- loading a cross-encoder model
# from disk/HuggingFace on every call would be needlessly slow. Similar
# in spirit to app.py's st.cache_resource, but this module doesn't
# depend on Streamlit, so it uses a plain module-level cache instead.
_reranker_instance: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Load (and cache) the cross-encoder reranking model."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _reranker_instance


def rerank_with_scores(
    query: str,
    documents: List[Document],
    top_n: int = config.RERANK_TOP_N,
) -> List[Tuple[Document, float]]:
    """
    Re-score a shortlist of candidate documents against the query using
    a cross-encoder, and return the top_n best (document, score) pairs,
    highest score first.

    Returning the score too (not just the document) matters for Step 6
    -- the Critic agent can use a low top reranker score as a signal
    that the evidence isn't actually a strong match, even if it was the
    best of what hybrid search returned.
    """
    if not documents:
        return []

    reranker = get_reranker()

    # Cross-encoders score (query, passage) PAIRS -- one pair per
    # candidate, all scored together in one batched call for speed.
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
    scored_docs.sort(key=lambda pair: pair[1], reverse=True)

    return scored_docs[:top_n]


def rerank(
    query: str,
    documents: List[Document],
    top_n: int = config.RERANK_TOP_N,
) -> List[Document]:
    """
    Same as `rerank_with_scores()`, but returns just the documents
    (no scores) -- convenient when the caller only needs the final
    ranked chunks, e.g. to hand straight to the LLM prompt.
    """
    return [doc for doc, _score in rerank_with_scores(query, documents, top_n)]