"""
core/retrieval.py
-------------------
Hybrid retrieval: run dense (embedding) search and sparse (BM25) search
independently, then merge their results into one final ranked list
using Reciprocal Rank Fusion (RRF).

Why RRF specifically?
-----------------------
Dense search returns a cosine-similarity score (e.g. 0.82). BM25 returns
a completely different kind of score (e.g. 4.3, unbounded). These
numbers aren't on the same scale, so averaging or comparing them
directly would be meaningless -- a 0.82 dense score isn't "better" or
"worse" than a 4.3 BM25 score in any comparable sense.

RRF sidesteps this entirely by ignoring the raw scores and only looking
at RANK POSITION (1st place, 2nd place, ...) in each list. A chunk that
ranks highly in BOTH lists rises to the top of the fused ranking. A
chunk that only shows up in one list still gets some credit, just less.
This is the standard, well-established way real hybrid search systems
(Elasticsearch, Weaviate, etc.) combine multiple retrievers.

Where does MMR fit here?
---------------------------
MMR (diversity within the dense results) already happens INSIDE
core/vector_store.py's get_retriever() -- we reuse that function as-is
for the dense leg of this hybrid search, rather than re-implementing
MMR again here. So the dense candidates arriving into fusion are
already diversified before BM25 even gets involved.
"""

from typing import Dict, List

from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

import config
from core.sparse_store import SparseStore
from core.vector_store import get_retriever


def _chunk_key(doc: Document) -> str:
    """
    A stable identity for a chunk, used to recognize "this is the same
    chunk" whether it came back from the dense retriever or the sparse
    (BM25) retriever.

    Prefers the deterministic chunk_id set in document_utils.py. Falls
    back to a best-effort key (source file + page + first 50 chars) for
    any chunk that somehow doesn't have one, so fusion never crashes --
    it just treats that chunk as unique if it can't find a proper ID.
    """
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    return f"{doc.metadata.get('source_file')}::{doc.metadata.get('page')}::{doc.page_content[:50]}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Combine multiple ranked lists of documents into a single fused
    ranking using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : List[List[Document]]
        One list per retriever (e.g. [dense_results, sparse_results]),
        each already sorted best-first.
    rrf_k : int
        The RRF smoothing constant. 60 is the value used in the
        original RRF paper and is a common default -- it softens how
        much a rank-1 vs rank-2 difference matters, so one retriever's
        very top pick doesn't completely dominate the fused ranking.

    Returns
    -------
    A single list of Documents, fused-score descending (best first),
    with duplicates across lists merged into one entry.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = _chunk_key(doc)
            doc_lookup[key] = doc
            # rank is 0-indexed here, so the top result contributes
            # 1 / (rrf_k + 1), the second contributes 1 / (rrf_k + 2), etc.
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_lookup[key] for key in ranked_keys]


def hybrid_search(
    query: str,
    vector_store: AstraDBVectorStore,
    sparse_store: SparseStore,
    k: int = config.RETRIEVAL_K,
    candidate_pool_size: int = config.MMR_FETCH_K,
) -> List[Document]:
    """
    Run dense (MMR) and sparse (BM25) search for the same query, fuse
    the two ranked lists, and return the final top-k chunks.

    Parameters
    ----------
    query : str
        The search query.
    vector_store : AstraDBVectorStore
        The dense vector store (see core/vector_store.py).
    sparse_store : SparseStore
        The BM25 keyword index (see core/sparse_store.py). Must already
        have `.build()` called on it.
    k : int
        Final number of chunks to return after fusion.
    candidate_pool_size : int
        How many candidates to pull from EACH retriever before fusion --
        larger than k, so fusion has enough material to find the best
        overlap between the two lists rather than only ever seeing k
        candidates from each side.
    """
    dense_retriever = get_retriever(vector_store, k=candidate_pool_size)
    dense_results = dense_retriever.invoke(query)

    sparse_results: List[Document] = []
    if sparse_store.is_ready():
        sparse_results = sparse_store.search(query, k=candidate_pool_size)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    return fused[:k]