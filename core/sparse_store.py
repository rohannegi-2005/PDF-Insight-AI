"""
core/sparse_store.py
----------------------
Sparse (keyword-based) retrieval using BM25.

Why do we need this if we already have dense (embedding) search?
--------------------------------------------------------------------
Dense embeddings are great at matching MEANING ("financial trouble" ~
"money problems") but can miss EXACT matches -- a specific name, an
acronym, a number, a rare technical term the embedding model never
learned to weigh heavily. BM25 is the opposite: it's a classic
keyword-ranking algorithm (the backbone of search engines for decades
before embeddings existed) -- it scores a chunk higher the more its
exact words overlap with the query, giving extra weight to rare/
important words and less weight to common ones ("the", "is", "paper").

Combining both (Step 3 -- hybrid fusion) gives the best of each: BM25
catches exact terms dense search might miss; dense search catches
paraphrases/synonyms BM25 would miss.

No paid API, no external service -- rank_bm25 runs entirely locally in
Python, scoring purely from word overlap statistics.
"""

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    Turn a string into a list of lowercase word tokens.

    BM25 doesn't understand meaning -- it just counts which exact words
    appear where -- so we lowercase everything and strip punctuation so
    that "Method", "method,", and "method." are all treated as the same
    token instead of three different ones.
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class SparseStore:
    """
    An in-memory BM25 keyword index over a set of Document chunks.

    Unlike AstraDB (a real hosted database), this index lives in memory
    for the lifetime of the process -- it's built from a list of
    Documents you hand it via `build()`. That's the right scope for a
    portfolio project (a handful of papers); a production system would
    back this with a dedicated search engine (Elasticsearch/OpenSearch)
    instead of an in-memory index.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of Document chunks.
        Must be called once before `search()` -- typically right after
        SmartPDFProcessor.process_pdf() produces the chunk list, so the
        sparse index and the dense vector store are built from the exact
        same chunks.
        """
        self._documents = documents
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 8) -> List[Document]:
        """
        Return the top-k chunks whose exact word content best matches
        the query, ranked by BM25 score (highest first).
        """
        if self._bm25 is None:
            raise RuntimeError("SparseStore.build() must be called before search().")

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked: List[Tuple[Document, float]] = sorted(
            zip(self._documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [doc for doc, _score in ranked[:k]]

    def search_with_scores(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """
        Same as `search()`, but also returns each chunk's raw BM25
        score. Needed in Step 3, where we combine BM25 scores with
        dense similarity scores during fusion.
        """
        if self._bm25 is None:
            raise RuntimeError("SparseStore.build() must be called before search().")

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked: List[Tuple[Document, float]] = sorted(
            zip(self._documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:k]

    def is_ready(self) -> bool:
        """Whether build() has been called yet."""
        return self._bm25 is not None