"""
core/sparse_store.py
----------------------
Sparse (keyword) retrieval with BM25 -- the SPARSE half of retrieval.

Dense search understands meaning; BM25 matches exact words (names,
acronyms, numbers). Together (see core/retrieval.py) they cover each
other's blind spots. Runs fully locally, no API.
"""

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase words only, so "Method," and "method" count as the same word."""
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseStore:
    """An in-memory BM25 index over a list of Document chunks."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._word_sets: List[set] = []  # the set of words in each chunk

    def build(self, documents: List[Document]) -> None:
        """
        Build (or rebuild) the index. Build it from the SAME chunk list
        that you insert into the vector store, so dense and sparse agree.

        An empty list is allowed: the store simply stays "not ready"
        (BM25Okapi itself crashes with ZeroDivisionError on an empty corpus).
        """
        self._documents = list(documents)
        if not self._documents:
            self._bm25 = None
            self._word_sets = []
            return
        tokenized = [_tokenize(doc.page_content) for doc in self._documents]
        self._word_sets = [set(words) for words in tokenized]
        self._bm25 = BM25Okapi(tokenized)

    def search_with_scores(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """
        Top-k chunks with their BM25 scores, best first.

        Only chunks that contain at least ONE word of the query are
        returned. Without this rule, a query that matches nothing would
        still return the first k chunks of the corpus (all tied at 0), and
        those junk chunks would pollute the fusion.

        Why "contains a query word" and not "score > 0"? BM25 gives a word
        zero or NEGATIVE weight when it appears in more than half of the
        chunks, which happens easily in a small collection. A score filter
        would then throw away real matches.
        """
        if self._bm25 is None:
            raise RuntimeError("SparseStore.build() must be called with documents before search().")

        tokens = _tokenize(query)
        if not tokens:
            return []
        query_words = set(tokens)

        scores = self._bm25.get_scores(tokens)
        matches = [
            (doc, float(score))
            for doc, score, words in zip(self._documents, scores, self._word_sets)
            if query_words & words
        ]
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches[:k]

    def search(self, query: str, k: int = 8) -> List[Document]:
        """Same as search_with_scores(), documents only."""
        return [doc for doc, _score in self.search_with_scores(query, k)]

    def paper_ids(self) -> List[str]:
        """
        Names of the papers in this index (from each chunk's paper_id).
        The planner and the Critic use this to know which papers exist.
        """
        names: List[str] = []
        for doc in self._documents:
            paper = doc.metadata.get("paper_id") or doc.metadata.get("source_file")
            if paper and paper not in names:
                names.append(paper)
        return names

    def is_ready(self) -> bool:
        """True once build() has been called with at least one document."""
        return self._bm25 is not None