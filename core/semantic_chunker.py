"""
core/semantic_chunker.py
--------------------------
Embedding-based ("semantic") text chunking.

The problem with fixed-size chunking
--------------------------------------
Splitting text every N characters doesn't know anything about *meaning*.
It's perfectly happy to cut a chunk in half mid-argument, or to lump two
unrelated paragraphs into one chunk just because they happen to fall
inside the same 1000-character window. For research papers -- where a
single passage can shift from "related work" to "our method" -- that
hurts retrieval: a query about "our method" can pull back a chunk that's
mostly about someone else's approach.

What semantic chunking does instead
--------------------------------------
1. Split the page into individual sentences.
2. Embed every sentence (using the SAME embedding model already used
   for the vector store -- no new model, no paid API).
3. Compare each sentence to the one before it with cosine similarity.
   A big drop in similarity usually means the topic just shifted.
4. Cut a new chunk wherever similarity drops below a threshold, instead
   of at a fixed character count.
5. Size guards: no tiny chunks, and NO chunk longer than max_chunk_chars
   (a long stretch of text with no sentence breaks is cut at spaces).
   The limit matters: the embedding model and the reranker both silently
   ignore text beyond their token limit.

This is a from-scratch implementation (not the langchain_experimental
library) so every decision is visible -- useful when you need to explain
exactly how and why a chunk boundary was chosen.
"""

import re
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings


def _split_into_sentences(text: str) -> List[str]:
    """
    A lightweight, regex-based sentence splitter -- good enough for
    already-cleaned PDF text, and avoids pulling in a heavy NLP library
    (nltk/spacy) just for sentence boundaries.

    Splits after '.', '!', or '?' when followed by whitespace and then
    an uppercase letter or digit. Requiring a capital/number afterward
    is a simple heuristic that avoids most false splits on abbreviations
    like "Dr." or "e.g." (which are usually followed by a lowercase word).
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity between two vectors, in [-1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _hard_split(text: str, max_chars: int) -> List[str]:
    """
    Cut a text that is too long into pieces of at most max_chars,
    always at a space (a single word longer than max_chars is cut inside).
    """
    pieces: List[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(word) > max_chars:  # one enormous "word"
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        current = word
    if current:
        pieces.append(current)
    return pieces


class SemanticChunker:
    """
    Groups sentences into chunks based on where their *meaning* shifts,
    instead of cutting at a fixed character count.

    Exposes `.split_text(text) -> List[str]`, the same interface
    LangChain's own text splitters use -- so calling code (see
    core/document_utils.py) can treat this and RecursiveCharacterTextSplitter
    interchangeably without an if/else branch.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        similarity_threshold: float = 0.5,
        min_chunk_chars: int = 300,
        max_chunk_chars: int = 1500,
    ):
        """
        Parameters
        ----------
        embeddings : Embeddings
            Any LangChain-compatible embedding model. We pass in the
            same HuggingFace MiniLM model used for the vector store, so
            no extra model needs to be downloaded or loaded into memory.
        similarity_threshold : float
            Below this cosine similarity between consecutive sentences,
            start a new chunk. Lower = fewer, bigger chunks (only cuts
            on big topic shifts). Higher = more, smaller chunks (cuts
            more eagerly). 0.5 is a reasonable starting point for MiniLM.
        min_chunk_chars : int
            Chunks below this size get merged forward -- avoids tiny,
            low-context fragments that hurt more than they help.
        max_chunk_chars : int
            Chunks above this size get force-cut regardless of
            similarity -- avoids one giant chunk if a long stretch of
            text all reads as "similar."
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def split_text(self, text: str) -> List[str]:
        """Split one block of text (e.g. a PDF page) into chunks."""
        text = text.strip()
        if not text:
            return []

        sentences = _split_into_sentences(text)
        if len(sentences) <= 1:
            return _hard_split(text, self.max_chunk_chars)

        # One batched call is much faster than one call per sentence.
        vectors = [np.array(v) for v in self.embeddings.embed_documents(sentences)]

        chunks: List[str] = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = _cosine_similarity(vectors[i - 1], vectors[i])
            current_length = sum(len(s) for s in current) + len(current) - 1

            topic_changed = (
                similarity < self.similarity_threshold
                and current_length >= self.min_chunk_chars
            )
            would_be_too_long = current_length + 1 + len(sentences[i]) > self.max_chunk_chars

            if topic_changed or would_be_too_long:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        chunks.append(" ".join(current))

        # A single sentence can be longer than the limit: cut it at spaces.
        sized: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_chars:
                sized.extend(_hard_split(chunk, self.max_chunk_chars))
            else:
                sized.append(chunk)

        # A tiny last chunk is merged into the one before it (if it fits).
        if (
            len(sized) >= 2
            and len(sized[-1]) < self.min_chunk_chars
            and len(sized[-2]) + 1 + len(sized[-1]) <= self.max_chunk_chars
        ):
            sized[-2] = f"{sized[-2]} {sized[-1]}"
            sized.pop()

        return sized