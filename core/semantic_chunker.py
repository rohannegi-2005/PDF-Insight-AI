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
5. Enforce a min/max chunk size so we don't end up with lots of tiny
   one-sentence chunks, or one giant chunk if similarity stays high for
   a very long stretch.

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
        """
        Split a block of text (e.g. one PDF page) into semantically
        coherent chunks.
        """
        sentences = _split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Embed every sentence in ONE batched call -- much faster than
        # calling the model sentence-by-sentence, since the model can
        # process the whole batch together.
        sentence_vectors = self.embeddings.embed_documents(sentences)

        chunks: List[str] = []
        current_chunk_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            prev_vector = np.array(sentence_vectors[i - 1])
            curr_vector = np.array(sentence_vectors[i])
            similarity = _cosine_similarity(prev_vector, curr_vector)

            current_length = sum(len(s) for s in current_chunk_sentences)

            # Cut here if EITHER: the topic just shifted (similarity
            # dropped) AND the chunk-so-far is already a reasonable
            # size, OR the chunk has hit the hard max-size ceiling.
            should_break = (
                similarity < self.similarity_threshold
                and current_length >= self.min_chunk_chars
            ) or current_length >= self.max_chunk_chars

            if should_break:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks