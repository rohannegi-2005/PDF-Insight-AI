"""
core/document_utils.py
-----------------------
Document ingestion utilities.

This module turns a raw PDF file into a list of clean, metadata-rich
`Document` chunks ready to be embedded and stored in the vector database.

Supports two chunking strategies, chosen via config.CHUNKING_STRATEGY
(or overridden per-instance):
    - "fixed"    : RecursiveCharacterTextSplitter -- splits every N characters
    - "semantic" : SemanticChunker (core/semantic_chunker.py) -- splits where
                   the topic actually shifts, using sentence embeddings

Both strategies are kept (rather than deleting the old one) so we can
directly compare them later in the evaluation harness -- "semantic
chunking improved Recall@5 by X%" is a much stronger claim than just
asserting it's better.
"""

import hashlib
import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from core.semantic_chunker import SemanticChunker


class SmartPDFProcessor:
    """
    Loads a PDF, cleans the extracted text, and splits it into
    metadata-rich chunks suitable for embedding.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        chunk_method: Optional[str] = None,
        chunk_size: int = config.FIXED_CHUNK_SIZE,
        chunk_overlap: int = config.FIXED_CHUNK_OVERLAP,
    ):
        """
        Parameters
        ----------
        embeddings : Embeddings, optional
            Required only when chunk_method == "semantic" -- the
            embedding model used to compare sentence-to-sentence
            similarity. Reuse the same model as the vector store
            (get_embedding_model() in core/vector_store.py).
        chunk_method : str, optional
            "fixed" or "semantic". Defaults to config.CHUNKING_STRATEGY
            if not given.
        chunk_size, chunk_overlap : int
            Only used when chunk_method == "fixed".
        """
        self.chunk_method = chunk_method or config.CHUNKING_STRATEGY

        # Build only the splitter we're actually going to use.
        if self.chunk_method == "semantic":
            if embeddings is None:
                raise ValueError(
                    "chunk_method='semantic' requires an embeddings model. "
                    "Pass one in: SmartPDFProcessor(embeddings=get_embedding_model())."
                )
            self.splitter = SemanticChunker(
                embeddings=embeddings,
                similarity_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
                min_chunk_chars=config.SEMANTIC_MIN_CHUNK_CHARS,
                max_chunk_chars=config.SEMANTIC_MAX_CHUNK_CHARS,
            )
        elif self.chunk_method == "fixed":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[" "],
            )
        else:
            raise ValueError(f"Unknown chunk_method: {self.chunk_method!r}")

    def process_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> List[Document]:
        """
        Load a PDF from disk and return a list of cleaned, chunked
        `Document` objects, each carrying page-level and paper-level
        metadata.

        `paper_id` lets multiple papers share one vector store without
        their chunks being confused for each other -- needed once the
        Research Agent starts comparing evidence ACROSS several papers.
        Defaults to the filename if not given.
        """
        source_file = os.path.basename(pdf_path)
        paper_id = paper_id or source_file

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        processed_chunks: List[Document] = []

        for page_num, page in enumerate(pages):
            cleaned_text = self._clean_text(page.page_content)

            if len(cleaned_text.strip()) < 40:
                continue

            # Both RecursiveCharacterTextSplitter and SemanticChunker
            # expose the same `.split_text(text) -> List[str]` interface,
            # so this one line works no matter which strategy is active --
            # process_pdf doesn't need an if/else here.
            chunk_texts = self.splitter.split_text(cleaned_text)

            for chunk_text in chunk_texts:
                # A stable, deterministic ID for this exact chunk (same
                # source file + page + text always produces the same ID).
                # Needed in core/retrieval.py to recognize "this is the
                # same chunk" when it shows up in BOTH the dense results
                # and the sparse (BM25) results, so fusion can combine
                # their scores correctly instead of treating them as two
                # different chunks.
                chunk_id = hashlib.md5(
                    f"{source_file}|{page_num + 1}|{chunk_text}".encode("utf-8")
                ).hexdigest()[:16]

                processed_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **page.metadata,
                            "page": page_num + 1,
                            "total_pages": len(pages),
                            "chunk_method": self.chunk_method,
                            "char_count": len(chunk_text),
                            "source_file": source_file,
                            "paper_id": paper_id,
                            "chunk_id": chunk_id,
                        },
                    )
                )

        return processed_chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light-touch text normalisation:
        - collapse repeated whitespace introduced by PDF text extraction
        - fix common ligature artefacts (ﬁ -> fi, ﬂ -> fl)
        - strip a known boilerplate phrase ("Scan to Download") that
          leaks into the extracted text of some e-book PDFs
        """
        text = " ".join(text.split())
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("Scan to Download", "").strip()
        return text