"""
core/document_utils.py
-----------------------
Document ingestion utilities.

This module owns everything related to turning a raw PDF file into a list
of clean, appropriately-sized `Document` chunks that are ready to be
embedded and stored in the vector database.

Kept 1:1 in spirit with the original `SmartPDFProcessor` from the
single-file prototype -- same loader (PyPDFLoader), same splitter
(RecursiveCharacterTextSplitter) -- just isolated into its own module so
it can be imported by the Streamlit UI (for on-demand uploads) or any
future ingestion script/CLI without dragging in Streamlit or AstraDB code.
"""

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartPDFProcessor:
    """
    Loads a PDF, cleans the extracted text, and splits it into
    metadata-rich chunks suitable for embedding.

    Why a class instead of a function?
    -----------------------------------
    Chunking behaviour (size/overlap) is a *configuration*, not a one-off
    parameter -- wrapping it in a class lets you build one processor with
    a given configuration and reuse it across many PDFs, which is exactly
    what happens in the Streamlit app's "upload & index" flow.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Parameters
        ----------
        chunk_size : int
            Target number of characters per chunk. 1000 chars is a good
            middle ground for MiniLM-based embeddings (max ~256 tokens) --
            large enough to preserve context, small enough to stay well
            under the embedding model's token limit.
        chunk_overlap : int
            Number of overlapping characters between consecutive chunks.
            Overlap prevents a sentence that straddles a chunk boundary
            from losing meaning in either chunk.
        """
        # NOTE: the original prototype accidentally wrote these as 1-tuples
        # (`self.chunk_size = chunk_size,`  <- trailing comma). Fixed here
        # to plain ints -- they weren't read again in the original code,
        # but leaving a latent bug like that in a "production-ready"
        # rewrite is exactly the kind of thing an interviewer will notice.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Splitting only on spaces (rather than the default
            # ["\n\n", "\n", " ", ""] cascade) sidesteps the noisy/
            # inconsistent newline structure PDFs often extract with.
            separators=[" "],
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF from disk and return a list of cleaned, chunked
        `Document` objects, each carrying page-level metadata.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # one Document per PDF page

        processed_chunks: List[Document] = []

        for page_num, page in enumerate(pages):
            cleaned_text = self._clean_text(page.page_content)

            # Skip near-empty pages (cover pages, blank pages, scanned
            # images with no extractable text, etc.)
            if len(cleaned_text.strip()) < 40:
                continue

            chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[{
                    **page.metadata,
                    "page": page_num + 1,
                    "total_pages": len(pages),
                    "chunk_method": "smart_pdf_processor",
                    "char_count": len(cleaned_text),
                    # Track the source filename so citations in the UI
                    # can show *which document* an answer came from once
                    # the app supports indexing more than one PDF.
                    "source_file": pdf_path.split("/")[-1],
                }],
            )
            processed_chunks.extend(chunks)

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
