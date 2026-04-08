"""
Scenario 1: Scanned PDF (image-only, no text layer)

Tests whether the system fails silently or loudly when given a PDF
with no extractable text, and what the downstream chunk count looks like.
"""

import pytest
from langchain_community.document_loaders import PyPDFLoader

from ingestion.splitter import split_documents


def test_scanned_pdf_loads_without_exception(image_pdf_path):
    """PyPDFLoader must not raise on an image-only PDF — failure must be silent."""
    loader = PyPDFLoader(str(image_pdf_path))
    docs = loader.load()  # must not raise
    assert isinstance(docs, list)


def test_scanned_pdf_page_content_is_empty(image_pdf_path):
    """All pages must have empty or whitespace-only page_content — no text extracted."""
    loader = PyPDFLoader(str(image_pdf_path))
    docs = loader.load()
    for doc in docs:
        assert doc.page_content.strip() == "", (
            f"Expected empty page_content from image-only PDF, got: {doc.page_content!r}\n"
            "This means text was unexpectedly extracted from the image layer."
        )


def test_scanned_pdf_produces_zero_chunks(image_pdf_path):
    """Empty pages must produce zero chunks — nothing gets indexed."""
    loader = PyPDFLoader(str(image_pdf_path))
    docs = loader.load()
    chunks = split_documents(docs)
    assert len(chunks) == 0, (
        f"Expected 0 chunks from image-only PDF, got {len(chunks)}.\n"
        "The system will silently index empty/whitespace chunks into ChromaDB."
    )
