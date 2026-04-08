"""
Scenario 2: Very long document (120 pages)

Tests chunking stability, retrieval k-cap, and chunk provenance on large inputs.
"""

import pytest
from langchain_community.document_loaders import PyPDFLoader

from ingestion.splitter import split_documents


@pytest.fixture(scope="module")
def long_pdf_chunks(long_pdf_path):
    """Load and split the long PDF once per module — loading 120 pages is slow."""
    loader = PyPDFLoader(str(long_pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["department"] = "hr"
        doc.metadata["permission_level"] = "public"
    return split_documents(docs)


def test_long_pdf_chunks_without_error(long_pdf_chunks):
    """Chunking a 120-page doc must not raise and must produce a meaningful number of chunks."""
    assert len(long_pdf_chunks) > 500, (
        f"Expected > 500 chunks from a 120-page PDF, got {len(long_pdf_chunks)}.\n"
        "chunk_size=1000 with 50 lines/page × 120 pages should exceed 500 chunks."
    )


def test_long_pdf_retrieval_returns_exactly_k(long_pdf_chunks, make_store, hr_user):
    """Retrieval must return exactly k=4 results regardless of how large the index is."""
    repo = make_store(long_pdf_chunks)
    results = repo.find_relevant("PTO accrual benefits policy", hr_user, k=4)
    assert len(results) == 4, (
        f"Expected exactly 4 results from find_relevant(k=4), got {len(results)}."
    )


def test_long_pdf_chunks_preserve_start_index(long_pdf_chunks):
    """Every chunk must carry a start_index for source provenance."""
    for chunk in long_pdf_chunks[:20]:  # spot-check first 20
        assert "start_index" in chunk.metadata, (
            f"Chunk missing start_index metadata: {chunk.metadata}\n"
            "Provenance tracking will be broken for this chunk."
        )
