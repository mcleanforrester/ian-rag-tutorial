import hashlib
import uuid

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from identity.models import User, CONFIDENTIAL
from retrieval.repository import DocumentRepository
from retrieval.store import index_documents


class MockEmbeddings(Embeddings):
    """Deterministic character-trigram embeddings. Never calls Ollama."""

    DIM = 128

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for i in range(max(0, len(text) - 2)):
            trigram = text[i : i + 3].encode()
            idx = int(hashlib.md5(trigram).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else [1.0 / self.DIM**0.5] * self.DIM


@pytest.fixture(scope="session")
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def make_store(mock_embeddings):
    """Factory: pass a list of Documents, get back a DocumentRepository backed by
    an ephemeral (in-memory) Chroma instance. Each call gets its own isolated store."""

    def _make(docs: list[Document]) -> DocumentRepository:
        client = chromadb.EphemeralClient()
        store = Chroma(
            client=client,
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            embedding_function=mock_embeddings,
        )
        if docs:
            index_documents(store, docs)
        return DocumentRepository(store)

    return _make


@pytest.fixture
def hr_user():
    """A CONFIDENTIAL-level HR user — can access all permission levels in 'hr'."""
    return User(name="Test", department="hr", permission_level=CONFIDENTIAL)


def hr_doc(content: str, source: str = "test.pdf", **extra_meta) -> Document:
    """Helper: create a Document with valid HR metadata for retrieval."""
    meta = {
        "department": "hr",
        "permission_level": "public",
        "source": source,
        **extra_meta,
    }
    return Document(page_content=content, metadata=meta)


@pytest.fixture(scope="session")
def image_pdf_path(tmp_path_factory):
    """A PDF with only a gray rectangle — no text layer (simulates a scanned doc)."""
    from reportlab.pdfgen import canvas as rl_canvas

    path = tmp_path_factory.mktemp("pdfs") / "scanned.pdf"
    c = rl_canvas.Canvas(str(path))
    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.rect(72, 72, 468, 648, fill=1)  # gray box, no text
    c.showPage()
    c.save()
    return path


@pytest.fixture(scope="session")
def long_pdf_path(tmp_path_factory):
    """A 120-page PDF with HR boilerplate — ~50 lines per page."""
    from reportlab.pdfgen import canvas as rl_canvas

    path = tmp_path_factory.mktemp("pdfs") / "long_doc.pdf"
    c = rl_canvas.Canvas(str(path))
    line = (
        "Employee PTO and benefits policy: accrual rates, eligibility, "
        "and HR procedures are outlined here for all staff members. "
    )
    for page_num in range(120):
        c.setFont("Helvetica", 10)
        for row in range(50):
            c.drawString(
                72,
                750 - row * 14,
                f"[P{page_num + 1:03d}R{row + 1:02d}] {line}",
            )
        c.showPage()
    c.save()
    return path
