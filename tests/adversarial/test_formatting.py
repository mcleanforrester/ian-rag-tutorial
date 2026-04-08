"""
Scenario 3: Inconsistent formatting

Tests that raw HTML, Markdown, table syntax, and non-ASCII characters flow into
chunks unchanged — the system performs no sanitization.
"""

import re

import pytest
from langchain_core.documents import Document

from ingestion.splitter import split_documents

MESSY_CONTENT = """
<h1>Employee Benefits Overview</h1>
<p>All employees are <strong>entitled</strong> to the following benefits:</p>

## Health Insurance

| Plan  | Deductible | Premium   |
|-------|------------|-----------|
| Basic | $1,000     | $200/mo   |
| Plus  | $500       | $350/mo   |

<ul><li>Dental</li><li>Vision</li><li>401k matching</li></ul>

<!-- HR comment: update premiums before open enrollment -->

Café benefit: employees receive a daily café voucher worth €5.

Reimbursement policy: submit receipts within 30 días of the expense.
""".strip()


@pytest.fixture
def messy_doc():
    return Document(
        page_content=MESSY_CONTENT,
        metadata={"department": "hr", "permission_level": "public"},
    )


def test_messy_content_splits_without_exception(messy_doc):
    """Chunking must not raise on HTML, Markdown, unicode, or mixed encoding."""
    chunks = split_documents([messy_doc])
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_html_tags_survive_into_chunks(messy_doc):
    """Raw HTML tags must appear in chunks unchanged — no sanitization happens."""
    chunks = split_documents([messy_doc])
    all_content = " ".join(c.page_content for c in chunks)
    assert re.search(r"<[a-zA-Z/]", all_content), (
        "Expected raw HTML tags (e.g. <h1>, <p>) in chunks.\n"
        "If this fails, the splitter or loader is stripping HTML — document this."
    )


def test_markdown_artifacts_survive_into_chunks(messy_doc):
    """Markdown headers and table pipe syntax must appear in chunks unchanged."""
    chunks = split_documents([messy_doc])
    all_content = " ".join(c.page_content for c in chunks)
    assert "##" in all_content or "|" in all_content, (
        "Expected Markdown headers (##) or table delimiters (|) in chunks.\n"
        "If this fails, the splitter is stripping Markdown — document this."
    )


def test_unicode_content_survives_into_chunks(messy_doc):
    """Non-ASCII characters (accented, currency symbols) must survive chunking."""
    chunks = split_documents([messy_doc])
    all_content = " ".join(c.page_content for c in chunks)
    assert "€" in all_content or "é" in all_content or "í" in all_content, (
        "Expected unicode characters (€, é, í) in chunks.\n"
        "If this fails, encoding is being mangled during splitting."
    )
