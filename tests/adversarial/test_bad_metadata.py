"""
Scenario 5: Missing or wrong metadata

Tests that malformed metadata causes silent filtering rather than crashes,
and that no malformed-metadata doc leaks to a valid user query.
"""

import pytest
from langchain_core.documents import Document

from tests.adversarial.conftest import hr_doc


def test_missing_department_key_does_not_crash(make_store, hr_user):
    """A doc with no 'department' key must not cause find_relevant to raise."""
    docs = [
        hr_doc("Valid HR doc about PTO policy and benefits."),
        Document(
            page_content="No department key at all — metadata incomplete.",
            metadata={"permission_level": "public"},  # 'department' absent
        ),
    ]
    repo = make_store(docs)
    results = repo.find_relevant("PTO policy", hr_user)
    assert isinstance(results, list), "find_relevant must return a list, not raise"


def test_missing_department_doc_is_not_surfaced(make_store, hr_user):
    """A doc missing 'department' must be filtered out, not returned to the user."""
    docs = [
        Document(
            page_content="LEAK: confidential data with no department tag",
            metadata={"permission_level": "public"},
        ),
    ]
    repo = make_store(docs)
    results = repo.find_relevant("confidential data", hr_user)
    for doc in results:
        assert "LEAK" not in doc.page_content, (
            "A doc missing the 'department' key was returned to the user.\n"
            "Metadata filter is not blocking documents with absent department."
        )


def test_typo_department_key_is_not_surfaced(make_store, hr_user):
    """A doc with a key typo ('deparment' instead of 'department') must not be returned."""
    docs = [
        Document(
            page_content="LEAK: doc with typo in department key",
            metadata={"deparment": "hr", "permission_level": "public"},  # typo
        ),
    ]
    repo = make_store(docs)
    results = repo.find_relevant("department policy", hr_user)
    for doc in results:
        assert "LEAK" not in doc.page_content, (
            "A doc with a 'deparment' typo key was returned to the user.\n"
            "Only exact 'department' key match should pass the filter."
        )


def test_wrong_department_doc_is_not_surfaced(make_store, hr_user):
    """A doc from a different department must never appear in an hr_user's results."""
    docs = [
        Document(
            page_content="LEAK: accounting-only confidential billing rates",
            metadata={"department": "accounting", "permission_level": "public"},
        ),
        hr_doc("Normal HR PTO policy for all employees."),
    ]
    repo = make_store(docs)
    results = repo.find_relevant("billing rates policy", hr_user)
    for doc in results:
        assert "LEAK" not in doc.page_content, (
            "A doc from 'accounting' was returned to an 'hr' user.\n"
            "Department-scoped filter is not working correctly."
        )
