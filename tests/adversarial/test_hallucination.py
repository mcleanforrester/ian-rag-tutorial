"""
Scenario 6: Confidently wrong answers

Tests that the retrieval layer surfaces contradictory facts with no resolution —
the attack vector for the model producing confident but wrong answers.

We don't call the model here. We assert that the context string passed to the
model contains both conflicting values, proving retrieval is the failure point.
"""

import pytest
from langchain_core.documents import Document

from tests.adversarial.conftest import hr_doc

# Two policy docs with directly contradictory PTO numbers
DOC_15_DAYS = hr_doc(
    content=(
        "PTO Accrual Policy (2022): All full-time employees accrue 15 paid time off "
        "days per year. Part-time employees accrue PTO at a pro-rated rate. "
        "PTO begins accruing from the employee's first day of employment."
    ),
    source="pto-policy-2022.pdf",
)

DOC_25_DAYS = hr_doc(
    content=(
        "PTO Accrual Policy (2024): All full-time employees accrue 25 paid time off "
        "days per year. Part-time employees accrue PTO at a pro-rated rate. "
        "PTO begins accruing from the employee's first day of employment."
    ),
    source="pto-policy-2024.pdf",
)


def test_contradictory_docs_both_retrieved(make_store, hr_user):
    """Both contradictory policy versions must appear in k=4 results."""
    repo = make_store([DOC_15_DAYS, DOC_25_DAYS])
    results = repo.find_relevant("how many PTO days do I accrue per year", hr_user, k=4)
    sources = {r.metadata["source"] for r in results}
    assert "pto-policy-2022.pdf" in sources, "Older (15-day) policy not retrieved"
    assert "pto-policy-2024.pdf" in sources, "Newer (25-day) policy not retrieved"


def test_context_sent_to_model_contains_both_conflicting_numbers(make_store, hr_user):
    """The docs_content string built from retrieval must contain both '15' and '25'.

    This documents the hallucination attack vector: the model receives contradictory
    context with no signal about which version is authoritative, and may confidently
    cite either number (or blend them) depending on which chunk ranks higher.
    """
    repo = make_store([DOC_15_DAYS, DOC_25_DAYS])
    results = repo.find_relevant("how many PTO days do I accrue per year", hr_user, k=4)

    # Replicate the docs_content construction from api/server.py
    docs_content = "\n\n".join(doc.page_content for doc in results)

    assert "15" in docs_content, (
        "15-day policy not present in context sent to model.\n"
        "Retrieval may be suppressing one version of the contradictory pair."
    )
    assert "25" in docs_content, (
        "25-day policy not present in context sent to model.\n"
        "Retrieval may be suppressing one version of the contradictory pair."
    )


def test_no_resolution_signal_in_context(make_store, hr_user):
    """The context must not contain any authoritativeness signal (e.g., 'supersedes',
    'latest version') — the model has no way to resolve the conflict correctly.

    This test passes if both conflicting chunks are present AND no resolution hint
    exists, documenting that the system is blind to document versioning.
    """
    repo = make_store([DOC_15_DAYS, DOC_25_DAYS])
    results = repo.find_relevant("how many PTO days do I accrue per year", hr_user, k=4)
    docs_content = "\n\n".join(doc.page_content for doc in results)

    resolution_signals = ["supersedes", "replaces", "latest", "current version", "deprecated"]
    found_signals = [s for s in resolution_signals if s.lower() in docs_content.lower()]

    assert not found_signals, (
        f"Unexpected resolution signals found in context: {found_signals}\n"
        "If these are present, the model may be able to resolve the conflict.\n"
        "Update test docs to remove them if intentional."
    )
