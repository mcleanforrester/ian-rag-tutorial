"""
Scenario 4: Near-duplicate documents

Tests whether retrieval surfaces both versions of a near-duplicate document or
lets one version monopolize the top-k slots.
"""

import pytest
from langchain_core.documents import Document

from tests.adversarial.conftest import hr_doc

# Two near-identical PTO policies — same structure, different accrual numbers
_PTO_BODY = (
    "New employees must complete a 90-day probationary period before taking PTO. "
    "PTO must be approved by your manager at least two weeks in advance. "
    "Unused PTO rolls over up to a maximum of 10 days per year. "
    "PTO requests submitted after the deadline may be denied at manager discretion. "
)

DOC_V1 = hr_doc(
    "PTO Policy v1: Employees accrue fifteen paid time off days per year. " + _PTO_BODY,
    source="pto-policy-v1.pdf",
)
DOC_V2 = hr_doc(
    "PTO Policy v2: Employees accrue twenty paid time off days per year. " + _PTO_BODY,
    source="pto-policy-v2.pdf",
)

# Two unrelated docs to pad the index — expense report policy
_EXPENSE_BODY = (
    "Submit all expense reports within 30 days using form EXP-001. "
    "Attach original receipts for all items over $25. "
    "Manager approval is required for expenses exceeding $500. "
    "Reimbursement is processed in the next payroll cycle after approval. "
)
DOC_OTHER_1 = hr_doc("Expense Policy A: " + _EXPENSE_BODY, source="expense-a.pdf")
DOC_OTHER_2 = hr_doc("Expense Policy B: " + _EXPENSE_BODY, source="expense-b.pdf")


def test_both_near_dup_versions_retrieved(make_store, hr_user):
    """Both near-duplicate versions must appear in k=4 results for a PTO query."""
    repo = make_store([DOC_V1, DOC_V2, DOC_OTHER_1, DOC_OTHER_2])
    results = repo.find_relevant(
        "how many PTO days do employees accrue per year", hr_user, k=4
    )
    sources = {r.metadata["source"] for r in results}
    assert "pto-policy-v1.pdf" in sources, (
        "Near-duplicate v1 (15-day policy) was not retrieved.\n"
        "The answer may be based solely on v2, hiding the version conflict."
    )
    assert "pto-policy-v2.pdf" in sources, (
        "Near-duplicate v2 (20-day policy) was not retrieved.\n"
        "The answer may be based solely on v1, hiding the version conflict."
    )


def test_near_dup_retrieval_does_not_exclude_distinct_docs(make_store, hr_user):
    """With only 4 total docs (2 near-dups + 2 others), all 4 must be returned at k=4.
    This establishes the baseline: near-dups don't suppress other results when k fits all."""
    repo = make_store([DOC_V1, DOC_V2, DOC_OTHER_1, DOC_OTHER_2])
    results = repo.find_relevant(
        "how many PTO days do employees accrue per year", hr_user, k=4
    )
    assert len(results) == 4, (
        f"Expected 4 results (all docs), got {len(results)}.\n"
        "Near-duplicate filtering or a retrieval bug may be dropping valid documents."
    )
