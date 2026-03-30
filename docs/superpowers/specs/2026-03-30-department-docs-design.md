# Department Document Generation & Rename

## Problem

The tutorial uses three placeholder company PDFs (companyA/B/C) with one doc each. We need realistic consulting firm content across three departments (engineering, accounting, hr) with 20 documents each, and rename `company_id` to `department` throughout.

## Design

### Generation Script (`generate_docs.py`)

A standalone two-phase script:

**Phase 1 — Generate content fixture:** Call Claude API to generate `docs_fixture.json` containing 60 document definitions — 20 per department (engineering, accounting, hr). Each entry has `title`, `department`, `permission_level`, and `content` (1-2 pages of realistic consulting firm text). Permission mix per department: 10 public, 7 internal, 3 confidential. If `docs_fixture.json` already exists, this phase is skipped.

**Phase 2 — Convert to PDFs:** Read the fixture, use `fpdf2` to render each entry as a PDF into `pdfs/{department}/`. Filename format: `{slugified-title}-{permission_level}.pdf`. Max 2 pages per document.

### Document Content by Department

- **Engineering:** Coding standards, architecture decision records, project specs, deployment procedures, tech stack guidelines, incident response playbooks, code review policies, sprint retrospective templates, etc.
- **Accounting:** Expense report policies, invoice procedures, budget templates, audit procedures, financial reporting guidelines, vendor payment processes, travel reimbursement rules, etc.
- **HR:** PTO policies, onboarding checklists, performance review procedures, benefits guides, termination procedures, employee handbook sections, interview guidelines, compensation philosophy, etc.

### Permission Level Distribution (per department)

- 10 public (general knowledge, available to all)
- 7 internal (department-specific, not for external sharing)
- 3 confidential (sensitive — salary data, termination procedures, audit findings)

### Rename `company_id` to `department`

All references in `tutorial.py` change:

- `current_user["company_id"]` -> `current_user["department"]`
- `doc.metadata["company_id"]` -> `doc.metadata["department"]`
- Chroma filter: `{"company_id": ...}` -> `{"department": ...}`
- `company_dirs` variable -> `department_dirs`
- Startup message: `Logged in as Alice (engineering, internal access)`

```python
current_user = {
    "department": "engineering",
    "permission_level": "internal",
    "name": "Alice",
}
```

### Folder Structure After Generation

```
pdfs/
  engineering/   (20 PDFs)
  accounting/    (20 PDFs)
  hr/            (20 PDFs)

docs_fixture.json   # cached LLM output
```

Old `pdfs/companyA/`, `companyB/`, `companyC/` folders are deleted.

### Dependencies

- **Add:** `fpdf2` (PDF generation), `anthropic` (Claude API for content generation)
- `generate_docs.py` is standalone — `tutorial.py` does not import it

### Fixture Schema

```json
[
  {
    "title": "Coding Standards Guide",
    "department": "engineering",
    "permission_level": "internal",
    "content": "..."
  }
]
```

## What Changes in `tutorial.py`

- `company_id` -> `department` in metadata, user context, Chroma filter, variable names
- Startup message wording

## What Does NOT Change

- Chroma vector store setup, query-time `$and`/`$in` filtering pattern
- `PERMISSION_RANK`, `parse_permission_level()`
- `owner` metadata (still "hr-team")
- Agent, guardrails, extract mode, REPL loop
- Embeddings, text splitting params
