# Multi-Tenant RAG with Metadata Filtering

## Problem

The current RAG chatbot loads all company PDFs into a single FAISS index with no metadata. A CompanyA employee's query can return CompanyB's documents. We need tenant isolation and permission-based access control.

## Design

### Folder Structure & Metadata Derivation

```
pdfs/
  companyA/
    pto-policy-internal.pdf
    benefits-public.pdf
  companyB/
    handbook-confidential.pdf
  companyC/
    leave-policy-public.pdf

faiss_index/
  companyA/
  companyB/
  companyC/
```

Metadata is derived from the filesystem:

- `company_id` — parent folder name (e.g., `pdfs/companyA/` -> `"companyA"`)
- `permission_level` — parsed from the filename; must contain one of `public`, `internal`, or `confidential`
- `owner` — hardcoded to `"hr-team"`

Each document chunk gets all three fields attached as LangChain `Document.metadata`.

### Per-Company FAISS Index Storage

On startup, the code scans `pdfs/` for company subdirectories. For each company:

1. Check if `faiss_index/{company_id}/` exists — if so, load it
2. If not, load all PDFs from `pdfs/{company_id}/`, split into chunks, attach metadata, build a FAISS index, and save to `faiss_index/{company_id}/`

This produces a `dict[str, FAISS]` mapping company_id to its FAISS vector store.

```python
vector_stores = {
    "companyA": FAISS(...),
    "companyB": FAISS(...),
    "companyC": FAISS(...),
}
```

Only the current user's index is queried at runtime.

### Simulated User Context & Permission Filtering

A simple user context simulated at the top of the script:

```python
current_user = {
    "company_id": "companyA",
    "permission_level": "internal",  # can see public + internal
    "name": "Alice",
}
```

Permission hierarchy: `public` (0) < `internal` (1) < `confidential` (2)

```python
PERMISSION_RANK = {"public": 0, "internal": 1, "confidential": 2}
```

At query time:

1. Select the FAISS index matching `current_user["company_id"]` — tenant isolation layer
2. Run similarity search against that index
3. Post-filter results: only keep chunks where the chunk's `permission_level` rank is <= the user's rank

A user with `internal` access sees chunks tagged `public` or `internal`, but not `confidential`.

### Integration with the Existing Agent

The `prompt_with_context` middleware changes to:

1. Look up `vector_stores[current_user["company_id"]]` instead of a single global `vector_store`
2. Run `similarity_search` on that company-specific index
3. Filter returned docs by permission level
4. Inject only the filtered docs into the system prompt

The agent, guardrails, extract mode, and REPL loop remain unchanged.

### Error Handling

- If a user's `company_id` doesn't match any loaded index, print a clear error and refuse to answer
- If a PDF filename doesn't contain a recognized permission level (`public`, `internal`, `confidential`), raise an error at index-build time so bad data doesn't silently enter the system

## What Does NOT Change

- LLM provider (Anthropic Claude)
- Embeddings (Ollama nomic-embed-text)
- Text splitting strategy (RecursiveCharacterTextSplitter, 1000 chunk / 200 overlap)
- Agent setup, guardrails, extract mode, REPL loop
