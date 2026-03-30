# Single-Index Chroma with Query-Time Filtering

## Problem

The current multi-tenant implementation uses separate FAISS indexes per company. We want to switch to a single shared Chroma vector store with query-time metadata filtering, which is the standard pattern for multi-tenant RAG and demonstrates native filtering capabilities.

## Design

### Single Chroma Index with Metadata

Replace the per-company FAISS dict (`vector_stores = {}`) with a single Chroma vector store. All documents from all companies go into one collection, with `company_id`, `permission_level`, and `owner` metadata on every chunk.

```python
vector_store = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
```

The PDF folder structure (`pdfs/companyA/`, etc.) and metadata derivation logic stay exactly the same — `company_id` from folder name, `permission_level` from filename via `parse_permission_level()`, `owner` hardcoded to `"hr-team"`.

The `faiss_index/` directory and `vector_stores` dict are removed entirely.

### Query-Time Filtering

At query time in `prompt_with_context`, use Chroma's native `filter` parameter for both tenant isolation and permission filtering in a single call:

```python
user_rank = PERMISSION_RANK[current_user["permission_level"]]
allowed_levels = [level for level, rank in PERMISSION_RANK.items() if rank <= user_rank]

retrieved_docs = vector_store.similarity_search(
    last_query,
    filter={
        "$and": [
            {"company_id": current_user["company_id"]},
            {"permission_level": {"$in": allowed_levels}},
        ]
    },
)
```

Chroma applies filters before selecting top-k results, so you always get the full number of results back. No post-retrieval filtering needed.

### Dependency Changes

- **Add:** `langchain-chroma`, `chromadb`
- **Remove:** `from langchain_community.vectorstores import FAISS`
- **Add:** `from langchain_chroma import Chroma`
- **Keep:** Ollama embeddings, PyPDFLoader, RecursiveCharacterTextSplitter, guardrails, agent, REPL loop

### Error Handling

- If `current_user["company_id"]` has no documents in the store, the similarity search returns an empty list — the LLM will say "I don't know." No special handling needed.
- Bad filenames still raise `ValueError` at build time via `parse_permission_level()`.

### Index Loading Strategy

On startup, check if the Chroma persist directory exists and has data. If so, load it. If not, scan `pdfs/` subdirectories, load and split all PDFs, attach metadata, and add them to a new Chroma collection.

## What Changes from Previous Implementation

- `vector_stores` dict (per-company) -> single `vector_store` (Chroma)
- `faiss_index/` directory -> `chroma_db/` directory
- FAISS imports -> Chroma imports
- Post-retrieval permission filtering -> Chroma native `$and`/`$in` filter at query time
- Company-not-found error handling in middleware -> removed (empty results handled naturally)

## What Does NOT Change

- PDF folder structure (`pdfs/companyA/`, etc.)
- Metadata derivation logic (`company_id`, `permission_level`, `owner`)
- `PERMISSION_RANK`, `current_user`, `parse_permission_level()`
- LLM provider (Anthropic Claude)
- Embeddings (Ollama nomic-embed-text)
- Text splitting strategy (RecursiveCharacterTextSplitter, 1000 chunk / 200 overlap)
- Agent setup, guardrails, extract mode, REPL loop, startup message
