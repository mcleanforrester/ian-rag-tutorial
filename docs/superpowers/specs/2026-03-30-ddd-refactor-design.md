# Domain-Driven Design Refactor

## Problem

`tutorial.py` is a 199-line monolith mixing identity, ingestion, retrieval, conversation, and validation concerns. We want to refactor into bounded contexts following DDD principles so each domain concept is isolated, testable, and swappable independently.

## Package Structure

```
ragtutorial/
  identity/
    __init__.py
    models.py
    permissions.py
  ingestion/
    __init__.py
    loader.py
    splitter.py
  retrieval/
    __init__.py
    repository.py
    store.py
  conversation/
    __init__.py
    middleware.py
    guards.py
    repl.py
  app.py
  config.py
```

`tutorial.py` is deleted. `app.py` replaces it as the entry point.

## Bounded Contexts

### Identity

Owns the concept of who is asking and what they can see.

**`identity/models.py`:**

- `PermissionLevel` — Frozen dataclass value object with `name`, `rank`, and `can_access(other)` method. Encapsulates the comparison logic so it lives in one place.
- `PUBLIC`, `INTERNAL`, `CONFIDENTIAL` — Module-level constants.
- `LEVELS` — Dict mapping string names to `PermissionLevel` instances.
- `User` — Dataclass entity with `name`, `department`, `permission_level` (a `PermissionLevel` instance).

**`identity/permissions.py`:**

- `parse_permission_level(filename) -> PermissionLevel` — Extracts permission level from a PDF filename using regex word boundary matching. Returns a `PermissionLevel` value object. Raises `ValueError` for unrecognized filenames.

**`identity/__init__.py`** exports: `User`, `PermissionLevel`, `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `LEVELS`, `parse_permission_level`.

### Ingestion

Owns how documents get into the system.

**`ingestion/loader.py`:**

- `load_department_docs(pdf_base: str) -> list[Document]` — Scans `pdfs/{department}/` subdirectories, loads all PDFs with PyPDFLoader, attaches `department` (from folder name), `permission_level` (from filename via `identity.permissions.parse_permission_level()`), and `owner` ("hr-team") metadata to each document. Returns flat list of all documents across all departments.

**`ingestion/splitter.py`:**

- `split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]` — Wraps `RecursiveCharacterTextSplitter`. Preserves metadata through splitting.

**`ingestion/__init__.py`** exports: `load_department_docs`, `split_documents`.

**Cross-context dependency:** Ingestion imports `parse_permission_level` from Identity. This is read-only — Identity owns the concept of permission levels, Ingestion just uses it to tag documents.

### Retrieval

Owns how we find relevant documents for a user.

**`retrieval/store.py`:**

- `init_vector_store(persist_directory: str, embedding_function) -> Chroma` — Creates or loads a Chroma vector store with collection name from config.
- `index_documents(vector_store: Chroma, documents: list[Document]) -> None` — Adds documents to the store if it's empty (checks `_collection.count() == 0`).

**`retrieval/repository.py`:**

- `DocumentRepository` — Repository pattern class. Constructor takes a `Chroma` vector store.
  - `find_relevant(query: str, user: User, k: int = 4) -> list[Document]` — Builds a Chroma `$and`/`$in` filter from the user's department and allowed permission levels (using `user.permission_level.can_access()`), runs similarity search, returns filtered documents. This is the only place in the codebase that knows about Chroma query syntax.

**`retrieval/__init__.py`** exports: `DocumentRepository`, `init_vector_store`, `index_documents`.

**Cross-context dependency:** Repository imports `User` from Identity to type its `find_relevant` method parameter.

### Conversation

Owns how we interact with the user.

**`conversation/guards.py`:**

- `Opening` — Pydantic model for structured extraction.
- `custom_failed_response` — Validation failure handler.
- `input_guard` — ValidLength guard instance.
- `output_guard` — Pydantic extraction guard instance.
- `validate_input(query: str) -> bool` — Returns whether the input passes validation.
- `extract_structured(response: str) -> tuple[bool, Any]` — Attempts structured extraction, returns (success, result_or_error).

**`conversation/middleware.py`:**

- `create_prompt_middleware(repository: DocumentRepository, user: User)` — Factory function that returns a `@dynamic_prompt` decorated function. The middleware calls `repository.find_relevant(query, user)` and builds the system prompt. Uses dependency injection — repository and user are bound at creation time, not read from globals.

**`conversation/repl.py`:**

- `run(agent, user: User, input_guard, output_guard)` — The REPL loop. Handles input, extract mode, streaming, and output. No business logic.

**`conversation/__init__.py`** exports: `create_prompt_middleware`, `run`, `input_guard`, `output_guard`.

### App Wiring

**`config.py`:**

```python
MODEL_NAME = "claude-sonnet-4-6"
MODEL_PROVIDER = "anthropic"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_COLLECTION = "department_docs"
CHROMA_PATH = "chroma_db"
PDF_PATH = "pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

**`app.py`** — Composition root. The only file that imports from multiple contexts:

1. Create `User` (simulated)
2. Init LLM and embeddings
3. Init vector store, ingest documents if empty
4. Create `DocumentRepository`
5. Create agent with prompt middleware
6. Run REPL

No business logic — just assembly.

## Cross-Context Dependencies

```
Identity  <--  Ingestion (reads permission levels)
Identity  <--  Retrieval (User type for filtering)
Retrieval <--  Conversation (DocumentRepository for middleware)
Identity  <--  Conversation (User type for REPL display)
Config    <--  All (shared constants)
```

All dependencies flow one direction. No circular imports.

## What Does NOT Change

- LLM provider, embedding model, text splitting params
- Chroma as the vector store
- Permission hierarchy logic (just moved)
- Guardrails behavior
- REPL behavior and extract mode
- `generate_docs.py` (standalone, not part of the domain)
- PDF folder structure
