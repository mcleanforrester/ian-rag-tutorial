# Streamlit Frontend with FastAPI Backend

## Problem

The RAG chatbot is currently CLI-only (`app.py` REPL). We need a web-based chat interface with user switching to demonstrate multi-tenant filtering visually.

## Architecture

Two separate processes communicating over HTTP:

```
[Streamlit App :8501]  --HTTP-->  [FastAPI Server :8000]
     frontend/app.py                  api/server.py
     - chat UI                        - POST /chat
     - user dropdown                  - GET /users
     - session state                  - wires bounded contexts
```

Streamlit handles only UI and session state. FastAPI handles all domain logic by importing from the existing bounded contexts. The CLI (`app.py`, `conversation/repl.py`) remains functional.

## FastAPI Server (`api/`)

### `api/users.py`

Predefined user list:

| ID | Name | Department | Permission Level |
|----|------|-----------|-----------------|
| alice | Alice | engineering | internal |
| bob | Bob | engineering | confidential |
| carol | Carol | accounting | internal |
| dave | Dave | hr | confidential |
| eve | Eve | accounting | public |

Exported as a `USERS` dict mapping user ID strings to `User` instances.

### `api/server.py`

**Startup:** Initializes shared resources once — `load_dotenv()`, LLM model, embeddings, vector store (load or build), `DocumentRepository`.

**Endpoints:**

- `GET /users` — Returns list of `{id, name, department, permission_level}` for populating the frontend dropdown.
- `POST /chat` — Accepts `{query: str, user_id: str}`. Looks up user from `USERS`, calls `DocumentRepository.find_relevant()` to get context, invokes the LLM with the system prompt + context + query, runs `extract_structured()` on the response, returns `{response: str, policy_summary: {...} | null}`.

The `/chat` endpoint does not use the LangChain agent/middleware pattern — it makes a direct LLM call with the retrieved context injected into the system prompt. This avoids the complexity of per-request agent creation and streaming middleware.

## Streamlit Frontend (`frontend/`)

### `frontend/app.py`

Single file with:

**Sidebar:**
- User dropdown populated from `GET /users` on load
- Shows current department and permission level below the dropdown
- Switching users clears `st.session_state.messages`

**Main area:**
- Chat interface using `st.chat_message` components
- Messages stored in `st.session_state.messages` (list of `{role, content}` dicts)
- User input via `st.chat_input`
- On submit: `POST /chat`, display response as assistant message
- If `policy_summary` is returned, render it in a `st.expander` below the response

No streaming — response comes back complete from FastAPI.

## Dependencies

- **Add:** `streamlit`, `fastapi`, `uvicorn`, `httpx`
- **Keep:** All existing dependencies unchanged

## Running

```bash
# Terminal 1: API server
uvicorn api.server:app --reload

# Terminal 2: Streamlit frontend
streamlit run frontend/app.py
```

## What Does NOT Change

- CLI entry point (`app.py`, `conversation/repl.py`) — still works
- All bounded contexts (identity, ingestion, retrieval, conversation)
- Vector store, embeddings, document generation
- `config.py`
