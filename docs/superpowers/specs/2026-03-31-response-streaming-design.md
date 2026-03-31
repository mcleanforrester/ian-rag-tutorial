# Response Streaming Design

## Goal

Sub-second time to first token by streaming LLM responses via SSE from the API to the Streamlit frontend.

## Server Changes (`api/server.py`)

New `POST /chat/stream` endpoint alongside existing `/chat`:

- Retrieval and system prompt construction remain unchanged (local vector store, already fast)
- Use `model.stream()` instead of `model.invoke()` to yield token chunks
- Return `EventSourceResponse` (from `sse-starlette`) with content type `text/event-stream`
- SSE event protocol:
  - `event: token` / `data: <chunk text>` — each streamed token
  - `event: summary` / `data: <json policy summary>` — after extraction on accumulated text
  - `event: done` / `data: ` — signals end of stream
- Policy extraction runs after all tokens are accumulated, before the `done` event

## Frontend Changes (`frontend/app.py`)

- Chat input handler calls `/chat/stream` instead of `/chat`
- Parse SSE events from httpx streaming response (line-based parsing, no new dependency)
- Feed token events into `st.write_stream()` for incremental rendering
- After stream ends, render policy summary expander from the `summary` event if present
- Chat history storage unchanged — still stores full response text and policy summary

## New Dependencies

- `sse-starlette` — server-side SSE support for FastAPI

## What Stays the Same

- `/chat` endpoint (backward compat)
- `/users` endpoint
- All retrieval, ingestion, and guard logic
- Sidebar user selection and chat history management
