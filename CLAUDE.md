# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAG (Retrieval-Augmented Generation) application using LangChain. Loads PDFs from `./pdfs/`, chunks and indexes them in ChromaDB, and answers employee queries with department-scoped document retrieval. Has both a CLI REPL (`app.py`) and a FastAPI server (`api/server.py`) with a frontend.

## Stack

- **LLM**: Anthropic Claude (`claude-sonnet-4-6` via `langchain.chat_models.init_chat_model`)
- **Embeddings**: Ollama with `nomic-embed-text` (requires local Ollama server)
- **Vector Store**: ChromaDB (persisted to `./chroma_db/`)
- **Document Loading**: `PyPDFLoader` — loads all PDFs from `./pdfs/`, organized by department
- **API**: FastAPI with SSE streaming (`api/server.py`)
- **Frontend**: `frontend/app.py`
- **Guards**: Guardrails AI for input validation and structured extraction (`conversation/guards.py`)
- **Observability**: LangSmith tracing + Arize Phoenix (OTEL) — configured via `bootstrap.py`

## Project Structure

- `bootstrap.py` — shared startup: loads `.env`, registers Phoenix OTEL tracing. Imported first by both entry points.
- `config.py` — model names, chunk sizes, paths
- `app.py` — CLI REPL entry point
- `api/server.py` — FastAPI server entry point (`/chat`, `/chat/stream`, `/users`)
- `identity/` — user models and permission levels
- `ingestion/` — PDF loading and text splitting
- `retrieval/` — ChromaDB vector store and document repository
- `conversation/` — prompt middleware, input guards, REPL

## Running

```bash
# Requires Ollama running locally
ollama pull nomic-embed-text

# CLI mode
uv run python app.py

# API server
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Dependencies

Managed via `pyproject.toml` + `uv`. Install with `uv sync`.

## Environment

API keys and tracing config are in `.env` (loaded by `bootstrap.py` with `override=True`). See `.env` for required variables: `ANTHROPIC_API_KEY`, `LANGSMITH_*`, `PHOENIX_*`, `GUARDRAILS_API_KEY`.
