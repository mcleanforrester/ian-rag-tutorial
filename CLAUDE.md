# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file RAG (Retrieval-Augmented Generation) tutorial using LangChain. The script (`tutorial.py`) loads a web page, chunks it, stores embeddings in an in-memory vector store, and uses a LangChain agent to answer queries by retrieving relevant context.

## Stack

- **LLM**: Anthropic Claude (via `langchain.chat_models.init_chat_model`)
- **Embeddings**: Ollama with Llama 3 (requires local Ollama server running with `llama3` model pulled)
- **Vector Store**: LangChain `InMemoryVectorStore`
- **Document Loading**: `PyPDFLoader` — auto-loads all PDFs from `./pdfs/`
- **Agent**: LangChain `create_agent` with a custom retrieval tool

## Running

```bash
# Requires Ollama running locally with llama3 model
ollama pull llama3
python tutorial.py
```

## Dependencies

Uses `langchain`, `langchain-community`, `langchain-ollama`, `langchain-text-splitters`, `pypdf`. No requirements file exists yet — install manually.

## Security Note

The API key on line 12 of `tutorial.py` is hardcoded. It should be moved to an environment variable or `.env` file.
