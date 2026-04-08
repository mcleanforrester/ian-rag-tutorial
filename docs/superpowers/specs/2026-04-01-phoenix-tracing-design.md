# Phoenix Tracing Integration Design

## Goal

Add Arize Phoenix (cloud-hosted) tracing to both entry points (`app.py` and `api/server.py`) alongside the existing LangSmith tracing. Both systems run independently — LangSmith via LangChain callbacks, Phoenix via OpenTelemetry auto-instrumentation.

## Approach: Shared Bootstrap Module

A new `bootstrap.py` at the project root centralizes all pre-import startup configuration. Both entry points import it as their first action, before any LangChain imports.

### `bootstrap.py`

Executes two things in order:

1. `load_dotenv(override=True)` — loads `.env` values into the process environment, overriding any stale shell-level vars (known issue with `LANGSMITH_API_KEY` being shadowed).
2. `phoenix.otel.register(project_name="tutorial2", auto_instrument=True)` — registers the OpenTelemetry tracer provider and auto-patches supported libraries (LangChain, Anthropic). The Phoenix cloud endpoint and API key are read from the environment, which is why `load_dotenv` must run first.

### Entry Point Changes

**`app.py`:**
- Remove inline `from dotenv import load_dotenv`, `load_dotenv(override=True)`, `from phoenix.otel import register`, and the `register(...)` call.
- Add `import bootstrap` before any LangChain imports.

**`api/server.py`:**
- Remove inline `from dotenv import load_dotenv` and `load_dotenv(override=True)`.
- Add `import bootstrap` before any LangChain imports.

### Environment Variables (`.env`)

Two new vars for Phoenix cloud, alongside existing LangSmith vars:

- `PHOENIX_COLLECTOR_ENDPOINT` — cloud collector URL (e.g. `https://app.phoenix.arize.com`)
- `PHOENIX_API_KEY` — Phoenix cloud API key

### How Both Tracing Systems Coexist

- **LangSmith**: Activated by `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`. LangChain registers a `LangChainTracer` callback at import time, which sends trace data to the LangSmith API.
- **Phoenix**: Activated by `register(auto_instrument=True)`. This sets up an OpenTelemetry tracer provider that monkey-patches LangChain/Anthropic to emit OTEL spans, exported to the Phoenix cloud collector.

These are independent mechanisms (LangChain callbacks vs. OTEL spans) and do not conflict.
