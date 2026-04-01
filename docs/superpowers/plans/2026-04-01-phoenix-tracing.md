# Phoenix Tracing Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Arize Phoenix cloud tracing to both entry points via a shared bootstrap module, alongside existing LangSmith tracing.

**Architecture:** A new `bootstrap.py` at the project root runs `load_dotenv(override=True)` then `phoenix.otel.register(...)` before any LangChain imports. Both `app.py` and `api/server.py` import this module as their first action.

**Tech Stack:** arize-phoenix-otel (already installed), phoenix.otel, python-dotenv

---

## File Structure

- **Create:** `bootstrap.py` — shared startup module (dotenv + Phoenix registration)
- **Modify:** `app.py:1-11` — remove inline dotenv/Phoenix setup, add `import bootstrap`
- **Modify:** `api/server.py:1-6` — remove inline dotenv setup, add `import bootstrap`

---

### Task 1: Create `bootstrap.py`

**Files:**
- Create: `bootstrap.py`

- [ ] **Step 1: Create `bootstrap.py` with dotenv and Phoenix registration**

```python
from dotenv import load_dotenv

load_dotenv(override=True)

from phoenix.otel import register

register(project_name="tutorial2", auto_instrument=True)
```

Key ordering: `load_dotenv` must run before `register()` so that `PHOENIX_API_KEY` and `PHOENIX_COLLECTOR_ENDPOINT` are in the environment when Phoenix reads them. The `register()` import is after `load_dotenv()` because `phoenix.otel` may read env vars at import time.

- [ ] **Step 2: Verify bootstrap loads without error**

Run: `uv run python -c "import bootstrap; print('bootstrap OK')"`
Expected: `bootstrap OK` (plus any Phoenix registration output)

- [ ] **Step 3: Commit**

```bash
git add bootstrap.py
git commit -m "feat: add shared bootstrap module for dotenv and Phoenix tracing"
```

---

### Task 2: Update `app.py` to use bootstrap

**Files:**
- Modify: `app.py:1-11`

- [ ] **Step 1: Replace the top of `app.py`**

Current lines 1-11:
```python
import os
import warnings
from dotenv import load_dotenv
from phoenix.otel import register

tracer_provider = register(
  project_name="tutorial2",
  auto_instrument=True
)

load_dotenv(override=True)
```

Replace with:
```python
import bootstrap  # noqa: F401 — must run before any LangChain imports

import os
import warnings
```

This removes the inline `load_dotenv`, `register()`, and the `from dotenv import load_dotenv` / `from phoenix.otel import register` imports. The `bootstrap` import handles all of it.

- [ ] **Step 2: Verify app.py imports cleanly**

Run: `uv run python -c "import app; print('app OK')" 2>&1 | head -5`
Expected: No `ImportError` or `ModuleNotFoundError`. May print Chroma/startup messages.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: use shared bootstrap module in app.py"
```

---

### Task 3: Update `api/server.py` to use bootstrap

**Files:**
- Modify: `api/server.py:1-6`

- [ ] **Step 1: Replace the top of `api/server.py`**

Current lines 1-6:
```python
import json
import os
import warnings
from dotenv import load_dotenv

load_dotenv(override=True)
```

Replace with:
```python
import bootstrap  # noqa: F401 — must run before any LangChain imports

import json
import os
import warnings
```

This removes the inline `load_dotenv` and its import. The `bootstrap` import handles dotenv loading and Phoenix registration.

- [ ] **Step 2: Verify server starts**

Run: `uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 &`
Then: `curl -s http://localhost:8000/users | head -c 100`
Expected: JSON array of users (confirms server boots and responds).
Cleanup: Kill the background uvicorn process.

- [ ] **Step 3: Commit**

```bash
git add api/server.py
git commit -m "refactor: use shared bootstrap module in api/server.py"
```

---

### Task 4: Verify end-to-end tracing

- [ ] **Step 1: Start the server**

Run: `uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 &`

- [ ] **Step 2: Send a chat request**

Run:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the travel policy?", "user_id": "alice"}'
```
Expected: JSON response with `response` field.

- [ ] **Step 3: Verify traces in Phoenix**

Open `https://app.phoenix.arize.com/s/ibarczewski` and check the `tutorial2` project for a new trace containing the LLM call.

- [ ] **Step 4: Verify traces in LangSmith**

Open LangSmith and check the `default` project for the same LLM call trace.

- [ ] **Step 5: Cleanup**

Kill the background uvicorn process.
