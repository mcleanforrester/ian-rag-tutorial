# Tool Calling Design

**Date:** 2026-04-14  
**Scope:** Add two tools (calculator, employee lookup) to a new `/chat/tools` API endpoint using the raw Anthropic SDK, bypassing LangChain, so the full tool call cycle is explicit in code.

## Learning Objective

Demonstrate the complete tool call cycle: schema definition → model requests tool → application executes function → result injected back into conversation → model finishes reply. Every step visible in code.

## Architecture

New module `tools/` added alongside existing `conversation/`, `retrieval/` etc.

```
tools/
  __init__.py
  definitions.py   # tool schemas (Anthropic format) + Python functions
  executor.py      # dispatcher: tool_name + input_dict → result string
```

New endpoint `POST /chat/tools` in `api/server.py` — raw `anthropic.Anthropic().messages.create()`, manual loop.

## Tools

### Calculator
- **Schema inputs:** `operation` (enum: add/subtract/multiply/divide), `a` (number), `b` (number)
- **Returns:** result as string, or error string on divide-by-zero

### Employee Lookup
- **Schema inputs:** `employee_name` (string)
- **Returns:** JSON record `{id, name, department, role, manager}` or not-found message
- **Mock data:** Alice (engineering, senior engineer), Bob (engineering lead), Carol (VP of operations)

## Endpoint

`POST /chat/tools` — request: `{query, user_id}` (same as `/chat`)

Loop:
1. Call `messages.create()` with user query + tool definitions
2. If `stop_reason == "tool_use"`: extract tool_use blocks, dispatch each, build `tool_result` content, call model again
3. Repeat until `stop_reason == "end_turn"`
4. Return final text response

## What This Does NOT Do

- No RAG retrieval (tools replace it for this endpoint)
- No streaming
- No guardrails / structured extraction
- No LangChain (raw SDK only)
