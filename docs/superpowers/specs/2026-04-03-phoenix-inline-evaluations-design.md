# Phoenix Inline Evaluations Design

## Summary

Add programmatic, inline quality scoring to the RAG pipeline using Arize Phoenix evaluations. Every `/chat` and `/chat/stream` request will be evaluated for retrieval relevance, response faithfulness, and response correctness. Evaluations run asynchronously (fire-and-forget) so they add zero latency to the user-facing response. Results are logged back to Phoenix as span annotations, visible in the cloud dashboard.

## Context

The project already has Phoenix OTEL tracing live via `bootstrap.py`, exporting traces to `app.phoenix.arize.com`. All LLM, embedding, and retrieval calls are captured. However, there is no quality scoring — traces show what happened but not how good the results were. This design adds that quality signal.

## Design Decisions

- **Online, not batch**: Evaluations run on every request rather than in periodic batch jobs. This gives immediate quality visibility.
- **Fire-and-forget**: Evaluations run as background `asyncio` tasks after the response is sent. Zero added latency for the user.
- **Claude Sonnet 4.6 as judge**: Same model already configured in `config.py`. Simplest option — no additional API keys or providers needed.
- **Phoenix span annotations**: Results are logged via the Phoenix Client API, appearing as first-class annotations in the Phoenix UI (not raw OTel attributes).

## New Dependencies

- `arize-phoenix-evals` — Phoenix evaluation framework (evaluators, scoring)
- `arize-phoenix` — Phoenix Client for logging annotations back to the dashboard

## Architecture

### New Module: `evaluation/evaluators.py`

Single module responsible for all evaluation logic.

**Responsibilities:**
1. Initialize 3 Phoenix evaluators using Claude Sonnet 4.6 as the judge LLM
2. Expose `evaluate_and_log(span_id, query, response, context)` — an async function
3. Run all 3 evaluations concurrently via `asyncio.gather`
4. Build a DataFrame with `context.span_id`, `label`, and `score` columns
5. Log results to Phoenix via `Client().spans.log_span_annotations_dataframe()`

**Evaluators:**

| Name | Evaluator Class | Inputs | Purpose |
|------|----------------|--------|---------|
| Retrieval Relevance | `DocumentRelevanceEvaluator` | query, retrieved doc text | Are the retrieved documents relevant to the user's question? |
| Faithfulness | `FaithfulnessEvaluator` | query, response, context | Is the LLM's response grounded in the retrieved context? |
| Correctness | `CorrectnessEvaluator` | query, response | Is the response a correct and helpful answer? |

**Error handling:** Evaluation failures are logged but never propagate to the user. A failed eval simply means that annotation is missing from the trace.

### Integration: `api/server.py`

Both `/chat` and `/chat/stream` handlers are modified to:

1. Capture the current OTel span ID from the active trace context (`opentelemetry.trace.get_current_span().get_span_context().span_id`)
2. After generating the response, call `asyncio.create_task(evaluate_and_log(span_id, query, response_text, docs_content))`
3. Return the response immediately — the task runs in the background

No changes to request/response models. No changes to `bootstrap.py`, `config.py`, or any other existing module.

### New File: `evaluation/__init__.py`

Empty init file for the new package.

## Data Flow

```
User Request
    |
    v
api/server.py: retrieve docs, invoke LLM, get response
    |
    +---> Return response to user immediately
    |
    +---> asyncio.create_task(evaluate_and_log(...))
              |
              v
          evaluation/evaluators.py:
              |
              +-- DocumentRelevanceEvaluator.evaluate() --|
              +-- FaithfulnessEvaluator.evaluate()    ---|-- asyncio.gather (concurrent)
              +-- CorrectnessEvaluator.evaluate()     --|
              |
              v
          Build annotations DataFrame
              |
              v
          Client().spans.log_span_annotations_dataframe()
              |
              v
          Scores visible in Phoenix dashboard
```

## What Appears in Phoenix

Each traced request gets 3 annotations attached to its span:

- **retrieval_relevance**: label (e.g., "relevant"/"irrelevant") + numeric score
- **faithfulness**: label (e.g., "faithful"/"hallucinated") + numeric score
- **correctness**: label (e.g., "correct"/"incorrect") + numeric score

These are filterable and aggregatable in the Phoenix cloud UI at `app.phoenix.arize.com`.

## Files Changed

| File | Change |
|------|--------|
| `evaluation/__init__.py` | New (empty) |
| `evaluation/evaluators.py` | New — evaluator init, `evaluate_and_log()` function |
| `api/server.py` | Modified — capture span ID, fire-and-forget eval task in both handlers |
| `requirements.txt` | Modified — add `arize-phoenix-evals`, `arize-phoenix` |

## Out of Scope

- Batch/offline evaluation
- Custom evaluation prompts or rubrics
- Evaluation of the `/chat/stream` partial tokens (only the full accumulated response is evaluated)
- Dashboard configuration or alerting
- Evaluation of the structured `PolicySummary` extraction quality
