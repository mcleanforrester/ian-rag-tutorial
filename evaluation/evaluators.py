import asyncio
import logging

from opentelemetry import trace
from phoenix.evals import (
    AnthropicModel,
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)
from phoenix.client import Client

logger = logging.getLogger(__name__)

judge_model = AnthropicModel(model="claude-sonnet-4-6-20250514", temperature=0.0)

hallucination_eval = HallucinationEvaluator(judge_model)
qa_eval = QAEvaluator(judge_model)
relevance_eval = RelevanceEvaluator(judge_model)

phoenix_client = Client()

_EVALUATIONS = [
    (
        "hallucination",
        hallucination_eval,
        lambda q, r, c: {"input": q, "reference": c, "output": r},
    ),
    (
        "qa_correctness",
        qa_eval,
        lambda q, r, c: {"input": q, "reference": c, "output": r},
    ),
    (
        "retrieval_relevance",
        relevance_eval,
        lambda q, r, c: {"input": q, "reference": c},
    ),
]


def get_current_span_id() -> str:
    span = trace.get_current_span()
    return trace.format_span_id(span.get_span_context().span_id)


async def _run_single_eval(name, evaluator, record, span_id):
    loop = asyncio.get_running_loop()
    try:
        label, score, explanation = await loop.run_in_executor(
            None, evaluator.evaluate, record
        )
        phoenix_client.spans.add_span_annotation(
            span_id=span_id,
            annotation_name=name,
            annotator_kind="LLM",
            label=label,
            score=score,
            explanation=explanation,
        )
    except Exception:
        logger.exception("Evaluation '%s' failed for span %s", name, span_id)


async def evaluate_and_log(
    span_id: str,
    query: str,
    response: str,
    context: str,
) -> None:
    tasks = []
    for name, evaluator, build_record in _EVALUATIONS:
        record = build_record(query, response, context)
        tasks.append(_run_single_eval(name, evaluator, record, span_id))
    await asyncio.gather(*tasks)
