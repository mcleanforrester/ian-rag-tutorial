import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


def test_evaluate_and_log_calls_all_three_evaluators():
    """All three evaluators should be called with the correct record dicts."""
    with (
        patch("evaluation.evaluators.hallucination_eval") as mock_hall,
        patch("evaluation.evaluators.qa_eval") as mock_qa,
        patch("evaluation.evaluators.relevance_eval") as mock_rel,
        patch("evaluation.evaluators.phoenix_client") as mock_client,
    ):
        mock_hall.evaluate.return_value = ("factual", 1.0, "no hallucination")
        mock_qa.evaluate.return_value = ("correct", 1.0, "answer is correct")
        mock_rel.evaluate.return_value = ("relevant", 1.0, "docs are relevant")

        from evaluation.evaluators import evaluate_and_log

        asyncio.run(evaluate_and_log(
            span_id="abc123",
            query="What is the PTO policy?",
            response="Employees get 20 days PTO.",
            context="PTO policy: 20 days per year.",
        ))

        mock_hall.evaluate.assert_called_once_with(
            {
                "input": "What is the PTO policy?",
                "reference": "PTO policy: 20 days per year.",
                "output": "Employees get 20 days PTO.",
            }
        )
        mock_qa.evaluate.assert_called_once_with(
            {
                "input": "What is the PTO policy?",
                "reference": "PTO policy: 20 days per year.",
                "output": "Employees get 20 days PTO.",
            }
        )
        mock_rel.evaluate.assert_called_once_with(
            {
                "input": "What is the PTO policy?",
                "reference": "PTO policy: 20 days per year.",
            }
        )


def test_evaluate_and_log_logs_annotations_to_phoenix():
    """Each evaluator result should produce an add_span_annotation call."""
    with (
        patch("evaluation.evaluators.hallucination_eval") as mock_hall,
        patch("evaluation.evaluators.qa_eval") as mock_qa,
        patch("evaluation.evaluators.relevance_eval") as mock_rel,
        patch("evaluation.evaluators.phoenix_client") as mock_client,
    ):
        mock_hall.evaluate.return_value = ("factual", 1.0, "no hallucination")
        mock_qa.evaluate.return_value = ("correct", 0.9, "mostly correct")
        mock_rel.evaluate.return_value = ("relevant", 1.0, "docs relevant")

        from evaluation.evaluators import evaluate_and_log

        asyncio.run(evaluate_and_log(
            span_id="span123",
            query="q",
            response="r",
            context="c",
        ))

        calls = mock_client.spans.add_span_annotation.call_args_list
        assert len(calls) == 3

        annotation_names = {c.kwargs["annotation_name"] for c in calls}
        assert annotation_names == {"hallucination", "qa_correctness", "retrieval_relevance"}

        for call in calls:
            assert call.kwargs["span_id"] == "span123"
            assert call.kwargs["annotator_kind"] == "LLM"


def test_evaluate_and_log_handles_evaluator_failure():
    """If one evaluator raises, the others should still log their results."""
    with (
        patch("evaluation.evaluators.hallucination_eval") as mock_hall,
        patch("evaluation.evaluators.qa_eval") as mock_qa,
        patch("evaluation.evaluators.relevance_eval") as mock_rel,
        patch("evaluation.evaluators.phoenix_client") as mock_client,
    ):
        mock_hall.evaluate.side_effect = Exception("API error")
        mock_qa.evaluate.return_value = ("correct", 1.0, "good")
        mock_rel.evaluate.return_value = ("relevant", 1.0, "good")

        from evaluation.evaluators import evaluate_and_log

        # Should not raise
        asyncio.run(evaluate_and_log(
            span_id="span456",
            query="q",
            response="r",
            context="c",
        ))

        # Only 2 annotations logged (hallucination failed)
        calls = mock_client.spans.add_span_annotation.call_args_list
        assert len(calls) == 2
