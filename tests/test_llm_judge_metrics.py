"""Unit tests for lm_eval.api.metrics_llm_judge module.

These tests use mocking to avoid actual API calls and DeepEval dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

# Ensure metrics are registered by importing via the main metrics module
import lm_eval.api.metrics  # noqa: F401


class TestBuildEvaluationParams:
    """Tests for _build_evaluation_params helper function."""

    def test_minimal_params(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text=None,
            expected_output=None,
            context=None,
            retrieval_context=None,
        )
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert len(params) == 1

    def test_with_expected_output(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text=None,
            expected_output="expected",
            context=None,
            retrieval_context=None,
        )
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params

    def test_with_input_text(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text="question",
            expected_output=None,
            context=None,
            retrieval_context=None,
        )
        assert LLMTestCaseParams.INPUT in params

    def test_with_context(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text=None,
            expected_output=None,
            context="context",
            retrieval_context=None,
        )
        assert LLMTestCaseParams.CONTEXT in params

    def test_with_retrieval_context(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text=None,
            expected_output=None,
            context=None,
            retrieval_context=["doc1", "doc2"],
        )
        assert LLMTestCaseParams.RETRIEVAL_CONTEXT in params

    def test_all_params(self):
        from lm_eval.api.metrics_llm_judge import _build_evaluation_params
        from deepeval.test_case import LLMTestCaseParams

        params = _build_evaluation_params(
            input_text="question",
            expected_output="expected",
            context="context",
            retrieval_context=["doc"],
        )
        assert len(params) == 5
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert LLMTestCaseParams.INPUT in params
        assert LLMTestCaseParams.CONTEXT in params
        assert LLMTestCaseParams.RETRIEVAL_CONTEXT in params


class TestBuildTestCase:
    """Tests for _build_test_case helper function."""

    def test_minimal_test_case(self):
        from lm_eval.api.metrics_llm_judge import _build_test_case

        test_case = _build_test_case(actual_output="output")
        assert test_case.actual_output == "output"
        assert test_case.input == ""

    def test_with_all_fields(self):
        from lm_eval.api.metrics_llm_judge import _build_test_case

        test_case = _build_test_case(
            actual_output="actual",
            input_text="input",
            expected_output="expected",
            context="context",
            retrieval_context=["doc1", "doc2"],
        )
        assert test_case.actual_output == "actual"
        assert test_case.input == "input"
        assert test_case.expected_output == "expected"
        assert test_case.context == ["context"]
        assert test_case.retrieval_context == ["doc1", "doc2"]

    def test_context_converted_to_list(self):
        from lm_eval.api.metrics_llm_judge import _build_test_case

        test_case = _build_test_case(
            actual_output="output",
            context="single context string",
        )
        assert test_case.context == ["single context string"]


class TestRunMetricMeasure:
    """Tests for _run_metric_measure helper function."""

    def test_sync_mode(self):
        from lm_eval.api.metrics_llm_judge import _run_metric_measure

        mock_metric = MagicMock()
        mock_test_case = MagicMock()

        _run_metric_measure(mock_metric, mock_test_case, async_mode=False)

        mock_metric.measure.assert_called_once_with(mock_test_case)
        mock_metric.a_measure.assert_not_called()

    def test_async_mode_no_running_loop(self):
        from lm_eval.api.metrics_llm_judge import _run_metric_measure

        mock_metric = MagicMock()
        mock_metric.a_measure = MagicMock(return_value=MagicMock())

        async def mock_a_measure(tc):
            return None

        mock_metric.a_measure = mock_a_measure
        mock_test_case = MagicMock()

        with patch("asyncio.run") as mock_run:
            with patch("asyncio.get_running_loop", side_effect=RuntimeError):
                _run_metric_measure(mock_metric, mock_test_case, async_mode=True)
                mock_run.assert_called()

    def test_async_mode_fallback_on_error(self):
        from lm_eval.api.metrics_llm_judge import _run_metric_measure

        mock_metric = MagicMock()
        mock_test_case = MagicMock()

        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=Exception("async error")):
                _run_metric_measure(mock_metric, mock_test_case, async_mode=True)
                mock_metric.measure.assert_called_once_with(mock_test_case)


class TestGEvalFn:
    """Tests for g_eval_fn metric function."""

    def test_basic_call(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Good answer"

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch(
                "lm_eval.api.metrics_llm_judge._run_metric_measure"
            ) as mock_run:
                score = g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    criteria="Test criteria",
                )
                assert score == 0.85
                mock_run.assert_called_once()

    def test_with_evaluation_steps(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ) as mock_geval:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    evaluation_steps=["Step 1", "Step 2"],
                )
                assert score == 0.9
                call_kwargs = mock_geval.call_args.kwargs
                assert call_kwargs["evaluation_steps"] == ["Step 1", "Step 2"]

    def test_criteria_used_when_no_steps(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.7
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ) as mock_geval:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    criteria="Custom criteria",
                )
                call_kwargs = mock_geval.call_args.kwargs
                assert call_kwargs["criteria"] == "Custom criteria"

    def test_custom_judge_model(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ) as mock_geval:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    judge_model="gpt-4-turbo",
                )
                call_kwargs = mock_geval.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4-turbo"

    def test_threshold_and_strict_mode(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.6
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ) as mock_geval:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    threshold=0.7,
                    strict_mode=True,
                )
                call_kwargs = mock_geval.call_args.kwargs
                assert call_kwargs["threshold"] == 0.7
                assert call_kwargs["strict_mode"] is True

    def test_empty_references(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.5
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = g_eval_fn(
                    references=[],
                    predictions=["actual"],
                )
                assert score == 0.5

    def test_none_score_returns_zero(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = None
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                )
                assert score == 0.0


class TestAnswerRelevancyFn:
    """Tests for answer_relevancy_fn metric function."""

    def test_basic_call(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = answer_relevancy_fn(
                    references=["not used"],
                    predictions=["answer"],
                    input_text="question",
                )
                assert score == 0.9

    def test_custom_model(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ) as mock_class:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                answer_relevancy_fn(
                    references=[],
                    predictions=["answer"],
                    judge_model="gpt-4-turbo",
                )
                call_kwargs = mock_class.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4-turbo"

    def test_without_input_text(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.5
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                with patch(
                    "lm_eval.api.metrics_llm_judge.LLMTestCase"
                ) as mock_tc:
                    answer_relevancy_fn(
                        references=[],
                        predictions=["answer"],
                    )
                    call_kwargs = mock_tc.call_args.kwargs
                    assert call_kwargs["input"] == ""

    def test_none_score_returns_zero(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = None
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = answer_relevancy_fn(
                    references=[],
                    predictions=["answer"],
                )
                assert score == 0.0


class TestFaithfulnessFn:
    """Tests for faithfulness_fn metric function."""

    def test_basic_call(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = faithfulness_fn(
                    references=[],
                    predictions=["answer based on context"],
                    retrieval_context=["context document"],
                )
                assert score == 0.95

    def test_with_input_text(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                with patch(
                    "lm_eval.api.metrics_llm_judge.LLMTestCase"
                ) as mock_tc:
                    faithfulness_fn(
                        references=[],
                        predictions=["answer"],
                        input_text="question",
                        retrieval_context=["doc"],
                    )
                    call_kwargs = mock_tc.call_args.kwargs
                    assert call_kwargs["input"] == "question"

    def test_empty_retrieval_context(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                with patch(
                    "lm_eval.api.metrics_llm_judge.LLMTestCase"
                ) as mock_tc:
                    faithfulness_fn(
                        references=[],
                        predictions=["answer"],
                    )
                    call_kwargs = mock_tc.call_args.kwargs
                    assert call_kwargs["retrieval_context"] == []

    def test_custom_threshold(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.7
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ) as mock_class:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                faithfulness_fn(
                    references=[],
                    predictions=["answer"],
                    threshold=0.8,
                    strict_mode=True,
                )
                call_kwargs = mock_class.call_args.kwargs
                assert call_kwargs["threshold"] == 0.8
                assert call_kwargs["strict_mode"] is True

    def test_none_score_returns_zero(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = None
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                score = faithfulness_fn(
                    references=[],
                    predictions=["answer"],
                )
                assert score == 0.0


class TestMetricRegistration:
    """Tests for metric registration in the registry."""

    def test_g_eval_registered(self):
        from lm_eval.api.registry import METRIC_REGISTRY

        assert "g_eval" in METRIC_REGISTRY

    def test_answer_relevancy_registered(self):
        from lm_eval.api.registry import METRIC_REGISTRY

        assert "answer_relevancy" in METRIC_REGISTRY

    def test_faithfulness_registered(self):
        from lm_eval.api.registry import METRIC_REGISTRY

        assert "faithfulness" in METRIC_REGISTRY

    def test_g_eval_metadata(self):
        from lm_eval.api.registry import (
            HIGHER_IS_BETTER_REGISTRY,
            METRIC_AGGREGATION_REGISTRY,
        )

        assert HIGHER_IS_BETTER_REGISTRY.get("g_eval") is True
        assert "g_eval" in METRIC_AGGREGATION_REGISTRY

    def test_answer_relevancy_metadata(self):
        from lm_eval.api.registry import (
            HIGHER_IS_BETTER_REGISTRY,
            METRIC_AGGREGATION_REGISTRY,
        )

        assert HIGHER_IS_BETTER_REGISTRY.get("answer_relevancy") is True
        assert "answer_relevancy" in METRIC_AGGREGATION_REGISTRY

    def test_faithfulness_metadata(self):
        from lm_eval.api.registry import (
            HIGHER_IS_BETTER_REGISTRY,
            METRIC_AGGREGATION_REGISTRY,
        )

        assert HIGHER_IS_BETTER_REGISTRY.get("faithfulness") is True
        assert "faithfulness" in METRIC_AGGREGATION_REGISTRY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
