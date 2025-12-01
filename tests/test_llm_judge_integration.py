"""Integration tests for LLM Judge metrics with ConfigurableTask.

These tests verify that LLM Judge metrics integrate correctly with the
lm-evaluation-harness metrics pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

# Ensure metrics are registered by importing via the main metrics module
import lm_eval.api.metrics  # noqa: F401


class TestMetricResolverIntegration:
    """Tests for metric resolution through the registry."""

    def test_g_eval_resolves_from_registry(self):
        from lm_eval.api.registry import get_metric

        metric_fn = get_metric("g_eval")
        assert metric_fn is not None
        assert callable(metric_fn)

    def test_answer_relevancy_resolves_from_registry(self):
        from lm_eval.api.registry import get_metric

        metric_fn = get_metric("answer_relevancy")
        assert metric_fn is not None
        assert callable(metric_fn)

    def test_faithfulness_resolves_from_registry(self):
        from lm_eval.api.registry import get_metric

        metric_fn = get_metric("faithfulness")
        assert metric_fn is not None
        assert callable(metric_fn)

    def test_metric_signature_compatibility(self):
        from lm_eval.api.registry import get_metric
        import inspect

        for metric_name in ["g_eval", "answer_relevancy", "faithfulness"]:
            metric_fn = get_metric(metric_name)
            sig = inspect.signature(metric_fn)
            params = list(sig.parameters.keys())
            assert "references" in params
            assert "predictions" in params


class TestMetricKwargsFlow:
    """Tests that verify kwargs flow correctly from YAML to metric functions."""

    def test_g_eval_receives_kwargs(self):
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
                    criteria="custom criteria",
                    judge_model="gpt-4-turbo",
                    threshold=0.7,
                    strict_mode=True,
                    async_mode=False,
                    input_text="question",
                    context="context",
                    retrieval_context=["doc1"],
                )

                call_kwargs = mock_geval.call_args.kwargs
                assert call_kwargs["criteria"] == "custom criteria"
                assert call_kwargs["model"] == "gpt-4-turbo"
                assert call_kwargs["threshold"] == 0.7
                assert call_kwargs["strict_mode"] is True
                assert call_kwargs["async_mode"] is False

    def test_answer_relevancy_receives_kwargs(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.9
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
                    threshold=0.6,
                    strict_mode=True,
                    async_mode=False,
                    input_text="question",
                )

                call_kwargs = mock_class.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4-turbo"
                assert call_kwargs["threshold"] == 0.6
                assert call_kwargs["strict_mode"] is True
                assert call_kwargs["async_mode"] is False

    def test_faithfulness_receives_kwargs(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ) as mock_class:
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                faithfulness_fn(
                    references=[],
                    predictions=["answer"],
                    judge_model="gpt-4-turbo",
                    threshold=0.75,
                    strict_mode=True,
                    async_mode=False,
                    input_text="question",
                    retrieval_context=["doc1", "doc2"],
                )

                call_kwargs = mock_class.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4-turbo"
                assert call_kwargs["threshold"] == 0.75
                assert call_kwargs["strict_mode"] is True
                assert call_kwargs["async_mode"] is False


class TestMetricOutputFormat:
    """Tests that verify metrics return correctly formatted output."""

    def test_g_eval_returns_float(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                )
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0

    def test_answer_relevancy_returns_float(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = answer_relevancy_fn(
                    references=[],
                    predictions=["answer"],
                )
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0

    def test_faithfulness_returns_float(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = faithfulness_fn(
                    references=[],
                    predictions=["answer"],
                )
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0


class TestAggregationIntegration:
    """Tests for metric aggregation compatibility."""

    def test_g_eval_aggregation_registered(self):
        from lm_eval.api.registry import METRIC_AGGREGATION_REGISTRY

        assert "g_eval" in METRIC_AGGREGATION_REGISTRY

    def test_answer_relevancy_aggregation_registered(self):
        from lm_eval.api.registry import METRIC_AGGREGATION_REGISTRY

        assert "answer_relevancy" in METRIC_AGGREGATION_REGISTRY

    def test_faithfulness_aggregation_registered(self):
        from lm_eval.api.registry import METRIC_AGGREGATION_REGISTRY

        assert "faithfulness" in METRIC_AGGREGATION_REGISTRY

    def test_mean_aggregation_works_with_scores(self):
        from lm_eval.api.metrics import mean

        scores = [0.8, 0.9, 0.7, 0.85]
        result = mean(scores)
        assert result == pytest.approx(0.8125)


class TestHigherIsBetterConfig:
    """Tests for higher_is_better configuration."""

    def test_g_eval_higher_is_better(self):
        from lm_eval.api.registry import HIGHER_IS_BETTER_REGISTRY

        assert HIGHER_IS_BETTER_REGISTRY["g_eval"] is True

    def test_answer_relevancy_higher_is_better(self):
        from lm_eval.api.registry import HIGHER_IS_BETTER_REGISTRY

        assert HIGHER_IS_BETTER_REGISTRY["answer_relevancy"] is True

    def test_faithfulness_higher_is_better(self):
        from lm_eval.api.registry import HIGHER_IS_BETTER_REGISTRY

        assert HIGHER_IS_BETTER_REGISTRY["faithfulness"] is True


class TestExtraKwargsHandling:
    """Tests that extra kwargs don't cause errors."""

    def test_g_eval_ignores_unknown_kwargs(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = g_eval_fn(
                    references=["expected"],
                    predictions=["actual"],
                    unknown_param="value",
                    another_unknown=123,
                )
                assert isinstance(result, float)

    def test_answer_relevancy_ignores_unknown_kwargs(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = answer_relevancy_fn(
                    references=[],
                    predictions=["answer"],
                    unknown_param="value",
                )
                assert isinstance(result, float)

    def test_faithfulness_ignores_unknown_kwargs(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = faithfulness_fn(
                    references=[],
                    predictions=["answer"],
                    unknown_param="value",
                )
                assert isinstance(result, float)


class TestEmptyInputHandling:
    """Tests for handling empty or minimal inputs."""

    def test_g_eval_handles_empty_predictions(self):
        from lm_eval.api.metrics_llm_judge import g_eval_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.GEval", return_value=mock_metric
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = g_eval_fn(
                    references=["expected"],
                    predictions=[],
                )
                assert result == 0.0

    def test_answer_relevancy_handles_empty_predictions(self):
        from lm_eval.api.metrics_llm_judge import answer_relevancy_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.AnswerRelevancyMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = answer_relevancy_fn(
                    references=[],
                    predictions=[],
                )
                assert result == 0.0

    def test_faithfulness_handles_empty_predictions(self):
        from lm_eval.api.metrics_llm_judge import faithfulness_fn

        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = None

        with patch(
            "lm_eval.api.metrics_llm_judge.FaithfulnessMetric",
            return_value=mock_metric,
        ):
            with patch("lm_eval.api.metrics_llm_judge._run_metric_measure"):
                result = faithfulness_fn(
                    references=[],
                    predictions=[],
                )
                assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
