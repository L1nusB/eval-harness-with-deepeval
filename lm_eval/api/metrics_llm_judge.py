"""LLM-as-Judge metrics using DeepEval for lm-evaluation-harness.

DeepEval is a required dependency for these metrics.
Install with: pip install deepeval
"""

import asyncio
import logging

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from lm_eval.api.registry import register_metric


eval_logger = logging.getLogger(__name__)


def _build_evaluation_params(
    input_text: str | None,
    expected_output: str | None,
    context: str | None = None,
    retrieval_context: list[str] | None = None,
) -> list[LLMTestCaseParams]:
    """Build evaluation params based on available data.

    DeepEval's GEval metric requires specifying which parameters to evaluate.
    This function determines the appropriate parameters based on what data
    is available.

    Args:
        input_text: Original input/question
        expected_output: Ground truth/expected answer
        context: Additional context for evaluation
        retrieval_context: Retrieved documents for RAG evaluation

    Returns:
        List of LLMTestCaseParams to use for evaluation
    """
    params = [LLMTestCaseParams.ACTUAL_OUTPUT]
    if expected_output:
        params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
    if input_text:
        params.append(LLMTestCaseParams.INPUT)
    if context:
        params.append(LLMTestCaseParams.CONTEXT)
    if retrieval_context:
        params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
    return params


def _build_test_case(
    actual_output: str,
    input_text: str | None = None,
    expected_output: str | None = None,
    context: str | None = None,
    retrieval_context: list[str] | None = None,
) -> LLMTestCase:
    """Build an LLMTestCase from available data.

    Creates a DeepEval LLMTestCase with the provided data. Handles
    the conversion of context from a single string to a list as
    required by DeepEval.

    Args:
        actual_output: Model's actual output (prediction)
        input_text: Original input/question
        expected_output: Ground truth/expected answer
        context: Additional context for evaluation
        retrieval_context: Retrieved documents for RAG evaluation

    Returns:
        LLMTestCase configured with the provided data
    """
    return LLMTestCase(
        input=input_text or "",
        actual_output=actual_output,
        expected_output=expected_output,
        context=[context] if context else None,
        retrieval_context=retrieval_context,
    )


def _run_metric_measure(metric, test_case, async_mode: bool) -> None:
    """Run metric measurement with proper async handling.

    Handles the complexity of running async metrics in various contexts,
    including when already inside an event loop.

    Args:
        metric: DeepEval metric to run
        test_case: LLMTestCase to evaluate
        async_mode: Whether to use async evaluation
    """
    if async_mode:
        try:
            try:
                loop = asyncio.get_running_loop()
                loop_is_running = True
            except RuntimeError:
                loop_is_running = False

            if loop_is_running:
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(metric.a_measure(test_case))
            else:
                asyncio.run(metric.a_measure(test_case))
        except Exception as e:
            eval_logger.warning(f"Async evaluation failed, falling back to sync: {e}")
            metric.measure(test_case)
    else:
        metric.measure(test_case)


@register_metric(
    metric="g_eval",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def g_eval_fn(
    references: list[str],
    predictions: list[str],
    criteria: str = "Determine if the actual output is correct based on the expected output.",
    evaluation_steps: list[str] | None = None,
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: str | None = None,
    context: str | None = None,
    retrieval_context: list[str] | None = None,
    **kwargs,
) -> float:
    """G-Eval metric using DeepEval's GEval implementation.

    This metric uses an LLM judge to evaluate response quality based on
    customizable criteria and evaluation steps. It supports both criterion-based
    and step-based evaluation modes.

    YAML Configuration Example:
        metric_list:
          - metric: g_eval
            aggregation: mean
            higher_is_better: true
            criteria: "Evaluate the mathematical reasoning and correctness."
            evaluation_steps:
              - "Check if reasoning steps are logically sound"
              - "Verify calculations are correct"
              - "Confirm final answer matches expected value"
            judge_model: gpt-4o
            threshold: 0.7
            strict_mode: false
            async_mode: true

    Args:
        references: List containing expected output (gold standard)
        predictions: List containing model's actual output
        criteria: Natural language description of evaluation criteria
        evaluation_steps: Optional list of specific evaluation steps
            (if provided, takes precedence over criteria)
        judge_model: Model to use as judge (default: gpt-4o)
        threshold: Score threshold for success (default: 0.5)
        strict_mode: If True, scores below threshold become 0 (default: False)
        async_mode: Use async evaluation (default: True)
        input_text: Original input/question (optional)
        context: Additional context for evaluation (optional)
        retrieval_context: Retrieved documents for RAG evaluation (optional)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Score between 0.0 and 1.0
    """
    expected = references[0] if references else None
    actual = predictions[0] if predictions else ""

    eval_params = _build_evaluation_params(
        input_text, expected, context, retrieval_context
    )

    if evaluation_steps:
        metric = GEval(
            name="g_eval",
            evaluation_steps=evaluation_steps,
            evaluation_params=eval_params,
            model=judge_model,
            threshold=threshold,
            strict_mode=strict_mode,
            async_mode=async_mode,
        )
    else:
        metric = GEval(
            name="g_eval",
            criteria=criteria,
            evaluation_params=eval_params,
            model=judge_model,
            threshold=threshold,
            strict_mode=strict_mode,
            async_mode=async_mode,
        )

    test_case = _build_test_case(
        actual_output=actual,
        input_text=input_text,
        expected_output=expected,
        context=context,
        retrieval_context=retrieval_context,
    )

    _run_metric_measure(metric, test_case, async_mode)

    score = metric.score if metric.score is not None else 0.0

    if metric.reason:
        eval_logger.debug(f"G-Eval reason: {metric.reason}")

    return score


@register_metric(
    metric="answer_relevancy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def answer_relevancy_fn(
    references: list[str],
    predictions: list[str],
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: str | None = None,
    **kwargs,
) -> float:
    """Answer Relevancy metric using DeepEval.

    Evaluates whether the LLM's output is relevant to the input query.
    This is a reference-free metric - it only needs input and actual_output.

    YAML Configuration Example:
        metric_list:
          - metric: answer_relevancy
            aggregation: mean
            higher_is_better: true
            judge_model: gpt-4o
            threshold: 0.7
            input_text: "{{question}}"

    Args:
        references: List containing expected output (not used for this metric)
        predictions: List containing model's actual output
        judge_model: Model to use as judge (default: gpt-4o)
        threshold: Score threshold for success (default: 0.5)
        strict_mode: If True, scores below threshold become 0 (default: False)
        async_mode: Use async evaluation (default: True)
        input_text: Original input/question (required for meaningful evaluation)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Score between 0.0 and 1.0
    """
    actual = predictions[0] if predictions else ""

    metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=threshold,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )

    test_case = LLMTestCase(
        input=input_text or "",
        actual_output=actual,
    )

    _run_metric_measure(metric, test_case, async_mode)

    score = metric.score if metric.score is not None else 0.0

    if hasattr(metric, "reason") and metric.reason:
        eval_logger.debug(f"Answer Relevancy reason: {metric.reason}")

    return score


@register_metric(
    metric="faithfulness",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def faithfulness_fn(
    references: list[str],
    predictions: list[str],
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: str | None = None,
    retrieval_context: list[str] | None = None,
    **kwargs,
) -> float:
    """Faithfulness metric using DeepEval.

    Evaluates whether the LLM's output is factually consistent with the
    retrieval context. Useful for RAG system evaluation.

    YAML Configuration Example:
        metric_list:
          - metric: faithfulness
            aggregation: mean
            higher_is_better: true
            judge_model: gpt-4o
            threshold: 0.7
            retrieval_context:
              - "{{context}}"

    Args:
        references: List containing expected output (not used for this metric)
        predictions: List containing model's actual output
        judge_model: Model to use as judge (default: gpt-4o)
        threshold: Score threshold for success (default: 0.5)
        strict_mode: If True, scores below threshold become 0 (default: False)
        async_mode: Use async evaluation (default: True)
        input_text: Original input/question (optional)
        retrieval_context: Retrieved documents for RAG evaluation
            (required for meaningful evaluation)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Score between 0.0 and 1.0
    """
    actual = predictions[0] if predictions else ""

    metric = FaithfulnessMetric(
        model=judge_model,
        threshold=threshold,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )

    test_case = LLMTestCase(
        input=input_text or "",
        actual_output=actual,
        retrieval_context=retrieval_context or [],
    )

    _run_metric_measure(metric, test_case, async_mode)

    score = metric.score if metric.score is not None else 0.0

    if hasattr(metric, "reason") and metric.reason:
        eval_logger.debug(f"Faithfulness reason: {metric.reason}")

    return score
