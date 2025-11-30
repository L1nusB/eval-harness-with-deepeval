"""Utility classes for LLM judge prompt building and response parsing."""

import json
import re
from typing import Any


DEFAULT_SCORE_PROMPT = """You are an expert evaluator. Rate the quality of the given response.

Question/Prompt: {question}
{answer_section}
Response to evaluate: {prediction}
{context_section}

Provide a score from 0.0 to 1.0 where:
- 0.0 = Completely incorrect or irrelevant
- 0.5 = Partially correct with significant issues
- 1.0 = Fully correct and comprehensive

Output ONLY a single decimal number between 0.0 and 1.0."""


class JudgePromptBuilder:
    """Helper class to build prompts for different judge evaluation types."""

    @staticmethod
    def build_score_prompt(
        question: str,
        prediction: str,
        answer: str | None = None,
        context: str | None = None,
        prompt_template: str | None = None,
        **kwargs,
    ) -> str:
        """Build a scoring prompt.

        If a custom prompt_template is provided, it will be used with format()
        substitution. Otherwise, the default scoring prompt is used.

        Args:
            question: The original question or prompt
            prediction: The model's prediction to evaluate
            answer: Optional ground truth answer
            context: Optional additional context
            prompt_template: Optional custom prompt template
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        if prompt_template:
            return prompt_template.format(
                question=question,
                prediction=prediction,
                answer=answer or "",
                context=context or "",
                **kwargs,
            )

        answer_section = f"\nExpected Answer: {answer}" if answer else ""
        context_section = f"\nContext: {context}" if context else ""

        return DEFAULT_SCORE_PROMPT.format(
            question=question,
            prediction=prediction,
            answer_section=answer_section,
            context_section=context_section,
        )

    @staticmethod
    def build_geval_prompt(
        criteria: str,
        evaluation_steps: list[str] | None = None,
        input_text: str | None = None,
        actual_output: str = "",
        expected_output: str | None = None,
        context: str | None = None,
        retrieval_context: str | None = None,
    ) -> str:
        """Build a G-Eval style prompt with criteria and evaluation steps.

        Args:
            criteria: The evaluation criteria description
            evaluation_steps: Optional list of evaluation steps
            input_text: The original input/question
            actual_output: The model's actual output to evaluate
            expected_output: Optional expected/ground truth output
            context: Optional additional context
            retrieval_context: Optional retrieval context for RAG evaluation

        Returns:
            Formatted G-Eval prompt string
        """
        prompt_parts = [f"Evaluation Criteria: {criteria}\n"]

        if evaluation_steps:
            steps_text = "\n".join(
                f"{i + 1}. {step}" for i, step in enumerate(evaluation_steps)
            )
            prompt_parts.append(f"Evaluation Steps:\n{steps_text}\n")

        if input_text:
            prompt_parts.append(f"Input: {input_text}\n")

        prompt_parts.append(f"Actual Output: {actual_output}\n")

        if expected_output:
            prompt_parts.append(f"Expected Output: {expected_output}\n")
        if context:
            prompt_parts.append(f"Context: {context}\n")
        if retrieval_context:
            prompt_parts.append(f"Retrieval Context: {retrieval_context}\n")

        prompt_parts.append(
            "\nBased on the criteria above, provide a score from 0.0 to 1.0. "
            "Output ONLY a single decimal number."
        )

        return "\n".join(prompt_parts)


class ResponseParser:
    """Helper class to parse different types of judge responses."""

    @staticmethod
    def parse_score_response(
        response: str, score_range: tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """Parse a numeric score from response text.

        Extracts the first number found in the response and clamps it
        to the specified range.

        Args:
            response: Raw response text from the judge
            score_range: Tuple of (min, max) for score clamping

        Returns:
            Parsed and clamped score, or min score if parsing fails
        """
        try:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response.strip())
            if numbers:
                score = float(numbers[0])
                return max(score_range[0], min(score, score_range[1]))
        except (ValueError, IndexError):
            pass
        return score_range[0]

    @staticmethod
    def parse_json_response(response: str) -> dict[str, Any]:
        """Parse JSON from response text.

        Attempts direct parsing first, then falls back to extracting
        embedded JSON if direct parsing fails.

        Args:
            response: Raw response text that may contain JSON

        Returns:
            Parsed JSON dict, or empty dict if parsing fails
        """
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {}
