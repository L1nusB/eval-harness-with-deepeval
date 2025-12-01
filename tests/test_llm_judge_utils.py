"""Unit tests for lm_eval.llm_judge.utils module."""

import pytest

from lm_eval.llm_judge.utils import (
    DEFAULT_SCORE_PROMPT,
    JudgePromptBuilder,
    ResponseParser,
)


class TestDefaultScorePrompt:
    """Test the default score prompt template."""

    def test_prompt_contains_placeholders(self):
        assert "{question}" in DEFAULT_SCORE_PROMPT
        assert "{prediction}" in DEFAULT_SCORE_PROMPT
        assert "{answer_section}" in DEFAULT_SCORE_PROMPT
        assert "{context_section}" in DEFAULT_SCORE_PROMPT

    def test_prompt_scoring_instructions(self):
        assert "0.0 to 1.0" in DEFAULT_SCORE_PROMPT
        assert "0.0 = Completely incorrect" in DEFAULT_SCORE_PROMPT
        assert "1.0 = Fully correct" in DEFAULT_SCORE_PROMPT


class TestJudgePromptBuilder:
    """Tests for JudgePromptBuilder class."""

    class TestBuildScorePrompt:
        """Tests for build_score_prompt method."""

        def test_minimal_prompt(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="What is 2+2?",
                prediction="4",
            )
            assert "What is 2+2?" in prompt
            assert "4" in prompt
            assert "0.0 to 1.0" in prompt

        def test_prompt_with_answer(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="What is the capital of France?",
                prediction="Paris",
                answer="Paris",
            )
            assert "What is the capital of France?" in prompt
            assert "Paris" in prompt
            assert "Expected Answer:" in prompt

        def test_prompt_with_context(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Summarize this text.",
                prediction="This is a summary.",
                context="The original text was about AI.",
            )
            assert "Context:" in prompt
            assert "The original text was about AI." in prompt

        def test_prompt_with_all_fields(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Explain gravity",
                prediction="Gravity is a force",
                answer="Gravity is the force of attraction",
                context="Physics context",
            )
            assert "Explain gravity" in prompt
            assert "Gravity is a force" in prompt
            assert "Expected Answer:" in prompt
            assert "Context:" in prompt

        def test_custom_template(self):
            custom_template = (
                "Q: {question}\nA: {prediction}\nExpected: {answer}\n{context}"
            )
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Test?",
                prediction="Answer",
                answer="Expected",
                context="Context",
                prompt_template=custom_template,
            )
            assert "Q: Test?" in prompt
            assert "A: Answer" in prompt
            assert "Expected: Expected" in prompt
            assert "Context" in prompt

        def test_custom_template_with_kwargs(self):
            custom_template = "{question} - {prediction} - {custom_field}"
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Q",
                prediction="P",
                prompt_template=custom_template,
                custom_field="CustomValue",
            )
            assert "CustomValue" in prompt

        def test_empty_answer_not_shown(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Q",
                prediction="P",
                answer=None,
            )
            assert "Expected Answer:" not in prompt

        def test_empty_context_not_shown(self):
            prompt = JudgePromptBuilder.build_score_prompt(
                question="Q",
                prediction="P",
                context=None,
            )
            assert "Context:" not in prompt

    class TestBuildGevalPrompt:
        """Tests for build_geval_prompt method."""

        def test_minimal_geval_prompt(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Evaluate correctness",
                actual_output="The answer is 42",
            )
            assert "Evaluation Criteria: Evaluate correctness" in prompt
            assert "Actual Output: The answer is 42" in prompt
            assert "0.0 to 1.0" in prompt

        def test_geval_with_steps(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Evaluate reasoning",
                evaluation_steps=[
                    "Check if logic is sound",
                    "Verify calculations",
                    "Confirm final answer",
                ],
                actual_output="Result",
            )
            assert "Evaluation Steps:" in prompt
            assert "1. Check if logic is sound" in prompt
            assert "2. Verify calculations" in prompt
            assert "3. Confirm final answer" in prompt

        def test_geval_with_input(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Relevance",
                input_text="What is AI?",
                actual_output="AI is artificial intelligence.",
            )
            assert "Input: What is AI?" in prompt

        def test_geval_with_expected_output(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Accuracy",
                actual_output="4",
                expected_output="4",
            )
            assert "Expected Output: 4" in prompt

        def test_geval_with_context(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Contextual relevance",
                actual_output="Based on the document...",
                context="Document about AI history",
            )
            assert "Context: Document about AI history" in prompt

        def test_geval_with_retrieval_context(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Faithfulness",
                actual_output="AI was invented in 1956.",
                retrieval_context="The field of AI was founded in 1956.",
            )
            assert (
                "Retrieval Context: The field of AI was founded in 1956."
                in prompt
            )

        def test_geval_all_fields(self):
            prompt = JudgePromptBuilder.build_geval_prompt(
                criteria="Comprehensive evaluation",
                evaluation_steps=["Step 1", "Step 2"],
                input_text="Question",
                actual_output="Actual",
                expected_output="Expected",
                context="Context info",
                retrieval_context="Retrieved info",
            )
            assert "Evaluation Criteria:" in prompt
            assert "Evaluation Steps:" in prompt
            assert "Input:" in prompt
            assert "Actual Output:" in prompt
            assert "Expected Output:" in prompt
            assert "Context:" in prompt
            assert "Retrieval Context:" in prompt


class TestResponseParser:
    """Tests for ResponseParser class."""

    class TestParseScoreResponse:
        """Tests for parse_score_response method."""

        def test_simple_decimal(self):
            assert ResponseParser.parse_score_response("0.85") == 0.85

        def test_integer(self):
            assert ResponseParser.parse_score_response("1") == 1.0

        def test_score_with_text(self):
            assert ResponseParser.parse_score_response("Score: 0.75") == 0.75

        def test_score_in_sentence(self):
            result = ResponseParser.parse_score_response(
                "After careful analysis, I would give this a score of 0.9."
            )
            assert result == 0.9

        def test_clamping_above_max(self):
            assert ResponseParser.parse_score_response("1.5") == 1.0

        def test_clamping_below_min(self):
            assert ResponseParser.parse_score_response("-0.5") == 0.0

        def test_custom_range(self):
            result = ResponseParser.parse_score_response("7", score_range=(0.0, 10.0))
            assert result == 7.0

        def test_custom_range_clamping(self):
            result = ResponseParser.parse_score_response("15", score_range=(0.0, 10.0))
            assert result == 10.0

        def test_multiple_numbers_takes_first(self):
            result = ResponseParser.parse_score_response("0.8 and then 0.9")
            assert result == 0.8

        def test_no_number_returns_min(self):
            result = ResponseParser.parse_score_response("No score provided")
            assert result == 0.0

        def test_empty_string_returns_min(self):
            result = ResponseParser.parse_score_response("")
            assert result == 0.0

        def test_whitespace_handling(self):
            result = ResponseParser.parse_score_response("  0.75  ")
            assert result == 0.75

        def test_negative_in_custom_range(self):
            result = ResponseParser.parse_score_response("-5", score_range=(-10.0, 10.0))
            assert result == -5.0

    class TestParseJsonResponse:
        """Tests for parse_json_response method."""

        def test_simple_json(self):
            result = ResponseParser.parse_json_response('{"score": 0.8}')
            assert result == {"score": 0.8}

        def test_complex_json(self):
            result = ResponseParser.parse_json_response(
                '{"score": 0.9, "reason": "Good answer", "details": {"accuracy": 0.95}}'
            )
            assert result["score"] == 0.9
            assert result["reason"] == "Good answer"
            assert result["details"]["accuracy"] == 0.95

        def test_json_with_surrounding_text(self):
            result = ResponseParser.parse_json_response(
                'Here is my evaluation: {"score": 0.7, "verdict": "pass"}'
            )
            assert result["score"] == 0.7
            assert result["verdict"] == "pass"

        def test_json_with_markdown_code_block(self):
            result = ResponseParser.parse_json_response(
                '```json\n{"score": 0.85}\n```'
            )
            assert result["score"] == 0.85

        def test_multiline_json(self):
            json_str = """
            {
                "score": 0.9,
                "breakdown": {
                    "accuracy": 0.95,
                    "relevance": 0.85
                }
            }
            """
            result = ResponseParser.parse_json_response(json_str)
            assert result["score"] == 0.9
            assert result["breakdown"]["accuracy"] == 0.95

        def test_invalid_json_returns_empty_dict(self):
            result = ResponseParser.parse_json_response("not valid json at all")
            assert result == {}

        def test_empty_string_returns_empty_dict(self):
            result = ResponseParser.parse_json_response("")
            assert result == {}

        def test_json_array_extraction(self):
            result = ResponseParser.parse_json_response(
                'Some text {"items": [1, 2, 3]} more text'
            )
            assert result["items"] == [1, 2, 3]

        def test_nested_braces(self):
            result = ResponseParser.parse_json_response(
                '{"outer": {"inner": {"value": 1}}}'
            )
            assert result["outer"]["inner"]["value"] == 1

        def test_whitespace_only_returns_empty_dict(self):
            result = ResponseParser.parse_json_response("   \n\t   ")
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
