"""Unit tests for lm_eval.llm_judge.protocol module."""

import pytest

from lm_eval.llm_judge.protocol import (
    DEFAULT_NUM_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
    Request,
    Response,
    ServerConfig,
)


class TestConstants:
    """Test default constants."""

    def test_default_num_retries(self):
        assert DEFAULT_NUM_RETRIES == 3

    def test_default_retry_delay(self):
        assert DEFAULT_RETRY_DELAY == 5.0

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 60


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_minimal_config(self):
        config = ServerConfig(model_name="gpt-4o")
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.top_p is None
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.num_retries == DEFAULT_NUM_RETRIES
        assert config.retry_delay == DEFAULT_RETRY_DELAY
        assert config.max_concurrent == 10
        assert config.system_prompt is None
        assert config.response_format is None

    def test_full_config(self):
        config = ServerConfig(
            model_name="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            timeout=120,
            num_retries=5,
            retry_delay=10.0,
            max_concurrent=20,
            system_prompt="You are a helpful assistant.",
            response_format="json",
        )
        assert config.model_name == "gpt-4-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.top_p == 0.9
        assert config.timeout == 120
        assert config.num_retries == 5
        assert config.retry_delay == 10.0
        assert config.max_concurrent == 20
        assert config.system_prompt == "You are a helpful assistant."
        assert config.response_format == "json"

    def test_temperature_boundary_values(self):
        config_zero = ServerConfig(model_name="test", temperature=0.0)
        assert config_zero.temperature == 0.0

        config_high = ServerConfig(model_name="test", temperature=2.0)
        assert config_high.temperature == 2.0

    def test_openrouter_model_name(self):
        config = ServerConfig(model_name="anthropic/claude-3-opus")
        assert config.model_name == "anthropic/claude-3-opus"


class TestRequest:
    """Tests for Request dataclass."""

    def test_minimal_request(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = Request(messages=messages)
        assert request.messages == messages
        assert request.config is None
        assert request.question is None
        assert request.answer is None
        assert request.prediction is None
        assert request.context is None
        assert request.prompt_kwargs == {}

    def test_full_request(self):
        messages = [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "Evaluate this."},
        ]
        config = ServerConfig(model_name="gpt-4o")
        request = Request(
            messages=messages,
            config=config,
            question="What is 2+2?",
            answer="4",
            prediction="4",
            context="Math evaluation",
            prompt_kwargs={"format": "json"},
        )
        assert request.messages == messages
        assert request.config == config
        assert request.question == "What is 2+2?"
        assert request.answer == "4"
        assert request.prediction == "4"
        assert request.context == "Math evaluation"
        assert request.prompt_kwargs == {"format": "json"}

    def test_request_messages_are_mutable(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = Request(messages=messages)
        request.messages.append({"role": "assistant", "content": "Hi"})
        assert len(request.messages) == 2

    def test_prompt_kwargs_default_factory(self):
        request1 = Request(messages=[])
        request2 = Request(messages=[])
        request1.prompt_kwargs["key"] = "value"
        assert "key" not in request2.prompt_kwargs


class TestResponse:
    """Tests for Response dataclass."""

    def test_minimal_response(self):
        response = Response(content="0.8", model_used="gpt-4o")
        assert response.content == "0.8"
        assert response.model_used == "gpt-4o"
        assert response.usage is None
        assert response.raw_response is None
        assert response.parsed_result is None
        assert response.success is True
        assert response.error_message is None

    def test_successful_response(self):
        response = Response(
            content="The answer is correct.",
            model_used="gpt-4o",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            parsed_result=0.85,
            success=True,
        )
        assert response.content == "The answer is correct."
        assert response.usage["total_tokens"] == 150
        assert response.parsed_result == 0.85
        assert response.success is True

    def test_failed_response(self):
        response = Response(
            content="",
            model_used="gpt-4o",
            success=False,
            error_message="Rate limit exceeded",
        )
        assert response.content == ""
        assert response.success is False
        assert response.error_message == "Rate limit exceeded"

    def test_parsed_result_types(self):
        response_float = Response(content="", model_used="", parsed_result=0.75)
        assert response_float.parsed_result == 0.75

        response_int = Response(content="", model_used="", parsed_result=8)
        assert response_int.parsed_result == 8

        response_bool = Response(content="", model_used="", parsed_result=True)
        assert response_bool.parsed_result is True

        response_dict = Response(
            content="", model_used="", parsed_result={"score": 0.9, "reason": "Good"}
        )
        assert response_dict.parsed_result["score"] == 0.9

    def test_raw_response_storage(self):
        raw = {"id": "chatcmpl-123", "object": "chat.completion"}
        response = Response(content="test", model_used="gpt-4o", raw_response=raw)
        assert response.raw_response["id"] == "chatcmpl-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
