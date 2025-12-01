"""Unit tests for lm_eval.llm_judge.providers module."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.llm_judge.protocol import Request, Response, ServerConfig

pytestmark = pytest.mark.anyio


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def setup_method(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)

    def test_import_without_api_key(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider is not None

    def test_is_available_without_key(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.is_available() is False

    def test_is_available_with_key(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            assert provider.is_available() is True

    def test_default_model_name(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.config.model_name == "gpt-4o"

    def test_custom_config(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        config = ServerConfig(model_name="gpt-4-turbo", temperature=0.7)
        provider = OpenAIProvider(config)
        assert provider.config.model_name == "gpt-4-turbo"
        assert provider.config.temperature == 0.7

    def test_evaluate_returns_error_when_unavailable(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        response = provider.evaluate(request)
        assert response.success is False
        assert "OPENAI_API_KEY" in response.error_message

    async def test_evaluate_async_returns_error_when_unavailable(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.evaluate_async(request)
        assert response.success is False
        assert "OPENAI_API_KEY" in response.error_message

    def test_build_payload_basic(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            request = Request(messages=[{"role": "user", "content": "Test"}])
            payload = provider._build_payload(request)

            assert payload["model"] == "gpt-4o"
            assert payload["messages"] == [{"role": "user", "content": "Test"}]
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 1024

    def test_build_payload_with_top_p(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        config = ServerConfig(model_name="gpt-4o", top_p=0.9)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(config)
            request = Request(messages=[{"role": "user", "content": "Test"}])
            payload = provider._build_payload(request)

            assert payload["top_p"] == 0.9

    def test_build_payload_with_json_format(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        config = ServerConfig(model_name="gpt-4o", response_format="json")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(config)
            request = Request(messages=[{"role": "user", "content": "Test"}])
            payload = provider._build_payload(request)

            assert payload["response_format"] == {"type": "json_object"}

    def test_evaluate_with_mocked_client(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "0.85"
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.model_dump.return_value = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

            provider._client = MagicMock()
            provider._client.chat.completions.create.return_value = mock_response

            request = Request(messages=[{"role": "user", "content": "Test"}])
            response = provider.evaluate(request)

            assert response.success is True
            assert response.content == "0.85"
            assert response.model_used == "gpt-4o"

    def test_evaluate_retries_on_failure(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        config = ServerConfig(
            model_name="gpt-4o", num_retries=3, retry_delay=0.01
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(config)

            provider._client = MagicMock()
            provider._client.chat.completions.create.side_effect = Exception(
                "API Error"
            )

            request = Request(messages=[{"role": "user", "content": "Test"}])
            response = provider.evaluate(request)

            assert response.success is False
            assert "All 3 attempts failed" in response.error_message
            assert provider._client.chat.completions.create.call_count == 3

    async def test_evaluate_async_with_mocked_client(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "0.9"
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.model_dump.return_value = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

            provider._async_client = MagicMock()
            provider._async_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            request = Request(messages=[{"role": "user", "content": "Test"}])
            response = await provider.evaluate_async(request)

            assert response.success is True
            assert response.content == "0.9"


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def setup_method(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENROUTER_SITE_URL", None)
        os.environ.pop("OPENROUTER_APP_NAME", None)

    def test_import_without_api_key(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        assert provider is not None

    def test_is_available_without_key(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        assert provider.is_available() is False

    def test_is_available_with_key(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider()
            assert provider.is_available() is True

    def test_default_model_name(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        assert provider.config.model_name == "anthropic/claude-3-haiku"

    def test_custom_config(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        config = ServerConfig(model_name="openai/gpt-4o", temperature=0.5)
        provider = OpenRouterProvider(config)
        assert provider.config.model_name == "openai/gpt-4o"
        assert provider.config.temperature == 0.5

    def test_get_headers(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_SITE_URL": "https://example.com",
                "OPENROUTER_APP_NAME": "test-app",
            },
        ):
            provider = OpenRouterProvider()
            headers = provider._get_headers()

            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "test-app"

    def test_get_headers_without_site_url(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider()
            headers = provider._get_headers()

            assert "HTTP-Referer" not in headers or headers.get("HTTP-Referer") == ""

    def test_evaluate_returns_error_when_unavailable(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        response = provider.evaluate(request)
        assert response.success is False
        assert "OPENROUTER_API_KEY" in response.error_message

    async def test_evaluate_async_returns_error_when_unavailable(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.evaluate_async(request)
        assert response.success is False
        assert "OPENROUTER_API_KEY" in response.error_message

    def test_build_payload_basic(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider()
            request = Request(messages=[{"role": "user", "content": "Test"}])
            payload = provider._build_payload(request)

            assert payload["model"] == "anthropic/claude-3-haiku"
            assert payload["messages"] == [{"role": "user", "content": "Test"}]
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 1024

    def test_build_payload_with_top_p(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        config = ServerConfig(model_name="anthropic/claude-3-opus", top_p=0.95)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider(config)
            request = Request(messages=[{"role": "user", "content": "Test"}])
            payload = provider._build_payload(request)

            assert payload["top_p"] == 0.95

    def test_evaluate_with_mocked_requests(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "0.75"}}],
                "model": "anthropic/claude-3-haiku",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            mock_response.raise_for_status = MagicMock()

            with patch("requests.post", return_value=mock_response):
                request = Request(messages=[{"role": "user", "content": "Test"}])
                response = provider.evaluate(request)

                assert response.success is True
                assert response.content == "0.75"
                assert response.model_used == "anthropic/claude-3-haiku"

    def test_evaluate_retries_on_failure(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        config = ServerConfig(
            model_name="anthropic/claude-3-haiku",
            num_retries=2,
            retry_delay=0.01,
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider(config)

            with patch(
                "requests.post", side_effect=Exception("Connection error")
            ) as mock_post:
                request = Request(messages=[{"role": "user", "content": "Test"}])
                response = provider.evaluate(request)

                assert response.success is False
                assert "All 2 attempts failed" in response.error_message
                assert mock_post.call_count == 2

    async def test_evaluate_async_with_mocked_aiohttp(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider()

            mock_response_data = {
                "choices": [{"message": {"content": "0.8"}}],
                "model": "anthropic/claude-3-haiku",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_response.raise_for_status = MagicMock()

            mock_session = MagicMock()
            mock_session.post = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=None),
                )
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                request = Request(messages=[{"role": "user", "content": "Test"}])
                response = await provider.evaluate_async(request)

                assert response.success is True
                assert response.content == "0.8"


class TestProviderSystemPrompt:
    """Tests for system prompt handling across providers."""

    def test_openai_adds_system_prompt(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider

        config = ServerConfig(
            model_name="gpt-4o",
            system_prompt="You are a helpful assistant.",
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(config)
            request = Request(messages=[{"role": "user", "content": "Hello"}])
            messages = provider.prepare_messages(request)

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    def test_openrouter_adds_system_prompt(self):
        from lm_eval.llm_judge.providers.openrouter import OpenRouterProvider

        config = ServerConfig(
            model_name="anthropic/claude-3-haiku",
            system_prompt="You are a judge.",
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider(config)
            request = Request(messages=[{"role": "user", "content": "Evaluate"}])
            messages = provider.prepare_messages(request)

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a judge."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
