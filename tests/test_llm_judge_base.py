"""Unit tests for lm_eval.llm_judge.base module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.llm_judge.base import ServerInterface
from lm_eval.llm_judge.protocol import Request, Response, ServerConfig

pytestmark = pytest.mark.anyio


class ConcreteProvider(ServerInterface):
    """Concrete implementation for testing abstract ServerInterface."""

    def __init__(self, config: ServerConfig | None = None, available: bool = True):
        super().__init__(config)
        self._available = available
        self._evaluate_response = Response(
            content="0.85", model_used="test-model", success=True
        )

    def evaluate(self, request: Request) -> Response:
        return self._evaluate_response

    async def evaluate_async(self, request: Request) -> Response:
        return self._evaluate_response

    def is_available(self) -> bool:
        return self._available

    def set_response(self, response: Response):
        self._evaluate_response = response


class TestServerInterfaceInit:
    """Tests for ServerInterface initialization."""

    def test_default_config(self):
        provider = ConcreteProvider()
        assert provider.config.model_name == "gpt-4o"
        assert provider.config.temperature == 0.0
        assert provider.config.max_tokens == 1024

    def test_custom_config(self):
        config = ServerConfig(model_name="custom-model", temperature=0.5)
        provider = ConcreteProvider(config)
        assert provider.config.model_name == "custom-model"
        assert provider.config.temperature == 0.5

    def test_semaphore_not_initialized_on_creation(self):
        provider = ConcreteProvider()
        assert provider._semaphore is None


class TestServerInterfaceSemaphore:
    """Tests for semaphore property."""

    def test_lazy_initialization(self):
        provider = ConcreteProvider()
        assert provider._semaphore is None
        _ = provider.semaphore
        assert provider._semaphore is not None

    def test_semaphore_uses_config_max_concurrent(self):
        config = ServerConfig(model_name="test", max_concurrent=5)
        provider = ConcreteProvider(config)
        semaphore = provider.semaphore
        assert isinstance(semaphore, asyncio.Semaphore)

    def test_semaphore_cached(self):
        provider = ConcreteProvider()
        sem1 = provider.semaphore
        sem2 = provider.semaphore
        assert sem1 is sem2

    def test_default_max_concurrent(self):
        provider = ConcreteProvider()
        assert provider.config.max_concurrent == 10


class TestServerInterfacePrepareMessages:
    """Tests for prepare_messages method."""

    def test_messages_copied(self):
        provider = ConcreteProvider()
        original = [{"role": "user", "content": "Hello"}]
        request = Request(messages=original)
        result = provider.prepare_messages(request)
        assert result == original
        assert result is not original

    def test_system_prompt_added(self):
        config = ServerConfig(
            model_name="test",
            system_prompt="You are a helpful assistant.",
        )
        provider = ConcreteProvider(config)
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        result = provider.prepare_messages(request)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_system_prompt_not_duplicated(self):
        config = ServerConfig(
            model_name="test",
            system_prompt="Default system prompt",
        )
        provider = ConcreteProvider(config)
        request = Request(
            messages=[
                {"role": "system", "content": "Custom system prompt"},
                {"role": "user", "content": "Hello"},
            ]
        )
        result = provider.prepare_messages(request)
        assert len(result) == 2
        assert result[0]["content"] == "Custom system prompt"

    def test_no_system_prompt_in_config(self):
        provider = ConcreteProvider()
        request = Request(messages=[{"role": "user", "content": "Hello"}])
        result = provider.prepare_messages(request)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestServerInterfaceEvaluateBatchAsync:
    """Tests for evaluate_batch_async method."""

    async def test_batch_evaluation(self):
        provider = ConcreteProvider()
        requests = [
            Request(messages=[{"role": "user", "content": f"Request {i}"}])
            for i in range(3)
        ]
        results = await provider.evaluate_batch_async(requests)
        assert len(results) == 3
        assert all(r.success for r in results)

    async def test_batch_respects_semaphore(self):
        config = ServerConfig(model_name="test", max_concurrent=2)
        provider = ConcreteProvider(config)

        call_order = []
        active_count = [0]
        max_active = [0]

        original_evaluate = provider.evaluate_async

        async def tracked_evaluate(request):
            active_count[0] += 1
            max_active[0] = max(max_active[0], active_count[0])
            call_order.append("start")
            await asyncio.sleep(0.01)
            result = await original_evaluate(request)
            active_count[0] -= 1
            call_order.append("end")
            return result

        provider.evaluate_async = tracked_evaluate

        requests = [
            Request(messages=[{"role": "user", "content": f"Request {i}"}])
            for i in range(5)
        ]
        await provider.evaluate_batch_async(requests)

        assert max_active[0] <= 2

    async def test_empty_batch(self):
        provider = ConcreteProvider()
        results = await provider.evaluate_batch_async([])
        assert results == []


class TestServerInterfaceEvaluateScore:
    """Tests for evaluate_score method."""

    def test_basic_score_evaluation(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.85", model_used="test-model", success=True)
        )
        result = provider.evaluate_score(
            question="What is 2+2?",
            prediction="4",
        )
        assert "score" in result
        assert result["score"] == 0.85
        assert result["success"] is True
        assert result["model"] == "test-model"

    def test_score_with_answer(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.9", model_used="test-model", success=True)
        )
        result = provider.evaluate_score(
            question="Capital of France?",
            prediction="Paris",
            answer="Paris",
        )
        assert result["score"] == 0.9

    def test_score_with_context(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.75", model_used="test-model", success=True)
        )
        result = provider.evaluate_score(
            question="Summarize the text",
            prediction="This is a summary",
            context="Original text content",
        )
        assert result["score"] == 0.75

    def test_score_with_custom_template(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.8", model_used="test-model", success=True)
        )
        template = "Q: {question}\nA: {prediction}"
        result = provider.evaluate_score(
            question="Test",
            prediction="Answer",
            prompt_template=template,
        )
        assert result["score"] == 0.8

    def test_score_parsing_text_response(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(
                content="The score for this response is 0.7 out of 1.0",
                model_used="test-model",
                success=True,
            )
        )
        result = provider.evaluate_score(
            question="Q",
            prediction="P",
        )
        assert result["score"] == 0.7

    def test_failed_evaluation(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(
                content="",
                model_used="test-model",
                success=False,
                error_message="API error",
            )
        )
        result = provider.evaluate_score(
            question="Q",
            prediction="P",
        )
        assert result["success"] is False
        assert result["score"] == 0.0


class TestServerInterfaceEvaluateScoreAsync:
    """Tests for evaluate_score_async method."""

    async def test_basic_async_score(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.9", model_used="test-model", success=True)
        )
        result = await provider.evaluate_score_async(
            question="What is 2+2?",
            prediction="4",
        )
        assert result["score"] == 0.9
        assert result["success"] is True

    async def test_async_score_with_all_params(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(content="0.85", model_used="test-model", success=True)
        )
        result = await provider.evaluate_score_async(
            question="Q",
            prediction="P",
            answer="A",
            context="C",
            prompt_template="{question} - {prediction}",
        )
        assert result["score"] == 0.85

    async def test_async_failed_evaluation(self):
        provider = ConcreteProvider()
        provider.set_response(
            Response(
                content="",
                model_used="test-model",
                success=False,
                error_message="Timeout",
            )
        )
        result = await provider.evaluate_score_async(
            question="Q",
            prediction="P",
        )
        assert result["success"] is False


class TestServerInterfaceAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            ServerInterface()

    def test_subclass_must_implement_evaluate(self):
        class IncompleteProvider(ServerInterface):
            async def evaluate_async(self, request):
                pass

            def is_available(self):
                return True

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_subclass_must_implement_evaluate_async(self):
        class IncompleteProvider(ServerInterface):
            def evaluate(self, request):
                pass

            def is_available(self):
                return True

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_subclass_must_implement_is_available(self):
        class IncompleteProvider(ServerInterface):
            def evaluate(self, request):
                pass

            async def evaluate_async(self, request):
                pass

        with pytest.raises(TypeError):
            IncompleteProvider()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
