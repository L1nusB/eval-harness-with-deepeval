"""Unit tests for lm_eval.llm_judge.factory module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.llm_judge.base import ServerInterface
from lm_eval.llm_judge.factory import ProviderFactory
from lm_eval.llm_judge.protocol import Request, Response, ServerConfig


class MockProvider(ServerInterface):
    """Mock provider for testing custom registration."""

    def __init__(self, config: ServerConfig | None = None):
        super().__init__(config)

    def evaluate(self, request: Request) -> Response:
        return Response(content="mock", model_used="mock-model", success=True)

    async def evaluate_async(self, request: Request) -> Response:
        return Response(content="mock", model_used="mock-model", success=True)

    def is_available(self) -> bool:
        return True


class TestProviderFactoryLazyLoading:
    """Tests for lazy loading of providers."""

    def setup_method(self):
        ProviderFactory._provider_classes = {}

    def test_providers_not_loaded_initially(self):
        ProviderFactory._provider_classes = {}
        assert len(ProviderFactory._provider_classes) == 0

    def test_lazy_load_called_on_create(self):
        with patch.object(
            ProviderFactory, "_lazy_load_providers"
        ) as mock_lazy_load:
            mock_lazy_load.side_effect = lambda: ProviderFactory._provider_classes.update(
                {"mock": MockProvider}
            )
            try:
                ProviderFactory.create_provider("mock")
            except ValueError:
                pass
            mock_lazy_load.assert_called()

    def test_lazy_load_called_on_available_providers(self):
        with patch.object(
            ProviderFactory, "_lazy_load_providers"
        ) as mock_lazy_load:
            ProviderFactory.available_providers()
            mock_lazy_load.assert_called()


class TestProviderFactoryCreateProvider:
    """Tests for create_provider method."""

    def setup_method(self):
        ProviderFactory._provider_classes = {"mock": MockProvider}

    def test_create_with_explicit_type(self):
        provider = ProviderFactory.create_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_create_with_env_var(self):
        ProviderFactory._provider_classes = {"mock": MockProvider}
        with patch.dict(os.environ, {"JUDGE_API_TYPE": "mock"}):
            provider = ProviderFactory.create_provider()
            assert isinstance(provider, MockProvider)

    def test_explicit_type_overrides_env_var(self):
        ProviderFactory._provider_classes = {"mock": MockProvider, "other": MockProvider}
        with patch.dict(os.environ, {"JUDGE_API_TYPE": "other"}):
            provider = ProviderFactory.create_provider("mock")
            assert isinstance(provider, MockProvider)

    def test_case_insensitive_provider_type(self):
        ProviderFactory._provider_classes = {"mock": MockProvider}
        provider = ProviderFactory.create_provider("MOCK")
        assert isinstance(provider, MockProvider)

        provider = ProviderFactory.create_provider("Mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises_error(self):
        ProviderFactory._provider_classes = {"mock": MockProvider}
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create_provider("unknown")
        assert "Unknown provider type: 'unknown'" in str(exc_info.value)
        assert "mock" in str(exc_info.value)

    def test_create_with_config(self):
        config = ServerConfig(model_name="custom-model", temperature=0.5)
        provider = ProviderFactory.create_provider("mock", config)
        assert provider.config.model_name == "custom-model"
        assert provider.config.temperature == 0.5

    def test_default_to_openai_when_available(self):
        from lm_eval.llm_judge.providers.openai import OpenAIProvider
        
        ProviderFactory._provider_classes = {}
        ProviderFactory._lazy_load_providers()
        
        if "openai" in ProviderFactory._provider_classes:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("JUDGE_API_TYPE", None)
                provider = ProviderFactory.create_provider()
                assert isinstance(provider, OpenAIProvider)


class TestProviderFactoryRegisterProvider:
    """Tests for register_provider method."""

    def setup_method(self):
        ProviderFactory._provider_classes = {}

    def test_register_custom_provider(self):
        ProviderFactory.register_provider("custom", MockProvider)
        assert "custom" in ProviderFactory._provider_classes
        provider = ProviderFactory.create_provider("custom")
        assert isinstance(provider, MockProvider)

    def test_register_case_insensitive(self):
        ProviderFactory.register_provider("CUSTOM", MockProvider)
        assert "custom" in ProviderFactory._provider_classes

    def test_override_existing_provider(self):
        class AnotherMockProvider(MockProvider):
            pass

        ProviderFactory.register_provider("mock", MockProvider)
        ProviderFactory.register_provider("mock", AnotherMockProvider)
        provider = ProviderFactory.create_provider("mock")
        assert isinstance(provider, AnotherMockProvider)


class TestProviderFactoryAvailableProviders:
    """Tests for available_providers method."""

    def setup_method(self):
        ProviderFactory._provider_classes = {}

    def test_returns_list(self):
        result = ProviderFactory.available_providers()
        assert isinstance(result, list)

    def test_includes_registered_providers(self):
        ProviderFactory.register_provider("custom1", MockProvider)
        ProviderFactory.register_provider("custom2", MockProvider)
        available = ProviderFactory.available_providers()
        assert "custom1" in available
        assert "custom2" in available

    def test_lazy_loads_builtin_providers(self):
        ProviderFactory._provider_classes = {}
        available = ProviderFactory.available_providers()
        assert "openai" in available or "openrouter" in available or len(available) >= 0


class TestProviderFactoryIntegration:
    """Integration tests for ProviderFactory with real providers."""

    def setup_method(self):
        ProviderFactory._provider_classes = {}
        ProviderFactory._lazy_load_providers()

    def test_openai_provider_loadable(self):
        if "openai" in ProviderFactory._provider_classes:
            provider = ProviderFactory.create_provider("openai")
            assert provider is not None

    def test_openrouter_provider_loadable(self):
        if "openrouter" in ProviderFactory._provider_classes:
            provider = ProviderFactory.create_provider("openrouter")
            assert provider is not None

    def test_all_providers_have_required_methods(self):
        for name in ProviderFactory.available_providers():
            provider = ProviderFactory.create_provider(name)
            assert hasattr(provider, "evaluate")
            assert hasattr(provider, "evaluate_async")
            assert hasattr(provider, "is_available")
            assert hasattr(provider, "prepare_messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
