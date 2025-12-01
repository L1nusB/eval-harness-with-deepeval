"""Factory for creating LLM judge provider instances.

This factory hides import-time errors (missing packages, missing API keys) and
provides a single entrypoint for constructing judge providers based on
configuration or environment variables.
"""

import logging
import os

from .base import ServerInterface
from .protocol import ServerConfig


eval_logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating judge provider instances based on configuration.

    Lazily loads built-in providers to avoid import errors when optional
    dependencies are not installed.
    """

    _provider_classes: dict[str, type[ServerInterface]] = {}

    @classmethod
    def _lazy_load_providers(cls) -> None:
        """Lazily load built-in providers to avoid import errors."""
        if cls._provider_classes:
            return

        try:
            from .providers.openai import OpenAIProvider

            cls._provider_classes["openai"] = OpenAIProvider
        except ImportError as e:
            eval_logger.debug(f"OpenAI provider not available: {e}")

        try:
            from .providers.openrouter import OpenRouterProvider

            cls._provider_classes["openrouter"] = OpenRouterProvider
        except ImportError as e:
            eval_logger.debug(f"OpenRouter provider not available: {e}")

    @classmethod
    def create_provider(
        cls,
        provider_type: str | None = None,
        config: ServerConfig | None = None,
    ) -> ServerInterface:
        """Create a provider instance.

        Provider selection precedence:
        1. Explicit provider_type argument
        2. JUDGE_API_TYPE environment variable
        3. Default to "openai"

        Args:
            provider_type: Provider type name (e.g., "openai", "openrouter")
            config: Optional ServerConfig to use

        Returns:
            Configured provider instance

        Raises:
            ValueError: If the requested provider type is not available
        """
        cls._lazy_load_providers()

        if provider_type is None:
            provider_type = os.getenv("JUDGE_API_TYPE", "openai")

        provider_type = provider_type.lower()

        if provider_type not in cls._provider_classes:
            available = list(cls._provider_classes.keys())
            raise ValueError(
                f"Unknown provider type: '{provider_type}'. "
                f"Available providers: {available}"
            )

        provider_class = cls._provider_classes[provider_type]
        return provider_class(config)

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[ServerInterface]
    ) -> None:
        """Register a custom provider class.

        Args:
            name: Name to register the provider under
            provider_class: Provider class implementing ServerInterface
        """
        cls._provider_classes[name.lower()] = provider_class

    @classmethod
    def available_providers(cls) -> list[str]:
        """List available provider names.

        Returns:
            List of registered provider names
        """
        cls._lazy_load_providers()
        return list(cls._provider_classes.keys())
