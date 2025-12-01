"""LLM-as-a-Judge infrastructure for lm-evaluation-harness.

This package provides a general-purpose adapter for calling external LLMs as
judges for evaluation metrics. It supports multiple providers (OpenAI, OpenRouter)
and both synchronous and asynchronous evaluation.

Example usage:
    >>> from lm_eval.llm_judge import ProviderFactory, ServerConfig, Request
    >>>
    >>> # Create a provider
    >>> config = ServerConfig(model_name="gpt-4o", temperature=0.0)
    >>> provider = ProviderFactory.create_provider("openai", config)
    >>>
    >>> # Create and evaluate a request
    >>> request = Request(
    ...     messages=[{"role": "user", "content": "Evaluate this response..."}]
    ... )
    >>> response = provider.evaluate(request)
    >>>
    >>> if response.success:
    ...     print(f"Judge response: {response.content}")
"""

from .base import ServerInterface
from .factory import ProviderFactory
from .protocol import Request, Response, ServerConfig

__all__ = [
    "Request",
    "Response",
    "ServerConfig",
    "ServerInterface",
    "ProviderFactory",
]
