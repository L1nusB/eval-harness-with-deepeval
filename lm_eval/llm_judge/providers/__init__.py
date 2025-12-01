"""LLM judge provider implementations.

This module provides concrete implementations of the ServerInterface for
different LLM API providers.

Available providers:
    - OpenAIProvider: OpenAI API (GPT-4, GPT-4o, etc.)
    - OpenRouterProvider: OpenRouter API (Claude, Llama, GPT-4, etc.)
"""

from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "OpenAIProvider",
    "OpenRouterProvider",
]
