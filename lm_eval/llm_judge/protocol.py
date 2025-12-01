"""Data classes for LLM judge communication protocol.

This module defines the core data structures that provide a provider-agnostic
contract between judge-backed metrics and concrete providers like OpenAI and
OpenRouter.
"""

from dataclasses import dataclass, field
from typing import Any


DEFAULT_NUM_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0  # seconds
DEFAULT_TIMEOUT = 60  # seconds


@dataclass
class ServerConfig:
    """Configuration for LLM judge providers.

    This dataclass represents the default configuration for a judge provider
    instance. Metric wrappers may override these defaults per-call by passing
    a config override on the Request.

    Attributes:
        model_name: Model identifier (e.g., "gpt-4o", "anthropic/claude-3-opus")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds
        num_retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        max_concurrent: Maximum concurrent async requests (for rate limiting).
            Enforced via asyncio.Semaphore in ServerInterface.
        system_prompt: Optional system prompt for all requests
        response_format: Expected response format ('json' or 'text')
    """

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float | None = None
    timeout: int = DEFAULT_TIMEOUT
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    max_concurrent: int = 10
    system_prompt: str | None = None
    response_format: str | None = None


@dataclass
class Request:
    """Standard request format for judge evaluation.

    This dataclass carries both the raw messages payload (chat-style) that will
    be sent to the provider, and structured fields that are useful for logging
    and for building prompts via helper utilities.

    Attributes:
        messages: List of message dicts with 'role' and 'content' keys
        config: Optional per-request config overrides
        question: Original question/prompt (for structured evaluation)
        answer: Ground truth answer
        prediction: Model prediction to evaluate
        context: Additional context for evaluation
        prompt_kwargs: Additional keyword arguments for prompt building
    """

    messages: list[dict[str, Any]]
    config: ServerConfig | None = None
    question: str | None = None
    answer: str | None = None
    prediction: str | None = None
    context: str | None = None
    prompt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Standard response format from judge evaluation.

    This dataclass separates raw text content from structured interpretation
    and status information, so metrics can reliably detect and react to
    failures instead of relying on exceptions.

    Attributes:
        content: Raw text content from the judge
        model_used: Actual model that processed the request
        usage: Token usage statistics (if available)
        raw_response: Original API response object
        parsed_result: Extracted score/result after parsing
        success: Whether the evaluation succeeded
        error_message: Error description if success is False
    """

    content: str
    model_used: str
    usage: dict[str, int] | None = None
    raw_response: Any | None = None
    parsed_result: int | float | bool | dict[str, Any] | None = None
    success: bool = True
    error_message: str | None = None
