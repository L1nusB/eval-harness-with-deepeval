"""OpenAI API provider implementation for LLM judge."""

import asyncio
import logging
import os
import time
from typing import Any

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    eval_logger.debug("OpenAI package not installed. OpenAI provider unavailable.")


class OpenAIProvider(ServerInterface):
    """OpenAI API implementation with sync and async support.

    Environment Variables:
        OPENAI_API_KEY: API key for authentication (required)
        OPENAI_API_BASE: Custom API base URL (optional)

    Example:
        >>> config = ServerConfig(model_name="gpt-4o", temperature=0.0)
        >>> provider = OpenAIProvider(config)
        >>> if provider.is_available():
        ...     response = provider.evaluate(request)
    """

    def __init__(self, config: ServerConfig | None = None):
        if config is None:
            config = ServerConfig(model_name="gpt-4o")
        super().__init__(config)

        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE")

        self._client: "OpenAI | None" = None
        self._async_client: "AsyncOpenAI | None" = None

        if OPENAI_AVAILABLE and self.api_key:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            self._client = OpenAI(**client_kwargs)
            self._async_client = AsyncOpenAI(**client_kwargs)

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return OPENAI_AVAILABLE and bool(self.api_key) and self._client is not None

    def _build_payload(self, request: Request) -> dict[str, Any]:
        """Build the API payload from request."""
        config = request.config or self.config
        messages = self.prepare_messages(request)

        payload: dict[str, Any] = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def evaluate(self, request: Request) -> Response:
        """Synchronous evaluation."""
        if not self.is_available():
            return Response(
                content="",
                model_used="",
                success=False,
                error_message=(
                    "OpenAI API key not configured. Set OPENAI_API_KEY env var."
                ),
            )

        config = request.config or self.config
        payload = self._build_payload(request)

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = self._client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                model_used = response.model
                usage = (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else None
                )

                return Response(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    success=True,
                )

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(
                    f"OpenAI attempt {attempt + 1}/{config.num_retries} failed: "
                    f"{last_error}"
                )
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)

        return Response(
            content="",
            model_used=config.model_name,
            success=False,
            error_message=(
                f"All {config.num_retries} attempts failed. Last error: {last_error}"
            ),
        )

    async def evaluate_async(self, request: Request) -> Response:
        """Asynchronous evaluation."""
        if not self.is_available():
            return Response(
                content="",
                model_used="",
                success=False,
                error_message=(
                    "OpenAI API key not configured. Set OPENAI_API_KEY env var."
                ),
            )

        config = request.config or self.config
        payload = self._build_payload(request)

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = await self._async_client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                model_used = response.model
                usage = (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else None
                )

                return Response(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    success=True,
                )

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(
                    f"OpenAI async attempt {attempt + 1}/{config.num_retries} failed: "
                    f"{last_error}"
                )
                if attempt < config.num_retries - 1:
                    await asyncio.sleep(config.retry_delay)

        return Response(
            content="",
            model_used=config.model_name,
            success=False,
            error_message=(
                f"All {config.num_retries} attempts failed. Last error: {last_error}"
            ),
        )
