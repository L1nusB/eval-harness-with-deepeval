"""OpenRouter API provider implementation for LLM judge."""

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp
import requests

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)


class OpenRouterProvider(ServerInterface):
    """OpenRouter API implementation with sync and async support.

    OpenRouter provides access to multiple LLM providers through a unified API,
    enabling use of GPT-4, Claude, Llama, and other models as judges.

    Environment Variables:
        OPENROUTER_API_KEY: API key for authentication (required)
        OPENROUTER_SITE_URL: Optional site URL for rankings
        OPENROUTER_APP_NAME: Optional app name for rankings

    Model Name Format: "provider/model-name", e.g.:
        - "anthropic/claude-3-opus"
        - "anthropic/claude-3-haiku"
        - "meta-llama/llama-3-70b-instruct"
        - "openai/gpt-4o"

    Example:
        >>> config = ServerConfig(model_name="anthropic/claude-3-haiku")
        >>> provider = OpenRouterProvider(config)
        >>> if provider.is_available():
        ...     response = provider.evaluate(request)
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: ServerConfig | None = None):
        if config is None:
            config = ServerConfig(model_name="anthropic/claude-3-haiku")
        super().__init__(config)

        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "lm-evaluation-harness")

    def is_available(self) -> bool:
        """Check if OpenRouter provider is available."""
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

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
        return payload

    def evaluate(self, request: Request) -> Response:
        """Synchronous evaluation."""
        if not self.is_available():
            return Response(
                content="",
                model_used="",
                success=False,
                error_message=(
                    "OpenRouter API key not configured. "
                    "Set OPENROUTER_API_KEY env var."
                ),
            )

        config = request.config or self.config
        payload = self._build_payload(request)
        url = f"{self.API_BASE}/chat/completions"

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=config.timeout,
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"]
                model_used = result.get("model", config.model_name)
                usage = result.get("usage")

                return Response(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    success=True,
                )

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(
                    f"OpenRouter attempt {attempt + 1}/{config.num_retries} failed: "
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
                    "OpenRouter API key not configured. "
                    "Set OPENROUTER_API_KEY env var."
                ),
            )

        config = request.config or self.config
        payload = self._build_payload(request)
        url = f"{self.API_BASE}/chat/completions"

        last_error = None
        for attempt in range(config.num_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self._get_headers(),
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout),
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                content = result["choices"][0]["message"]["content"]
                model_used = result.get("model", config.model_name)
                usage = result.get("usage")

                return Response(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    success=True,
                )

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(
                    f"OpenRouter async attempt {attempt + 1}/{config.num_retries} "
                    f"failed: {last_error}"
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
