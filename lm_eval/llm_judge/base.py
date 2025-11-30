"""Abstract base classes for LLM judge implementations.

This module defines the minimal contract that all judge providers must implement
so that higher-level code can call them uniformly.
"""

import abc
import asyncio
import logging
from typing import Any

from .protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)


class ServerInterface(abc.ABC):
    """Abstract base class for judge implementations with sync and async support.

    Subclasses must implement:
        - evaluate(request) -> Response (sync)
        - evaluate_async(request) -> Response (async)
        - is_available() -> bool

    The overall harness remains synchronous, so evaluate() must be a blocking
    call that returns a completed Response. evaluate_async() exposes an async
    variant for libraries or future callers that want to manage their own
    event loop and fine-grained scheduling.
    """

    def __init__(self, config: ServerConfig | None = None):
        """Initialize with optional default configuration."""
        self.config = config or ServerConfig(model_name="gpt-4o")
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore for rate limiting concurrent requests."""
        if self._semaphore is None:
            max_concurrent = getattr(self.config, "max_concurrent", 10)
            self._semaphore = asyncio.Semaphore(max_concurrent)
        return self._semaphore

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        """Synchronously evaluate the given request and return a response."""

    @abc.abstractmethod
    async def evaluate_async(self, request: Request) -> Response:
        """Asynchronously evaluate the given request and return a response."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the judge service is available (e.g., API key configured)."""

    def prepare_messages(self, request: Request) -> list[dict[str, Any]]:
        """Prepare messages in the format expected by the API.

        Adds system prompt if configured and not already present in messages.
        """
        messages = request.messages.copy()

        if self.config.system_prompt and not any(
            m.get("role") == "system" for m in messages
        ):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        return messages

    async def evaluate_batch_async(self, requests: list[Request]) -> list[Response]:
        """Evaluate multiple requests concurrently with rate limiting.

        Uses the semaphore from ServerConfig.max_concurrent to bound concurrency.
        Callers should prefer this rather than creating ad-hoc asyncio.gather
        calls, to respect provider rate limits.
        """

        async def _eval_with_semaphore(req: Request) -> Response:
            async with self.semaphore:
                return await self.evaluate_async(req)

        return await asyncio.gather(*[_eval_with_semaphore(req) for req in requests])

    def evaluate_score(
        self,
        question: str,
        prediction: str,
        answer: str | None = None,
        context: str | None = None,
        prompt_template: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method for scoring evaluation (sync).

        Builds a scoring prompt using JudgePromptBuilder, evaluates it,
        and parses the response to extract a numeric score.
        """
        from .utils import JudgePromptBuilder, ResponseParser

        prompt = JudgePromptBuilder.build_score_prompt(
            question=question,
            prediction=prediction,
            answer=answer,
            context=context,
            prompt_template=prompt_template,
            **kwargs,
        )

        request = Request(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            answer=answer,
            prediction=prediction,
            context=context,
            config=self.config,
        )

        response = self.evaluate(request)
        parsed_score = ResponseParser.parse_score_response(response.content, (0.0, 1.0))

        return {
            "score": parsed_score,
            "raw_response": response.content,
            "model": response.model_used,
            "success": response.success,
        }

    async def evaluate_score_async(
        self,
        question: str,
        prediction: str,
        answer: str | None = None,
        context: str | None = None,
        prompt_template: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method for scoring evaluation (async).

        Builds a scoring prompt using JudgePromptBuilder, evaluates it
        asynchronously, and parses the response to extract a numeric score.
        """
        from .utils import JudgePromptBuilder, ResponseParser

        prompt = JudgePromptBuilder.build_score_prompt(
            question=question,
            prediction=prediction,
            answer=answer,
            context=context,
            prompt_template=prompt_template,
            **kwargs,
        )

        request = Request(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            answer=answer,
            prediction=prediction,
            context=context,
            config=self.config,
        )

        response = await self.evaluate_async(request)
        parsed_score = ResponseParser.parse_score_response(response.content, (0.0, 1.0))

        return {
            "score": parsed_score,
            "raw_response": response.content,
            "model": response.model_used,
            "success": response.success,
        }
