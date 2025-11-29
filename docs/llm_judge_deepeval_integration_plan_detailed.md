# LLM-as-a-Judge + DeepEval Integration: Detailed Implementation Plan

## Executive Summary

This plan details integrating LLM-as-a-Judge capabilities and DeepEval metrics into lm-evaluation-harness. The integration enables evaluation of model outputs using external LLM judges (like GPT-4) while maintaining full compatibility with the existing metrics pipeline.

**Key Design Decisions:**
- **DeepEval is a required dependency** for LLM judge metrics
- **Providers**: OpenAI and OpenRouter (enables GPT-4, Claude, Llama, etc. as judges)
- **Async from the start**: Support async evaluation with batched API calls for performance
- All kwargs from YAML `metric_list` entries flow to `_metric_fn_kwargs[metric]`
- Metric functions receive `references=[gold], predictions=[result], **kwargs`
- Input context passed via YAML kwargs (consistent with framework), with `doc_to_input` enhancement planned

---

## Phase 2: LLM-Judge Adapter Package

### 2.1 Package Structure

```
lm_eval/llm_judge/
    __init__.py          # Public exports
    protocol.py          # ServerConfig, Request, Response dataclasses
    base.py              # ServerInterface ABC
    utils.py             # JudgePromptBuilder, ResponseParser helpers
    factory.py           # ProviderFactory for creating providers
    providers/
        __init__.py      # Provider exports
        openai.py        # OpenAI provider implementation
        openrouter.py    # OpenRouter provider implementation
```

### 2.2 File: `lm_eval/llm_judge/protocol.py`

**Purpose:** Define core data structures for judge communication.

```python
"""Data classes for LLM judge communication protocol."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


DEFAULT_NUM_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0  # seconds
DEFAULT_TIMEOUT = 60  # seconds


@dataclass
class ServerConfig:
    """Configuration for LLM judge providers.

    Attributes:
        model_name: Model identifier (e.g., "gpt-4o", "anthropic/claude-3-opus")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds
        num_retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        max_concurrent: Maximum concurrent async requests (for rate limiting)
        system_prompt: Optional system prompt for all requests
        response_format: Expected response format ('json' or 'text')
    """
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    timeout: int = DEFAULT_TIMEOUT
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    max_concurrent: int = 10  # For async batching rate limiting
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # 'json' or 'text'


@dataclass
class Request:
    """Standard request format for judge evaluation.

    Attributes:
        messages: List of message dicts with 'role' and 'content' keys
        config: Optional per-request config overrides
        question: Original question/prompt (for structured evaluation)
        answer: Ground truth answer
        prediction: Model prediction to evaluate
        context: Additional context for evaluation
    """
    messages: List[Dict[str, Any]]
    config: Optional[ServerConfig] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    prediction: Optional[str] = None
    context: Optional[str] = None
    prompt_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Standard response format from judge evaluation.

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
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    parsed_result: Optional[Union[int, float, bool, Dict[str, Any]]] = None
    success: bool = True
    error_message: Optional[str] = None
```

### 2.3 File: `lm_eval/llm_judge/base.py`

**Purpose:** Abstract base class defining both sync and async judge interfaces.

```python
"""Abstract base classes for LLM judge implementations."""
import abc
import asyncio
import logging
from typing import Any, Dict, List, Optional

from .protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)


class ServerInterface(abc.ABC):
    """Abstract base class for judge implementations with sync and async support.

    Subclasses must implement:
        - evaluate(request) -> Response (sync)
        - evaluate_async(request) -> Response (async)
        - is_available() -> bool
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize with optional default configuration."""
        self.config = config or ServerConfig(model_name="gpt-4o")
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore for rate limiting concurrent requests."""
        if self._semaphore is None:
            max_concurrent = getattr(self.config, 'max_concurrent', 10)
            self._semaphore = asyncio.Semaphore(max_concurrent)
        return self._semaphore

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        """Synchronously evaluate the given request and return a response."""
        pass

    @abc.abstractmethod
    async def evaluate_async(self, request: Request) -> Response:
        """Asynchronously evaluate the given request and return a response."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the judge service is available (e.g., API key configured)."""
        pass

    def prepare_messages(self, request: Request) -> List[Dict[str, Any]]:
        """Prepare messages in the format expected by the API."""
        messages = request.messages.copy()

        # Add system prompt if configured and not already present
        if self.config.system_prompt and not any(
            m.get("role") == "system" for m in messages
        ):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        return messages

    async def evaluate_batch_async(self, requests: List[Request]) -> List[Response]:
        """Evaluate multiple requests concurrently with rate limiting."""
        async def _eval_with_semaphore(req: Request) -> Response:
            async with self.semaphore:
                return await self.evaluate_async(req)

        return await asyncio.gather(*[_eval_with_semaphore(req) for req in requests])

    def evaluate_score(
        self,
        question: str,
        prediction: str,
        answer: Optional[str] = None,
        context: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for scoring evaluation (sync)."""
        from .utils import JudgePromptBuilder, ResponseParser

        prompt = JudgePromptBuilder.build_score_prompt(
            question=question, prediction=prediction, answer=answer,
            context=context, prompt_template=prompt_template, **kwargs
        )

        request = Request(
            messages=[{"role": "user", "content": prompt}],
            question=question, answer=answer, prediction=prediction,
            context=context, config=self.config,
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
        answer: Optional[str] = None,
        context: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for scoring evaluation (async)."""
        from .utils import JudgePromptBuilder, ResponseParser

        prompt = JudgePromptBuilder.build_score_prompt(
            question=question, prediction=prediction, answer=answer,
            context=context, prompt_template=prompt_template, **kwargs
        )

        request = Request(
            messages=[{"role": "user", "content": prompt}],
            question=question, answer=answer, prediction=prediction,
            context=context, config=self.config,
        )

        response = await self.evaluate_async(request)
        parsed_score = ResponseParser.parse_score_response(response.content, (0.0, 1.0))

        return {
            "score": parsed_score,
            "raw_response": response.content,
            "model": response.model_used,
            "success": response.success,
        }
```

### 2.4 File: `lm_eval/llm_judge/utils.py`

**Purpose:** Helper classes for prompt building and response parsing.

```python
"""Utility classes for LLM judge prompt building and response parsing."""
import json
import re
from typing import Any, Dict, Optional, Tuple


DEFAULT_SCORE_PROMPT = """You are an expert evaluator. Rate the quality of the given response.

Question/Prompt: {question}
{answer_section}
Response to evaluate: {prediction}
{context_section}

Provide a score from 0.0 to 1.0 where:
- 0.0 = Completely incorrect or irrelevant
- 0.5 = Partially correct with significant issues
- 1.0 = Fully correct and comprehensive

Output ONLY a single decimal number between 0.0 and 1.0."""


class JudgePromptBuilder:
    """Helper class to build prompts for different judge evaluation types."""

    @staticmethod
    def build_score_prompt(
        question: str,
        prediction: str,
        answer: Optional[str] = None,
        context: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build a scoring prompt."""
        if prompt_template:
            return prompt_template.format(
                question=question, prediction=prediction,
                answer=answer or "", context=context or "", **kwargs
            )

        answer_section = f"\nExpected Answer: {answer}" if answer else ""
        context_section = f"\nContext: {context}" if context else ""

        return DEFAULT_SCORE_PROMPT.format(
            question=question, prediction=prediction,
            answer_section=answer_section, context_section=context_section,
        )

    @staticmethod
    def build_geval_prompt(
        criteria: str,
        evaluation_steps: Optional[list] = None,
        input_text: Optional[str] = None,
        actual_output: str = "",
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        retrieval_context: Optional[str] = None,
    ) -> str:
        """Build a G-Eval style prompt with criteria and evaluation steps."""
        prompt_parts = [f"Evaluation Criteria: {criteria}\n"]

        if evaluation_steps:
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(evaluation_steps))
            prompt_parts.append(f"Evaluation Steps:\n{steps_text}\n")

        if input_text:
            prompt_parts.append(f"Input: {input_text}\n")

        prompt_parts.append(f"Actual Output: {actual_output}\n")

        if expected_output:
            prompt_parts.append(f"Expected Output: {expected_output}\n")
        if context:
            prompt_parts.append(f"Context: {context}\n")
        if retrieval_context:
            prompt_parts.append(f"Retrieval Context: {retrieval_context}\n")

        prompt_parts.append(
            "\nBased on the criteria above, provide a score from 0.0 to 1.0. "
            "Output ONLY a single decimal number."
        )

        return "\n".join(prompt_parts)


class ResponseParser:
    """Helper class to parse different types of judge responses."""

    @staticmethod
    def parse_score_response(response: str, score_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Parse a numeric score from response text."""
        try:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response.strip())
            if numbers:
                score = float(numbers[0])
                return max(score_range[0], min(score, score_range[1]))
        except (ValueError, IndexError):
            pass
        return score_range[0]

    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """Parse JSON from response text."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}
```

### 2.5 File: `lm_eval/llm_judge/factory.py`

**Purpose:** Factory pattern for creating provider instances.

```python
"""Factory for creating LLM judge provider instances."""
import logging
import os
from typing import Dict, Optional, Type

from .base import ServerInterface
from .protocol import ServerConfig


eval_logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating judge provider instances based on configuration."""

    _provider_classes: Dict[str, Type[ServerInterface]] = {}

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
        provider_type: Optional[str] = None,
        config: Optional[ServerConfig] = None,
    ) -> ServerInterface:
        """Create a judge provider instance."""
        cls._lazy_load_providers()

        if provider_type is None:
            provider_type = os.getenv("JUDGE_API_TYPE", "openai").lower()

        if provider_type not in cls._provider_classes:
            available = list(cls._provider_classes.keys())
            raise ValueError(
                f"Unknown provider type: '{provider_type}'. Available: {available}"
            )

        provider_class = cls._provider_classes[provider_type]
        return provider_class(config=config)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[ServerInterface]) -> None:
        """Register a custom provider implementation."""
        if not issubclass(provider_class, ServerInterface):
            raise ValueError(f"{provider_class.__name__} must subclass ServerInterface")
        cls._provider_classes[name] = provider_class

    @classmethod
    def available_providers(cls) -> list:
        """Get list of available provider names."""
        cls._lazy_load_providers()
        return list(cls._provider_classes.keys())
```

### 2.6 File: `lm_eval/llm_judge/providers/openai.py`

**Purpose:** OpenAI API provider implementation with sync and async support.

```python
"""OpenAI API provider implementation for LLM judge."""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)


class OpenAIProvider(ServerInterface):
    """OpenAI API implementation with sync and async support.

    Environment Variables:
        OPENAI_API_KEY: API key for authentication
        OPENAI_API_BASE: Base URL for API (optional, for proxies)
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(model_name="gpt-4o")
        super().__init__(config)

        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        self._client = None
        self._async_client = None
        try:
            from openai import OpenAI, AsyncOpenAI
            if self.api_key:
                self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        except ImportError:
            eval_logger.warning("OpenAI package not installed. Install with: pip install openai")

    def is_available(self) -> bool:
        return bool(self.api_key and self._client)

    def _build_payload(self, request: Request) -> Dict[str, Any]:
        """Build the API payload from request."""
        config = request.config or self.config
        messages = self.prepare_messages(request)

        payload = {
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
                content="", model_used="", success=False,
                error_message="OpenAI API key not configured. Set OPENAI_API_KEY env var.",
            )

        config = request.config or self.config
        payload = self._build_payload(request)

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = self._client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                model_used = response.model
                usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None

                return Response(content=content.strip(), model_used=model_used, usage=usage, success=True)

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(f"OpenAI attempt {attempt + 1}/{config.num_retries} failed: {last_error}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)

        return Response(
            content="", model_used=config.model_name, success=False,
            error_message=f"All {config.num_retries} attempts failed. Last error: {last_error}",
        )

    async def evaluate_async(self, request: Request) -> Response:
        """Asynchronous evaluation."""
        if not self.is_available():
            return Response(
                content="", model_used="", success=False,
                error_message="OpenAI API key not configured. Set OPENAI_API_KEY env var.",
            )

        config = request.config or self.config
        payload = self._build_payload(request)

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = await self._async_client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                model_used = response.model
                usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None

                return Response(content=content.strip(), model_used=model_used, usage=usage, success=True)

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(f"OpenAI async attempt {attempt + 1}/{config.num_retries} failed: {last_error}")
                if attempt < config.num_retries - 1:
                    await asyncio.sleep(config.retry_delay)

        return Response(
            content="", model_used=config.model_name, success=False,
            error_message=f"All {config.num_retries} attempts failed. Last error: {last_error}",
        )
```

### 2.7 File: `lm_eval/llm_judge/providers/openrouter.py`

**Purpose:** OpenRouter API provider with sync and async support for accessing multiple models (GPT-4, Claude, Llama, etc.).

```python
"""OpenRouter API provider implementation for LLM judge."""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp
import requests

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


eval_logger = logging.getLogger(__name__)


class OpenRouterProvider(ServerInterface):
    """OpenRouter API implementation with sync and async support.

    Environment Variables:
        OPENROUTER_API_KEY: API key for authentication
        OPENROUTER_SITE_URL: Optional site URL for rankings
        OPENROUTER_APP_NAME: Optional app name for rankings

    Model Name Format: "provider/model-name", e.g.:
        - "anthropic/claude-3-opus"
        - "anthropic/claude-3-haiku"
        - "meta-llama/llama-3-70b-instruct"
        - "openai/gpt-4o"
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(model_name="anthropic/claude-3-haiku")
        super().__init__(config)

        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "lm-evaluation-harness")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

    def _build_payload(self, request: Request) -> Dict[str, Any]:
        """Build the API payload from request."""
        config = request.config or self.config
        messages = self.prepare_messages(request)

        payload = {
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
                content="", model_used="", success=False,
                error_message="OpenRouter API key not configured. Set OPENROUTER_API_KEY env var.",
            )

        config = request.config or self.config
        payload = self._build_payload(request)
        url = f"{self.API_BASE}/chat/completions"

        last_error = None
        for attempt in range(config.num_retries):
            try:
                response = requests.post(
                    url, headers=self._get_headers(), json=payload, timeout=config.timeout
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"]
                model_used = result.get("model", config.model_name)
                usage = result.get("usage")

                return Response(content=content.strip(), model_used=model_used, usage=usage, success=True)

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(f"OpenRouter attempt {attempt + 1}/{config.num_retries} failed: {last_error}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)

        return Response(
            content="", model_used=config.model_name, success=False,
            error_message=f"All {config.num_retries} attempts failed. Last error: {last_error}",
        )

    async def evaluate_async(self, request: Request) -> Response:
        """Asynchronous evaluation."""
        if not self.is_available():
            return Response(
                content="", model_used="", success=False,
                error_message="OpenRouter API key not configured. Set OPENROUTER_API_KEY env var.",
            )

        config = request.config or self.config
        payload = self._build_payload(request)
        url = f"{self.API_BASE}/chat/completions"

        last_error = None
        for attempt in range(config.num_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=self._get_headers(), json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                content = result["choices"][0]["message"]["content"]
                model_used = result.get("model", config.model_name)
                usage = result.get("usage")

                return Response(content=content.strip(), model_used=model_used, usage=usage, success=True)

            except Exception as e:
                last_error = str(e)
                eval_logger.warning(f"OpenRouter async attempt {attempt + 1}/{config.num_retries} failed: {last_error}")
                if attempt < config.num_retries - 1:
                    await asyncio.sleep(config.retry_delay)

        return Response(
            content="", model_used=config.model_name, success=False,
            error_message=f"All {config.num_retries} attempts failed. Last error: {last_error}",
        )
```

### 2.8 File: `lm_eval/llm_judge/__init__.py`

```python
"""LLM-as-a-Judge infrastructure for lm-evaluation-harness."""
from .protocol import Request, Response, ServerConfig
from .base import ServerInterface
from .factory import ProviderFactory

__all__ = [
    "Request",
    "Response",
    "ServerConfig",
    "ServerInterface",
    "ProviderFactory",
]
```

---

## Phase 3: DeepEval Metric Wrappers

### 3.1 Understanding the Integration Point

From `lm_eval/api/task.py` (lines 1725-1731), the key invocation pattern is:

```python
# For generate_until output type:
result_score = self._metric_fn_list[metric](
    references=[gold],      # gold = doc_to_target(doc)
    predictions=[result],   # result = filtered response from model
    **self._metric_fn_kwargs[metric],  # kwargs from YAML metric_list
)
```

### 3.2 File: `lm_eval/api/metrics_llm_judge.py`

**Purpose:** DeepEval-backed metrics registered for use in task YAMLs. DeepEval is a required dependency.

```python
"""LLM-as-Judge metrics using DeepEval for lm-evaluation-harness.

DeepEval is a required dependency. Install with: pip install deepeval
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from lm_eval.api.registry import register_metric


eval_logger = logging.getLogger(__name__)


def _build_evaluation_params(
    input_text: Optional[str],
    expected_output: Optional[str],
    context: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
) -> List[LLMTestCaseParams]:
    """Build evaluation params based on available data."""
    params = [LLMTestCaseParams.ACTUAL_OUTPUT]
    if expected_output:
        params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
    if input_text:
        params.append(LLMTestCaseParams.INPUT)
    if context:
        params.append(LLMTestCaseParams.CONTEXT)
    if retrieval_context:
        params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
    return params


def _build_test_case(
    actual_output: str,
    input_text: Optional[str] = None,
    expected_output: Optional[str] = None,
    context: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
) -> LLMTestCase:
    """Build an LLMTestCase from available data."""
    return LLMTestCase(
        input=input_text or "",
        actual_output=actual_output,
        expected_output=expected_output,
        context=[context] if context else None,
        retrieval_context=retrieval_context,
    )


@register_metric(
    metric="g_eval",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def g_eval_fn(
    references: List[str],
    predictions: List[str],
    criteria: str = "Determine if the actual output is correct based on the expected output.",
    evaluation_steps: Optional[List[str]] = None,
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: Optional[str] = None,
    context: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
    **kwargs,
) -> float:
    """G-Eval metric using DeepEval's GEval implementation.

    This metric uses an LLM judge to evaluate response quality based on
    customizable criteria and evaluation steps.

    YAML Configuration Example:
        metric_list:
          - metric: g_eval
            aggregation: mean
            higher_is_better: true
            criteria: "Evaluate the mathematical reasoning and correctness."
            evaluation_steps:
              - "Check if reasoning steps are logically sound"
              - "Verify calculations are correct"
              - "Confirm final answer matches expected value"
            judge_model: gpt-4o
            threshold: 0.7
            strict_mode: false
            async_mode: true

    Args:
        references: List containing expected output (gold standard)
        predictions: List containing model's actual output
        criteria: Natural language description of evaluation criteria
        evaluation_steps: Optional list of specific evaluation steps
        judge_model: Model to use as judge (default: gpt-4o)
        threshold: Score threshold for success (default: 0.5)
        strict_mode: If True, scores below threshold become 0 (default: False)
        async_mode: Use async evaluation (default: True)
        input_text: Original input/question (optional)
        context: Additional context for evaluation (optional)
        retrieval_context: Retrieved documents for RAG evaluation (optional)

    Returns:
        float: Score between 0.0 and 1.0
    """
    expected = references[0] if references else None
    actual = predictions[0] if predictions else ""

    # Build evaluation parameters based on available data
    eval_params = _build_evaluation_params(input_text, expected, context, retrieval_context)

    # Create GEval metric
    metric = GEval(
        name="g_eval",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=eval_params,
        model=judge_model,
        threshold=threshold,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )

    # Build test case
    test_case = _build_test_case(
        actual_output=actual,
        input_text=input_text,
        expected_output=expected,
        context=context,
        retrieval_context=retrieval_context,
    )

    # Measure (use async if configured)
    if async_mode:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(metric.a_measure(test_case))
        except RuntimeError:
            # Fallback to sync if async fails
            metric.measure(test_case)
    else:
        metric.measure(test_case)

    score = metric.score if metric.score is not None else 0.0

    # Log reason for debugging
    if metric.reason:
        eval_logger.debug(f"G-Eval reason: {metric.reason}")

    return score


@register_metric(
    metric="answer_relevancy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def answer_relevancy_fn(
    references: List[str],
    predictions: List[str],
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: Optional[str] = None,
    **kwargs,
) -> float:
    """Answer Relevancy metric using DeepEval.

    Evaluates whether the LLM's output is relevant to the input query.
    This is a reference-free metric - it only needs input and actual_output.

    YAML Configuration Example:
        metric_list:
          - metric: answer_relevancy
            aggregation: mean
            higher_is_better: true
            judge_model: gpt-4o
            threshold: 0.7
    """
    actual = predictions[0] if predictions else ""

    metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=threshold,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )

    test_case = LLMTestCase(
        input=input_text or "",
        actual_output=actual,
    )

    if async_mode:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(metric.a_measure(test_case))
        except RuntimeError:
            metric.measure(test_case)
    else:
        metric.measure(test_case)

    return metric.score if metric.score is not None else 0.0


@register_metric(
    metric="faithfulness",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def faithfulness_fn(
    references: List[str],
    predictions: List[str],
    judge_model: str = "gpt-4o",
    threshold: float = 0.5,
    strict_mode: bool = False,
    async_mode: bool = True,
    input_text: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
    **kwargs,
) -> float:
    """Faithfulness metric using DeepEval.

    Evaluates whether the LLM's output is factually consistent with the
    retrieval context. Useful for RAG system evaluation.

    YAML Configuration Example:
        metric_list:
          - metric: faithfulness
            aggregation: mean
            higher_is_better: true
            judge_model: gpt-4o
            threshold: 0.7
    """
    actual = predictions[0] if predictions else ""

    metric = FaithfulnessMetric(
        model=judge_model,
        threshold=threshold,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )

    test_case = LLMTestCase(
        input=input_text or "",
        actual_output=actual,
        retrieval_context=retrieval_context or [],
    )

    if async_mode:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(metric.a_measure(test_case))
        except RuntimeError:
            metric.measure(test_case)
    else:
        metric.measure(test_case)

    return metric.score if metric.score is not None else 0.0
```

### 3.3 Ensure Metrics Are Loaded

Add to the end of `lm_eval/api/metrics.py`:

```python
# Import LLM judge metrics to register them
try:
    import lm_eval.api.metrics_llm_judge
except ImportError:
    pass  # LLM judge metrics not available
```

---

## Phase 4: YAML Configuration and Task Integration

### 4.1 Kwargs Flow Diagram

```
YAML metric_list entry:
    metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: "..."           # Passes through
    evaluation_steps: [...]   # Passes through
    judge_provider: openai    # Passes through
    judge_model: gpt-4o       # Passes through

    |
    v

ConfigurableTask.__init__():
    kwargs = {k: v for k, v in metric_config.items()
              if k not in ["metric", "aggregation", "higher_is_better", "hf_evaluate"]}
    self._metric_fn_kwargs["g_eval"] = kwargs

    |
    v

ConfigurableTask.process_results():
    result_score = self._metric_fn_list["g_eval"](
        references=[gold],
        predictions=[result],
        **self._metric_fn_kwargs["g_eval"],
    )

    |
    v

g_eval_fn(references=[gold], predictions=[result], criteria="...", ...)
```

### 4.2 Example Task YAML: GSM8K with G-Eval

File: `lm_eval/tasks/gsm8k/gsm8k_g_eval.yaml`

```yaml
task: gsm8k_g_eval
tag:
  - math_word_problems
  - llm_judge

dataset_path: gsm8k
dataset_name: main
output_type: generate_until

training_split: train
fewshot_split: train
test_split: test

doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"

metric_list:
  # Traditional exact match (baseline)
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"

  # G-Eval for reasoning quality
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: |
      Evaluate the mathematical reasoning and final answer correctness.
      A good response should:
      1. Show clear step-by-step reasoning
      2. Use correct mathematical operations
      3. Arrive at the correct numerical answer
    evaluation_steps:
      - "Check if the reasoning steps are logically sound"
      - "Verify all calculations are correct"
      - "Confirm the final answer matches the expected value"
    judge_provider: openai
    judge_model: gpt-4o
    threshold: 0.7
    strict_mode: false

generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0

repeats: 1
num_fewshot: 5

filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"

metadata:
  version: 1.0
```

---

## Phase 5: Practical Considerations

### 5.1 API Key Management

**Environment Variables:**
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_APP_NAME="my-eval-app"  # Optional

# Default provider selection
export JUDGE_API_TYPE="openai"  # or "openrouter"
```

### 5.2 Cost Estimation

```
GPT-4o pricing (approximate):
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

Per evaluation (~500 input + 100 output tokens):
- Cost: ~$0.002 per sample

1000 samples: ~$2.00
10000 samples: ~$20.00
```

**Mitigation Strategies:**
1. Use `--limit N` for testing
2. Use cheaper models (gpt-4o-mini, claude-3-haiku)
3. Use OpenRouter for model flexibility

### 5.3 Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
llm_judge = [
    "deepeval>=0.21.0",      # Required - provides GEval and other metrics
    "openai>=1.0.0",         # Required for OpenAI provider
    "aiohttp>=3.8.0",        # Required for async HTTP requests (OpenRouter)
    "requests>=2.28.0",      # Required for sync HTTP requests
]
```

**Note:** The `llm_judge` extra includes all dependencies needed for LLM-as-Judge metrics. DeepEval is a required dependency, not optional.

---

## Implementation Sequence (Git Commits)

### Commit 1: Core LLM Judge Infrastructure
```
feat: add core llm_judge protocol and interfaces

- Add lm_eval/llm_judge/protocol.py
- Add lm_eval/llm_judge/base.py
- Add lm_eval/llm_judge/utils.py
- Add lm_eval/llm_judge/__init__.py
```

### Commit 2: Provider Factory and Implementations
```
feat: add llm_judge provider factory and providers

- Add lm_eval/llm_judge/factory.py
- Add lm_eval/llm_judge/providers/openai.py
- Add lm_eval/llm_judge/providers/openrouter.py
- Add lm_eval/llm_judge/providers/__init__.py
```

### Commit 3: DeepEval Metric Wrappers
```
feat: add g_eval and llm_judge metrics

- Add lm_eval/api/metrics_llm_judge.py
- Update lm_eval/api/metrics.py to import llm_judge metrics
- Add optional dependencies for llm_judge and deepeval
```

### Commit 4: Task Configuration Examples
```
feat: add example tasks with llm_judge metrics

- Add lm_eval/tasks/gsm8k/gsm8k_g_eval.yaml
```

### Commit 5: Tests and Documentation
```
test: add llm_judge unit and integration tests
docs: add llm_judge usage documentation
```

---

## Critical Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `lm_eval/llm_judge/__init__.py` | Create | Package exports |
| `lm_eval/llm_judge/protocol.py` | Create | Data structures |
| `lm_eval/llm_judge/base.py` | Create | Abstract interface |
| `lm_eval/llm_judge/utils.py` | Create | Prompt/parsing helpers |
| `lm_eval/llm_judge/factory.py` | Create | Provider factory |
| `lm_eval/llm_judge/providers/__init__.py` | Create | Provider exports |
| `lm_eval/llm_judge/providers/openai.py` | Create | OpenAI implementation |
| `lm_eval/llm_judge/providers/openrouter.py` | Create | OpenRouter implementation |
| `lm_eval/api/metrics_llm_judge.py` | Create | LLM judge metrics |
| `lm_eval/api/metrics.py` | Modify | Import new metrics |
| `pyproject.toml` | Modify | Add optional dependencies |
| `lm_eval/tasks/gsm8k/gsm8k_g_eval.yaml` | Create | Example task config |

---

## Testing Strategy

1. **Unit Tests**: Test each component in isolation (protocol, utils, providers)
2. **Integration Tests**: Test metric functions with mock providers
3. **End-to-End Tests**: Test full evaluation pipeline with real API calls (limited samples)
4. **Cost Control**: Use `--limit 5` for development testing

---

## Deliverable

This plan supersedes the outline in `docs/llm_judge_deepeval_integration_plan.md` and provides complete implementation details including:
- Full code snippets for all new files
- Async support from the start
- DeepEval as a required dependency
- OpenAI + OpenRouter providers
- Example YAML configurations
