# LLM-as-a-Judge & DeepEval Integration – Implementation Checklist

This checklist operationalizes `docs/llm_judge_deepeval_integration_plan_detailed.md` into concrete implementation steps.

---

## 0. Phase 1: Baseline & Scope (Documentation)

Phase 1 from the detailed plan is the baseline and scoping phase. This work is complete:

- [x] Document motivation and high-level goals (see `llm_judge_deepeval_integration_plan_detailed.md` §1.1)
- [x] Analyze current metrics pipeline (see `metrics_workflow_analysis.md`)
- [x] Define target architecture (see `llm_judge_deepeval_integration_plan_detailed.md` §1.3)
- [x] Document non-goals and out-of-scope items (see `llm_judge_deepeval_integration_plan_detailed.md` §1.4)

---

## 1. Create lm_eval.llm_judge package scaffold (Phase 2)

- [ ] Create directory structure:
  - `lm_eval/llm_judge/__init__.py`
  - `lm_eval/llm_judge/protocol.py`
  - `lm_eval/llm_judge/base.py`
  - `lm_eval/llm_judge/utils.py`
  - `lm_eval/llm_judge/factory.py`
  - `lm_eval/llm_judge/providers/__init__.py`
  - `lm_eval/llm_judge/providers/openai.py`
  - `lm_eval/llm_judge/providers/openrouter.py`
- [ ] In `__init__.py`, re-export core types: `ServerConfig`, `Request`, `Response`, `ServerInterface`, `ProviderFactory`.

## 2. Implement protocol dataclasses

File: `lm_eval/llm_judge/protocol.py`

- [ ] Add constants: `DEFAULT_NUM_RETRIES`, `DEFAULT_RETRY_DELAY`, `DEFAULT_TIMEOUT`.
- [ ] Implement `ServerConfig` with fields and defaults from the plan, using built-in generics and `| None`.
- [ ] Implement `Request` with `messages: list[dict[str, Any]]` and optional metadata fields (`question`, `answer`, `prediction`, `context`, `prompt_kwargs`).
- [ ] Implement `Response` with `content`, `model_used`, optional `usage`, `raw_response`, `parsed_result`, `success`, `error_message`.
- [ ] Add docstrings explaining each field and how these types are used by providers/metrics.

## 3. Implement ServerInterface base class

File: `lm_eval/llm_judge/base.py`

- [ ] Implement constructor taking `ServerConfig | None`, defaulting to `ServerConfig(model_name="gpt-4o")`.
- [ ] Implement lazy `semaphore` property using `config.max_concurrent`.
- [ ] Declare abstract methods: `evaluate`, `evaluate_async`, `is_available`.
- [ ] Implement `prepare_messages` to insert a `system` message when `config.system_prompt` is set.
- [ ] Implement `evaluate_batch_async` using the semaphore to bound concurrency.
- [ ] Implement `evaluate_score` and `evaluate_score_async` using `JudgePromptBuilder` and `ResponseParser`.

## 4. Implement utils: JudgePromptBuilder & ResponseParser

File: `lm_eval/llm_judge/utils.py`

- [ ] Define `DEFAULT_SCORE_PROMPT` exactly as in the plan.
- [ ] Implement `JudgePromptBuilder.build_score_prompt` with optional template override.
- [ ] Implement `JudgePromptBuilder.build_geval_prompt` with criteria, optional evaluation steps, input, outputs, context, retrieval context.
- [ ] Implement `ResponseParser.parse_score_response` (regex numeric extraction and clamping).
- [ ] Implement `ResponseParser.parse_json_response` (direct parse, then embedded JSON fallback).

## 5. Implement ProviderFactory

File: `lm_eval/llm_judge/factory.py`

- [ ] Maintain `_provider_classes: dict[str, type[ServerInterface]]`.
- [ ] Implement `_lazy_load_providers` to import/register `OpenAIProvider` and `OpenRouterProvider`, logging missing deps instead of failing.
- [ ] Implement `create_provider` with precedence: explicit arg → `JUDGE_API_TYPE` env var → default `"openai"`.
- [ ] On unknown provider type, raise `ValueError` listing available providers.
- [ ] Implement `register_provider` for custom providers and `available_providers` to list names.

## 6. Implement OpenAI provider

File: `lm_eval/llm_judge/providers/openai.py`

- [ ] Read `OPENAI_API_KEY` and `OPENAI_API_BASE` env vars.
- [ ] Lazily construct `OpenAI` and `AsyncOpenAI` clients; log a warning if the package is missing.
- [ ] Implement `is_available` based on API key and client presence.
- [ ] Implement `_build_payload` using `Request` + `ServerConfig`, supporting `top_p` and optional JSON `response_format`.
- [ ] Implement `evaluate` with retry loop over `config.num_retries`, sleeping `retry_delay` between attempts, returning `Response` with usage and errors.
- [ ] Implement `evaluate_async` analogously using the async client and `asyncio.sleep`.

## 7. Implement OpenRouter provider

File: `lm_eval/llm_judge/providers/openrouter.py`

- [ ] Read `OPENROUTER_API_KEY`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME` env vars.
- [ ] Implement `_get_headers` including authorization, referer, and app name.
- [ ] Implement `_build_payload` mirroring OpenAI provider semantics.
- [ ] Implement sync `evaluate` using `requests.post` with retries and clear error reporting.
- [ ] Implement async `evaluate_async` using `aiohttp.ClientSession` with retries and timeout.

## 8. Implement DeepEval-backed metrics

File: `lm_eval/api/metrics_llm_judge.py`

- [ ] Implement `_build_evaluation_params` and `_build_test_case` as described in the plan.
- [ ] Implement `g_eval_fn` wrapper around `GEval`, enforcing `criteria` vs `evaluation_steps` precedence.
- [ ] Implement `answer_relevancy_fn` wrapper around `AnswerRelevancyMetric` (reference-free, uses `input_text` and `predictions`).
- [ ] Implement `faithfulness_fn` wrapper around `FaithfulnessMetric` (uses `input_text`, `predictions`, `retrieval_context`).
- [ ] Register all three metrics with `@register_metric` and correct `output_type`, `aggregation`, `higher_is_better`.

## 9. Wire metrics into lm-eval

- [ ] Add a guarded import of `lm_eval.api.metrics_llm_judge` in the central metrics registry so missing `deepeval` does not break imports.
- [ ] Verify that metrics `"g_eval"`, `"answer_relevancy"`, and `"faithfulness"` resolve via the registry.

## 10. Example YAML usage

- [ ] Add or update a small `lm_eval/tasks/**` YAML demonstrating the new metrics for a `generate_until` task.
- [ ] Show usage of extra kwargs like `criteria`, `evaluation_steps`, `judge_model`, `threshold`, `async_mode`, `input_text`, `context`, `retrieval_context`.

## 11. Testing tasks

- [ ] Unit tests for `protocol.py` and `utils.py` (defaults, prompt building, parsing).
- [ ] Unit tests for `ServerInterface` (semaphore, `prepare_messages`, `evaluate_batch_async`).
- [ ] Unit tests for `ProviderFactory` (lazy loading, defaults, error cases, custom registration).
- [ ] Unit tests for providers (payloads, retries, error handling) using mocks.
- [ ] Unit tests for DeepEval metrics (argument mapping, criteria vs evaluation_steps, async vs sync) using mocks.
- [ ] Integration tests for metrics in a `ConfigurableTask`.
- [ ] Optional CLI smoke test with a tiny task and low `--limit`.

