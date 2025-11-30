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

- [x] Create directory structure:
  - `lm_eval/llm_judge/__init__.py`
  - `lm_eval/llm_judge/protocol.py`
  - `lm_eval/llm_judge/base.py`
  - `lm_eval/llm_judge/utils.py`
  - `lm_eval/llm_judge/factory.py`
  - `lm_eval/llm_judge/providers/__init__.py`
  - `lm_eval/llm_judge/providers/openai.py`
  - `lm_eval/llm_judge/providers/openrouter.py`
- [x] In `__init__.py`, re-export core types: `ServerConfig`, `Request`, `Response`, `ServerInterface`, `ProviderFactory`.

## 2. Implement protocol dataclasses

File: `lm_eval/llm_judge/protocol.py`

- [x] Add constants: `DEFAULT_NUM_RETRIES`, `DEFAULT_RETRY_DELAY`, `DEFAULT_TIMEOUT`.
- [x] Implement `ServerConfig` with fields and defaults from the plan, using built-in generics and `| None`.
- [x] Implement `Request` with `messages: list[dict[str, Any]]` and optional metadata fields (`question`, `answer`, `prediction`, `context`, `prompt_kwargs`).
- [x] Implement `Response` with `content`, `model_used`, optional `usage`, `raw_response`, `parsed_result`, `success`, `error_message`.
- [x] Add docstrings explaining each field and how these types are used by providers/metrics.

## 3. Implement ServerInterface base class

File: `lm_eval/llm_judge/base.py`

- [x] Implement constructor taking `ServerConfig | None`, defaulting to `ServerConfig(model_name="gpt-4o")`.
- [x] Implement lazy `semaphore` property using `config.max_concurrent`.
- [x] Declare abstract methods: `evaluate`, `evaluate_async`, `is_available`.
- [x] Implement `prepare_messages` to insert a `system` message when `config.system_prompt` is set.
- [x] Implement `evaluate_batch_async` using the semaphore to bound concurrency.
- [x] Implement `evaluate_score` and `evaluate_score_async` using `JudgePromptBuilder` and `ResponseParser`.

## 4. Implement utils: JudgePromptBuilder & ResponseParser

File: `lm_eval/llm_judge/utils.py`

- [x] Define `DEFAULT_SCORE_PROMPT` exactly as in the plan.
- [x] Implement `JudgePromptBuilder.build_score_prompt` with optional template override.
- [x] Implement `JudgePromptBuilder.build_geval_prompt` with criteria, optional evaluation steps, input, outputs, context, retrieval context.
- [x] Implement `ResponseParser.parse_score_response` (regex numeric extraction and clamping).
- [x] Implement `ResponseParser.parse_json_response` (direct parse, then embedded JSON fallback).

## 5. Implement ProviderFactory

File: `lm_eval/llm_judge/factory.py`

- [x] Maintain `_provider_classes: dict[str, type[ServerInterface]]`.
- [x] Implement `_lazy_load_providers` to import/register `OpenAIProvider` and `OpenRouterProvider`, logging missing deps instead of failing.
- [x] Implement `create_provider` with precedence: explicit arg → `JUDGE_API_TYPE` env var → default `"openai"`.
- [x] On unknown provider type, raise `ValueError` listing available providers.
- [x] Implement `register_provider` for custom providers and `available_providers` to list names.

## 6. Implement OpenAI provider

File: `lm_eval/llm_judge/providers/openai.py`

- [x] Read `OPENAI_API_KEY` and `OPENAI_API_BASE` env vars.
- [x] Lazily construct `OpenAI` and `AsyncOpenAI` clients; log a warning if the package is missing.
- [x] Implement `is_available` based on API key and client presence.
- [x] Implement `_build_payload` using `Request` + `ServerConfig`, supporting `top_p` and optional JSON `response_format`.
- [x] Implement `evaluate` with retry loop over `config.num_retries`, sleeping `retry_delay` between attempts, returning `Response` with usage and errors.
- [x] Implement `evaluate_async` analogously using the async client and `asyncio.sleep`.

## 7. Implement OpenRouter provider

File: `lm_eval/llm_judge/providers/openrouter.py`

- [x] Read `OPENROUTER_API_KEY`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME` env vars.
- [x] Implement `_get_headers` including authorization, referer, and app name.
- [x] Implement `_build_payload` mirroring OpenAI provider semantics.
- [x] Implement sync `evaluate` using `requests.post` with retries and clear error reporting.
- [x] Implement async `evaluate_async` using `aiohttp.ClientSession` with retries and timeout.

## 8. Implement DeepEval-backed metrics (Phase 3)

File: `lm_eval/api/metrics_llm_judge.py`

- [x] Implement `_build_evaluation_params` and `_build_test_case` as described in the plan.
- [x] Implement `g_eval_fn` wrapper around `GEval`, enforcing `criteria` vs `evaluation_steps` precedence.
- [x] Implement `answer_relevancy_fn` wrapper around `AnswerRelevancyMetric` (reference-free, uses `input_text` and `predictions`).
- [x] Implement `faithfulness_fn` wrapper around `FaithfulnessMetric` (uses `input_text`, `predictions`, `retrieval_context`).
- [x] Register all three metrics with `@register_metric` and correct `output_type`, `aggregation`, `higher_is_better`.

## 9. Wire metrics into lm-eval

- [x] Add a guarded import of `lm_eval.api.metrics_llm_judge` in the central metrics registry so missing `deepeval` does not break imports.
- [x] Verify that metrics `"g_eval"`, `"answer_relevancy"`, and `"faithfulness"` resolve via the registry.

## 10. Example YAML usage (Phase 4)

- [x] Add or update a small `lm_eval/tasks/**` YAML demonstrating the new metrics for a `generate_until` task.
  - Created `lm_eval/tasks/gsm8k/gsm8k_g_eval.yaml` - GSM8K with G-Eval for math reasoning
  - Created `lm_eval/tasks/llm_judge_examples/` directory with comprehensive examples:
    - `qa_g_eval.yaml` - G-Eval with criteria and evaluation steps
    - `qa_answer_relevancy.yaml` - Answer Relevancy metric
    - `qa_faithfulness.yaml` - Faithfulness metric for RAG
    - `README.md` - Documentation for all LLM judge metrics
- [x] Show usage of extra kwargs like `criteria`, `evaluation_steps`, `judge_model`, `threshold`, `async_mode`, `input_text`, `context`, `retrieval_context`.

## 10.5. Documentation (Phase 5 - Practical Considerations)

- [x] Create comprehensive user-facing documentation `docs/llm_judge_usage.md`:
  - Quick start guide with installation and setup
  - Detailed documentation for each metric (G-Eval, Answer Relevancy, Faithfulness)
  - Parameter tables with types and defaults
  - Architecture overview and package structure
  - Provider configuration (OpenAI, OpenRouter)
  - Cost considerations and optimization tips
  - Troubleshooting guide
- [x] Update `docs/README.md` to link to LLM Judge documentation
- [x] Update main `README.md` to mention LLM-as-a-Judge feature
- [x] Update `docs/task_guide.md` to list LLM Judge metrics in supported metrics section

## 11. Testing tasks

- [ ] Unit tests for `protocol.py` and `utils.py` (defaults, prompt building, parsing).
- [ ] Unit tests for `ServerInterface` (semaphore, `prepare_messages`, `evaluate_batch_async`).
- [ ] Unit tests for `ProviderFactory` (lazy loading, defaults, error cases, custom registration).
- [ ] Unit tests for providers (payloads, retries, error handling) using mocks.
- [ ] Unit tests for DeepEval metrics (argument mapping, criteria vs evaluation_steps, async vs sync) using mocks.
- [ ] Integration tests for metrics in a `ConfigurableTask`.
- [ ] Optional CLI smoke test with a tiny task and low `--limit`.
