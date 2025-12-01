# LLM-as-a-Judge + DeepEval Integration Plan

## 1. Outline
- 2. Motivation
- 3. Current project structure
- 4. Future project structure
- 5. Implementation phases (with git commits)

## 2. Motivation
- Extend `eval-harness-with-deepeval` (a fork of `lm-evaluation-harness`) with LLM-as-a-judge evaluation.
- Reuse **lmms-eval**'s LLM-judge infrastructure and **deepeval**'s metric logic (e.g., G-Eval) while preserving the existing metrics pipeline.
- Expose judge-backed metrics as first-class, configurable metrics in the harness (via YAML `metric_list`).

## 3. Current project structure

### 3.1 Repo layout (high level)
- `lm_eval/` – forked lm-evaluation-harness core (CLI, evaluator, tasks, metrics, registry, filters).
- `docs/metrics_workflow_analysis.md` – detailed analysis of current metrics pipeline.
- `lmms-eval/lmms_eval/llm_judge/` – existing LLM-as-a-judge infra (protocol, base, providers, prompts, utils).
- `deepeval/` – deepeval library, including `metrics/g_eval/g_eval.py` and related docs/examples.

### 3.2 Metrics workflow (essential points from metrics_workflow_analysis)
- CLI (`lm_eval/__main__.py`) → `evaluator.simple_evaluate` → `evaluate`.
- Tasks and metrics are configured in YAML under `lm_eval/tasks/**`, via `metric_list` and optional kwargs.
- `ConfigurableTask` (in `lm_eval/api/task.py`) resolves metrics via `get_metric` / `register_metric` and stores:
  - `_metric_fn_list[metric_name]` – metric function (often passthrough per-sample).
  - `_metric_fn_kwargs[metric_name]` – kwargs populated from YAML (e.g., `ignore_case`).
- Request & response flow:
  - `build_all_requests()` → `Instance` objects (per doc, sometimes per choice).
  - `evaluator.evaluate()` groups instances by `request_type` and calls LM methods (batched inside the LM).
  - Filters (`lm_eval/api/filter.py`) operate on raw responses and store `instance.filtered_resps[filter_name]`.
- Per-sample metric computation (Section 7 of analysis):
  - For each doc and filter, `task.process_results(doc, filtered_resps_for_doc)` is called.
  - For `generate_until` tasks, `process_results` loops over `_metric_fn_list` and calls each metric as:
    - `metric_fn(references=[gold], predictions=[result], **_metric_fn_kwargs[metric])`.
  - **Important**: metrics run per document, not batched.
- Aggregation and stderr:
  - `TaskOutput.calculate_aggregate_metric()` aggregates per-sample values using `task.aggregation()[metric]` (e.g., `mean`, `bleu`).
  - Standard errors are computed via `stderr_for_metric` with either bootstrap or closed-form.

### 3.3 Current LLM-as-a-judge / deepeval usage
- No metric in `lm_eval/api/metrics.py` currently calls external LLMs as judges.
- `lmms-eval/lmms_eval/llm_judge/*` and `deepeval/metrics/*` exist but are not integrated into the harness metric pipeline.
- There are no task YAMLs using judge-backed metrics; deepeval is only used inside its own examples/tests.

## 4. Future project structure

### 4.1 High-level architecture
- Keep the **core pipeline unchanged** (CLI → evaluator → tasks → process_results → aggregation).
- Introduce an **LLM-judge adapter layer** inside `lm_eval` that:
  - Wraps lmms-eval-style judge providers (OpenAI, future OpenRouter, optionally Azure).
  - Provides a small, stable API that deepeval-backed metrics can call.
- Implement **deepeval-backed metrics** as standard lm-eval metrics:
  - Registered via `@register_metric`.
  - Configured via YAML `metric_list` kwargs.
  - Invoked from `process_results` just like existing metrics.

### 4.2 Proposed Python module layout
- `lm_eval/llm_judge/__init__.py` – public entrypoints and convenience constructors.
- `lm_eval/llm_judge/protocol.py` – minimal `ServerConfig`, `Request`, `Response` dataclasses.
- `lm_eval/llm_judge/base.py` – sync/async judge interfaces (subset of lmms-eval's `ServerInterface`).
- `lm_eval/llm_judge/prompt.py` – judge prompts (rubric, binary, comparative) trimmed to essentials.
- `lm_eval/llm_judge/utils.py` – prompt builders, response parsing, shared helpers.
- `lm_eval/llm_judge/factory.py` – `ProviderFactory` for constructing providers from names/config.
- `lm_eval/llm_judge/providers/__init__.py` – provider registry export.
- `lm_eval/llm_judge/providers/openai.py` – OpenAI judge implementation.
- `lm_eval/llm_judge/providers/openrouter.py` – new OpenRouter provider.
- `lm_eval/metrics/_llm_judge_utils.py` – helpers shared by judge-backed metrics.
- `lm_eval/metrics/deepeval_wrappers.py` – deepeval-based metric wrappers (e.g., `g_eval`).
- `docs/llm_judge_deepeval_integration_plan.md` – this plan (design reference).
- `docs/llm_judge_usage.md` – future user-facing usage/configuration guide.

### 4.3 Configuration surface
- **Per-task YAML** (`lm_eval/tasks/**.yaml`):
  - Use `metric_list` to select judge metrics and configure:
    - Metric type (`g_eval`, `answer_relevancy`, etc.).
    - Judge provider/model (`judge_provider`, `judge_model_name`).
    - Evaluation semantics (`criteria`, `evaluation_params`, `rubric`, `threshold`, `strict_mode`, etc.).
- **Optional global config** (later phase):
  - `configs/llm_judge_defaults.yaml` containing shared judge defaults, which tasks may override.

## 5. Implementation phases

### Phase 1 – Finalise design & document baseline (this document)
**Goal:** Consolidate understanding of metrics workflow and desired integration architecture.
- Validate the metrics pipeline using `docs/metrics_workflow_analysis.md` as ground truth.
- Decide to keep the pipeline unchanged and plug judge-backed metrics into `process_results` via `metric_fn`.
- Identify in-scope vs out-of-scope parts of lmms-eval and deepeval (e.g., likely exclude SGLang provider, non-judge deepeval features).
- Capture architecture and phases in `docs/llm_judge_deepeval_integration_plan.md`.

**Key reference files:**
- `docs/metrics_workflow_analysis.md`
- `lm_eval/api/metrics.py`, `lm_eval/api/registry.py`, `lm_eval/api/task.py`, `lm_eval/evaluator.py`
- `lmms-eval/lmms_eval/llm_judge/*`
- `deepeval/metrics/g_eval/g_eval.py` and related deepeval docs

**New/updated files:**
- `docs/llm_judge_deepeval_integration_plan.md`

**Git commit:**
- `docs: add llm-as-a-judge deepeval integration plan`

### Phase 2 – Introduce LLM-judge adapter package

#### Phase 2.1 – Create `lm_eval/llm_judge` core
**Goal:** Establish minimal protocol and interfaces without changing behaviour elsewhere.
- Create `lm_eval/llm_judge/__init__.py`, `protocol.py`, `base.py`, `prompt.py`, `utils.py`.
- Port/adapt only the necessary parts of `ServerConfig`, `Request`, `Response`, and judge interfaces from lmms-eval.
- Remove or avoid unused fields (such as unused `evaluation_criteria`) while keeping compatibility with planned metrics.

**Reference files:** `lmms-eval/lmms_eval/llm_judge/protocol.py`, `base.py`, `prompt.py`, `utils.py`.

**New/updated files:**
- `lm_eval/llm_judge/__init__.py`
- `lm_eval/llm_judge/protocol.py`
- `lm_eval/llm_judge/base.py`
- `lm_eval/llm_judge/prompt.py`
- `lm_eval/llm_judge/utils.py`

**Git commit:**
- `feat: add core llm_judge protocol and interfaces`

#### Phase 2.2 – Add provider factory and providers
**Goal:** Enable pluggable judge backends, including OpenAI and OpenRouter.
- Implement `ProviderFactory` in `lm_eval/llm_judge/factory.py` with a simple name→provider mapping.
- Port OpenAI provider into `lm_eval/llm_judge/providers/openai.py`, trimming unused features.
- Design and implement an initial `openrouter.py` provider (model naming, base URL, auth, error handling).
- Decide whether to include Azure or other providers; either port them or explicitly mark them as out-of-scope.

**Reference files:** `lmms-eval/lmms_eval/llm_judge/factory.py`, `providers/openai.py`, `providers/azure_openai.py`.

**New/updated files:**
- `lm_eval/llm_judge/factory.py`
- `lm_eval/llm_judge/providers/__init__.py`
- `lm_eval/llm_judge/providers/openai.py`
- `lm_eval/llm_judge/providers/openrouter.py`

**Git commit:**
- `feat: add llm_judge provider factory and openai/openrouter providers`

### Phase 3 – Design deepeval metric wrappers as harness metrics

#### Phase 3.1 – Define internal wrapper helper
**Goal:** Centralise common logic for LLM-judge-backed metrics.
- Add `lm_eval/metrics/_llm_judge_utils.py` with helpers to:
  - Build deepeval `LLMTestCase` objects from `(doc, prediction, optional context)`.
  - Instantiate configured deepeval metric instances (e.g., `GEval`) using kwargs.
  - Invoke the judge via the adapter and return a scalar score in [0,1].
- Decide exactly which kwargs (criteria, evaluation_params, rubric, threshold, judge provider/model, etc.) are supported and how missing values default.

**Reference files:** `lm_eval/api/metrics.py`, `lm_eval/api/task.py` (`process_results` for generate_until), `deepeval/metrics/g_eval/g_eval.py`.

**New/updated files:**
- `lm_eval/metrics/_llm_judge_utils.py`

**Git commit:**
- `feat: add internal helpers for llm-judge-backed metrics`

#### Phase 3.2 – Implement concrete deepeval-backed metrics
**Goal:** Expose at least one powerful, configurable LLM-judge metric.
- Create `lm_eval/metrics/deepeval_wrappers.py` and register metrics via `@register_metric` (e.g., `metric="g_eval"`, `output_type="generate_until"`, `aggregation="mean"`).
- Implement `g_eval` wrapper that accepts kwargs like `criteria`, `evaluation_params`, `rubric`, `threshold`, `strict_mode`, `async_mode`, `judge_provider`, `judge_model_name` and forwards them into deepeval.
- Optionally design a generic `deepeval_metric` wrapper for other deepeval metrics (e.g., answer relevancy) using the same helper.

**Reference files:** `deepeval/metrics/g_eval/g_eval.py`, deepeval docs/blog examples, `lm_eval/api/registry.py`.

**New/updated files:**
- `lm_eval/metrics/deepeval_wrappers.py`

**Git commit:**
- `feat: add g_eval deepeval metric wrapper`

### Phase 4 – Wire metrics into tasks and configuration

#### Phase 4.1 – YAML configuration patterns
**Goal:** Enable users to configure judge-backed metrics per task via YAML.
- Define a canonical YAML schema for G-Eval-style metrics, for example:
  - `metric: g_eval`, `aggregation: mean`, `higher_is_better: true`.
  - Additional keys: `criteria`, `evaluation_params`, `threshold`, `judge_provider`, `judge_model_name`, etc.
- Verify that `ConfigurableTask` passes all unknown keys in `metric_list` through to `_metric_fn_kwargs[metric]` so wrappers receive them unchanged.
- Add at least one example task YAML (e.g., for GSM8K or summarisation) that uses `g_eval` alone or alongside existing metrics.

**Reference files:** `lm_eval/tasks/gsm8k/gsm8k.yaml`, `lm_eval/api/task.py` (metric kwarg handling), `lmms-eval/docs/task_guide.md` (passing args to metrics).

**New/updated files:**
- Example YAML such as `lm_eval/tasks/gsm8k/gsm8k_g_eval.yaml`.

**Git commit:**
- `feat: add task configs demonstrating g_eval llm-judge metric`

#### Phase 4.2 – Optional global defaults
**Goal:** Avoid duplicating judge configuration across tasks.
- Introduce an optional `configs/llm_judge_defaults.yaml` describing default provider/model/temperature/etc.
- Decide how tasks reference these defaults (e.g., via a small loader or documented convention) without complicating the existing CLI.

**New/updated files:**
- `configs/llm_judge_defaults.yaml` (optional)

**Git commit:**
- `feat: add optional global llm-judge default configuration`

### Phase 5 – Cleanup, simplification, and provider enhancements

#### Phase 5.1 – Prune unused LLM-judge features
**Goal:** Reduce maintenance surface by removing dead or unused code.
- Audit `lm_eval/llm_judge/*` for unused config fields, prompts, and providers.
- Remove or clearly mark as experimental any components not used by deepeval metrics or planned tasks (e.g., niche providers).
- Ensure public APIs in `lm_eval.llm_judge` are minimal and well-documented.

**Git commit:**
- `chore: prune unused llm-judge features and simplify interface`

#### Phase 5.2 – Finalise OpenRouter provider
**Goal:** Make OpenRouter a first-class judge provider.
- Complete `openrouter.py` with robust handling of auth, model names, errors, and timeouts.
- Ensure metrics can select it via kwargs (e.g., `judge_provider: "openrouter"`).
- Document environment variables and configuration for OpenRouter in usage docs.

**New/updated files:**
- `lm_eval/llm_judge/providers/openrouter.py`
- `docs/llm_judge_usage.md` (provider configuration section)

**Git commit:**
- `feat: complete openrouter llm-judge provider`

### Phase 6 – Documentation and maintenance hooks
**Goal:** Make the system approachable for future contributors and agents.
- Write `docs/llm_judge_usage.md` explaining:
  - How the metrics pipeline works (linking to `metrics_workflow_analysis.md`).
  - How to add a new LLM-judge-backed metric wrapper.
  - How to add a new provider.
  - How to configure tasks with judge-backed metrics.
- Add brief code comments in key integration points pointing back to these docs.

**Git commit:**
- `docs: document llm-judge metrics and configuration`
