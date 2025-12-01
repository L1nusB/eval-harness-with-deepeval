# LLM-as-a-Judge Metrics Guide

This guide explains how to use LLM-as-a-Judge metrics in lm-evaluation-harness, powered by [DeepEval](https://github.com/confident-ai/deepeval).

## Overview

LLM-as-a-Judge evaluation uses a powerful language model (like GPT-4) to assess the quality of model outputs. This approach is particularly useful when:

- Traditional metrics (exact match, BLEU, etc.) don't capture semantic correctness
- You need to evaluate open-ended generation quality
- You want to assess reasoning, coherence, or factual accuracy
- Reference answers may have multiple valid formulations

This integration provides three DeepEval-backed metrics that work seamlessly with the existing lm-eval infrastructure.

## Quick Start

### 1. Install Dependencies

```bash
pip install deepeval
```

### 2. Set Up API Credentials

```bash
# For OpenAI (default)
export OPENAI_API_KEY="sk-..."

# For OpenRouter (alternative)
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Configure Your Task

Add LLM judge metrics to your task's `metric_list`:

```yaml
metric_list:
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: "Evaluate the correctness and quality of the response."
    judge_model: gpt-4o
```

### 4. Run Evaluation

```bash
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks your_task_name \
        --limit 10  # Start small to verify setup
```

## Available Metrics

### G-Eval (`g_eval`)

A flexible LLM-based evaluation metric that uses customizable criteria and evaluation steps to assess response quality. Based on the [G-Eval paper](https://arxiv.org/abs/2303.16634).

**When to use:**
- Evaluating open-ended generation quality
- Assessing reasoning and problem-solving
- Custom evaluation criteria specific to your task

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criteria` | string | See below | Natural language description of evaluation criteria |
| `evaluation_steps` | list[str] | None | Specific steps for structured evaluation (takes precedence over `criteria`) |
| `judge_model` | string | `"gpt-4o"` | Model to use as judge |
| `threshold` | float | `0.5` | Score threshold for success |
| `strict_mode` | bool | `False` | If True, scores below threshold become 0 |
| `async_mode` | bool | `True` | Use async evaluation for better throughput |
| `input_text` | string | None | Original input/question |
| `context` | string | None | Additional context for evaluation |
| `retrieval_context` | list[str] | None | Retrieved documents for RAG evaluation |

**Example with criteria:**

```yaml
metric_list:
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: |
      Evaluate the mathematical reasoning and final answer correctness.
      A good response should:
      1. Show clear step-by-step reasoning
      2. Use correct mathematical operations
      3. Arrive at the correct numerical answer
    judge_model: gpt-4o
    threshold: 0.7
```

**Example with evaluation steps:**

```yaml
metric_list:
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    evaluation_steps:
      - "Check if the reasoning steps are logically sound"
      - "Verify all calculations are correct"
      - "Confirm the final answer matches the expected value"
    judge_model: gpt-4o
```

### Answer Relevancy (`answer_relevancy`)

Evaluates whether the model's output is relevant to the input query. This is a **reference-free** metric - it doesn't require expected outputs.

**When to use:**
- Open-ended question answering
- Evaluating if responses stay on-topic
- Assessing response relevance without gold answers

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judge_model` | string | `"gpt-4o"` | Model to use as judge |
| `threshold` | float | `0.5` | Score threshold for success |
| `strict_mode` | bool | `False` | If True, scores below threshold become 0 |
| `async_mode` | bool | `True` | Use async evaluation |
| `input_text` | string | None | Original input/question (required for meaningful evaluation) |

**Example:**

```yaml
metric_list:
  - metric: answer_relevancy
    aggregation: mean
    higher_is_better: true
    judge_model: gpt-4o
    threshold: 0.7
```

### Faithfulness (`faithfulness`)

Evaluates whether the model's output is factually consistent with provided context. Designed for RAG (Retrieval-Augmented Generation) evaluation.

**When to use:**
- RAG system evaluation
- Checking if responses are grounded in provided context
- Detecting hallucinations

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judge_model` | string | `"gpt-4o"` | Model to use as judge |
| `threshold` | float | `0.5` | Score threshold for success |
| `strict_mode` | bool | `False` | If True, scores below threshold become 0 |
| `async_mode` | bool | `True` | Use async evaluation |
| `input_text` | string | None | Original input/question |
| `retrieval_context` | list[str] | None | Retrieved documents (required for meaningful evaluation) |

**Example:**

```yaml
metric_list:
  - metric: faithfulness
    aggregation: mean
    higher_is_better: true
    judge_model: gpt-4o
    threshold: 0.7
    retrieval_context:
      - "Context document 1..."
      - "Context document 2..."
```

## Architecture

### How It Works

The LLM judge metrics integrate with the standard lm-eval pipeline:

```
YAML Task Config
       │
       ▼
ConfigurableTask.__init__()
  - Resolves metric via registry
  - Stores kwargs in _metric_fn_kwargs[metric]
       │
       ▼
ConfigurableTask.process_results()
  - For each document:
    metric_fn(references=[gold], predictions=[result], **kwargs)
       │
       ▼
DeepEval Metric Wrapper (e.g., g_eval_fn)
  - Builds LLMTestCase
  - Creates DeepEval metric instance
  - Calls DeepEval's measure/a_measure
  - Returns score in [0.0, 1.0]
       │
       ▼
Aggregation (mean, etc.)
  - Scores aggregated across documents
```

### Package Structure

```
lm_eval/
├── llm_judge/                    # LLM Judge infrastructure
│   ├── __init__.py               # Public exports
│   ├── protocol.py               # ServerConfig, Request, Response
│   ├── base.py                   # ServerInterface ABC
│   ├── utils.py                  # JudgePromptBuilder, ResponseParser
│   ├── factory.py                # ProviderFactory
│   └── providers/
│       ├── __init__.py
│       ├── openai.py             # OpenAI provider
│       └── openrouter.py         # OpenRouter provider
│
├── api/
│   └── metrics_llm_judge.py      # DeepEval metric wrappers
│
└── tasks/
    ├── gsm8k/
    │   └── gsm8k_g_eval.yaml     # Example task with G-Eval
    └── llm_judge_examples/       # Comprehensive examples
        ├── README.md
        ├── qa_g_eval.yaml
        ├── qa_answer_relevancy.yaml
        └── qa_faithfulness.yaml
```

## Provider Configuration

### OpenAI (Default)

Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Optional configuration:
```bash
export OPENAI_API_BASE="https://api.openai.com/v1"  # Custom endpoint
```

### OpenRouter

OpenRouter provides access to multiple models (GPT-4, Claude, Llama, etc.) through a unified API.

```bash
export OPENROUTER_API_KEY="sk-or-..."
export JUDGE_API_TYPE="openrouter"  # Use OpenRouter as default provider
```

Optional configuration:
```bash
export OPENROUTER_APP_NAME="my-eval-app"  # For usage tracking
export OPENROUTER_SITE_URL="https://my-app.com"  # For rankings
```

Model naming for OpenRouter uses the format `provider/model-name`:
- `openai/gpt-4o`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-haiku`
- `meta-llama/llama-3-70b-instruct`

## Cost Considerations

LLM judge metrics incur API costs for each evaluation. Approximate costs (GPT-4o):

| Scale | Estimated Cost |
|-------|----------------|
| Per evaluation | ~$0.002 |
| 100 samples | ~$0.20 |
| 1,000 samples | ~$2.00 |
| 10,000 samples | ~$20.00 |

**Tips to reduce costs:**

1. **Use `--limit N` during development** to test with small sample sizes
2. **Use cheaper models** for initial experiments:
   - `gpt-4o-mini` (much cheaper than `gpt-4o`)
   - `claude-3-haiku` (via OpenRouter)
3. **Use OpenRouter** for model flexibility and competitive pricing
4. **Cache results** using `--use_cache` for repeated runs

## Combining with Traditional Metrics

LLM judge metrics work alongside traditional metrics in the same task:

```yaml
metric_list:
  # Traditional exact match
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true

  # LLM judge for reasoning quality
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: "Evaluate the reasoning quality..."
    judge_model: gpt-4o
```

This allows you to:
- Compare LLM judge scores against traditional metrics
- Get both surface-level accuracy and semantic quality metrics
- Validate that LLM judge metrics align with expected patterns

## Limitations and Future Work

### Current Limitations

1. **Static context**: The `input_text`, `context`, and `retrieval_context` parameters are currently global per evaluation run, not per-example. They come from YAML kwargs.

2. **No per-example input injection**: There's no `doc_to_input` hook to automatically pass per-example prompts to judge metrics (planned for future versions).

3. **Async handling**: While async mode is supported, the current implementation may have limitations when running inside existing event loops.

### Planned Enhancements

1. **`doc_to_metric_inputs`**: A new task configuration option to provide per-example inputs to judge metrics:
   ```yaml
   doc_to_metric_inputs: |
     {
       "input_text": "{{question}}",
       "retrieval_context": ["{{context}}"]
     }
   ```

2. **Result caching**: Cache judge API responses to avoid re-evaluating identical inputs.

3. **Additional DeepEval metrics**: Support for more DeepEval metrics like `ContextualRelevancy`, `Hallucination`, etc.

## Example Tasks

See the `lm_eval/tasks/llm_judge_examples/` directory for complete working examples:

- **qa_g_eval.yaml**: G-Eval for question answering
- **qa_answer_relevancy.yaml**: Answer Relevancy evaluation
- **qa_faithfulness.yaml**: Faithfulness for RAG tasks
- **gsm8k_g_eval.yaml**: Math reasoning with G-Eval

## Troubleshooting

### Common Issues

**"OpenAI API key not configured"**
- Ensure `OPENAI_API_KEY` is set in your environment
- Verify the key is valid and has available credits

**"deepeval not installed"**
- Run `pip install deepeval`
- The metrics will not be available if deepeval is not installed

**Async evaluation errors**
- Try setting `async_mode: false` in your metric configuration
- This uses synchronous evaluation which is more stable in some environments

**High costs**
- Start with `--limit 10` to verify your setup
- Use `gpt-4o-mini` instead of `gpt-4o` for development

### Debugging

Enable debug logging to see judge reasoning:

```python
import logging
logging.getLogger("lm_eval.api.metrics_llm_judge").setLevel(logging.DEBUG)
```

This will log the judge's reasoning for each evaluation, which is useful for understanding scores.

## References

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [G-Eval Paper](https://arxiv.org/abs/2303.16634)
- [LLM-as-a-Judge Survey](https://arxiv.org/abs/2306.05685)
- [lm-evaluation-harness Task Guide](./task_guide.md)
- [Metrics Workflow Analysis](./metrics_workflow_analysis.md)
