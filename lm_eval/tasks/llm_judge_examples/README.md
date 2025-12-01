# LLM-as-a-Judge Example Tasks

This directory contains example task configurations demonstrating how to use
DeepEval-backed LLM-as-a-Judge metrics in lm-evaluation-harness.

## Prerequisites

1. **Install DeepEval:**
   ```bash
   pip install deepeval
   ```

2. **Set up API credentials:**
   ```bash
   # For OpenAI (default judge provider)
   export OPENAI_API_KEY="sk-..."

   # For OpenRouter (alternative provider)
   export OPENROUTER_API_KEY="sk-or-..."
   export JUDGE_API_TYPE="openrouter"
   ```

## Available Metrics

### G-Eval (`g_eval`)

A flexible LLM-based evaluation metric that uses customizable criteria and
evaluation steps to assess response quality.

**Key Parameters:**
- `criteria`: Natural language description of evaluation criteria
- `evaluation_steps`: List of specific steps for structured evaluation
- `judge_model`: Model to use as judge (default: `gpt-4o`)
- `threshold`: Score threshold for success (default: 0.5)
- `strict_mode`: If True, scores below threshold become 0
- `async_mode`: Use async evaluation for better performance

**Example:**
```yaml
metric_list:
  - metric: g_eval
    aggregation: mean
    higher_is_better: true
    criteria: "Evaluate the quality and correctness of the response."
    judge_model: gpt-4o
```

### Answer Relevancy (`answer_relevancy`)

Evaluates whether the model's output is relevant to the input query.
This is a **reference-free** metric - it doesn't need expected outputs.

**Key Parameters:**
- `judge_model`: Model to use as judge
- `threshold`: Score threshold for success
- `input_text`: The original input/question (required for meaningful evaluation)

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

Evaluates whether the model's output is factually consistent with provided
context. Designed for RAG (Retrieval-Augmented Generation) evaluation.

**Key Parameters:**
- `judge_model`: Model to use as judge
- `threshold`: Score threshold for success
- `retrieval_context`: List of retrieved documents/context

**Example:**
```yaml
metric_list:
  - metric: faithfulness
    aggregation: mean
    higher_is_better: true
    judge_model: gpt-4o
    retrieval_context:
      - "Context document 1..."
      - "Context document 2..."
```

## Example Tasks

### qa_g_eval.yaml

Demonstrates G-Eval for evaluating question-answering tasks with reasoning
quality assessment.

### qa_answer_relevancy.yaml

Shows how to use Answer Relevancy metric for open-ended QA evaluation.

### qa_faithfulness.yaml

Example of Faithfulness metric for RAG-style tasks with retrieval context.

## Running Example Tasks

```bash
# Run G-Eval example
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks qa_g_eval \
        --limit 10

# Run all LLM judge examples
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks qa_g_eval,qa_answer_relevancy \
        --limit 10
```

## Cost Considerations

LLM judge metrics incur API costs. Approximate costs (GPT-4o):
- Per evaluation: ~$0.002
- 1,000 samples: ~$2.00
- 10,000 samples: ~$20.00

**Tips to reduce costs:**
1. Use `--limit N` during development
2. Use cheaper models (`gpt-4o-mini`, `claude-3-haiku`)
3. Use OpenRouter for model flexibility and pricing options

## Notes

- All LLM judge metrics require the `deepeval` package
- Metrics are async by default for better throughput
- Input context is currently passed via YAML kwargs (global for the run)
- Future versions may support per-example context via `doc_to_input`
