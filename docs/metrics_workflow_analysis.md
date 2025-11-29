# Metrics Computation Workflow Analysis

## Table of Contents
1. [Overview and Architecture](#1-overview-and-architecture)
2. [Entry Point and CLI](#2-entry-point-and-cli)
3. [Task Configuration and Metric Initialization](#3-task-configuration-and-metric-initialization)
4. [Request Building Phase](#4-request-building-phase)
5. [Model Execution and Batching](#5-model-execution-and-batching)
6. [Filter Pipeline](#6-filter-pipeline)
7. [Per-Sample Metric Computation](#7-per-sample-metric-computation)
8. [Aggregation Phase](#8-aggregation-phase)
9. [Standard Error Calculation](#9-standard-error-calculation)
10. [Group Metrics](#10-group-metrics)
11. [Concrete Examples](#11-concrete-examples)
12. [Key Takeaways](#12-key-takeaways)
13. [Files Reference Summary](#13-files-reference-summary)

---

## 1. Overview and Architecture

The lm-evaluation-harness uses a **pipeline architecture** for metric computation:

```
CLI Entry → Task Loading → Request Building → Model Execution → Filtering → 
Per-Sample Metrics → Aggregation → Standard Error → Results
```

**Key Components:**
- **Entry Point**: `lm_eval/__main__.py` - CLI interface
- **Orchestrator**: `lm_eval/evaluator.py` - Main evaluation loop
- **Task System**: `lm_eval/api/task.py` - Task definitions and metric configuration
- **Metric Registry**: `lm_eval/api/registry.py` - Metric/aggregation registration
- **Metric Functions**: `lm_eval/api/metrics.py` - Built-in metrics and aggregations
- **Filter System**: `lm_eval/api/filter.py` - Post-processing of model outputs
- **Result Container**: `lm_eval/evaluator_utils.py` - TaskOutput class

---

## 2. Entry Point and CLI

**File**: `lm_eval/__main__.py`

```python
# Line 474: Main evaluation call
results = evaluator.simple_evaluate(
    model=args.model,
    model_args=args.model_args,
    tasks=task_names,
    num_fewshot=args.num_fewshot,
    batch_size=args.batch_size,
    # ... other args
)
```

The CLI parses arguments and calls `evaluator.simple_evaluate()`, which is the main entry point for evaluation.

---

## 3. Task Configuration and Metric Initialization

### 3.1 YAML Configuration

Tasks are defined in YAML files (e.g., `lm_eval/tasks/gsm8k/gsm8k.yaml`):

```yaml
task: gsm8k
output_type: generate_until
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
```

### 3.2 Metric Initialization in ConfigurableTask

**File**: `lm_eval/api/task.py` (lines 787-863)

```python
# Lines 792-802: Use default metrics if none specified
if self.config.metric_list is None:
    _metric_list = DEFAULT_METRIC_REGISTRY[self.config.output_type]
    for metric_name in _metric_list:
        self._metric_fn_list[metric_name] = get_metric(metric_name)
        self._aggregation_list[metric_name] = get_metric_aggregation(metric_name)
        self._higher_is_better[metric_name] = is_higher_better(metric_name)
```

### 3.3 Default Metrics by Output Type

**File**: `lm_eval/api/registry.py` (lines 82-87)

```python
DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": ["perplexity", "acc"],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}
```

---

## 4. Request Building Phase

### 4.1 Building Instances for Each Document

**File**: `lm_eval/api/task.py` (lines 390-501)

The `build_all_requests()` method creates an **Instance** for each document in the dataset:

```python
# Lines 454-483: Loop through each document
for doc_id, doc in tqdm(doc_id_docs, total=num_docs):
    # Create fewshot context (prompt with examples)
    fewshot_ctx = self.fewshot_context(
        doc,
        num_fewshot=0 if self.config.num_fewshot is None else self.config.num_fewshot,
        ...
    )
    
    # Construct request(s) for this document
    inst = self.construct_requests(
        doc=doc,
        ctx=fewshot_ctx,
        metadata=(self.config["task"], doc_id, self.config.repeats),
        ...
    )
    
    instances.append(inst)
```

### 4.2 Instance Structure

**File**: `lm_eval/api/instance.py`

```python
@dataclass
class Instance:
    request_type: OutputType  # "loglikelihood", "generate_until", etc.
    doc: dict                  # The original document
    arguments: tuple           # Request-specific arguments
    idx: int                   # Index within document
    metadata: Tuple            # (task_name, doc_id, repeats)
    resps: list               # Model responses (filled later)
    filtered_resps: dict      # Filtered responses (filled later)
```

**Key Point**: One Instance is created **per document** (not batched at this stage). For multiple-choice tasks, multiple instances may be created per document (one per choice).

---

## 5. Model Execution and Batching

**File**: `lm_eval/evaluator.py` (lines 540-598)

### 5.1 Grouping Requests by Type

```python
# Lines 540-560: Group instances by request type
for task_output in eval_tasks:
    task = task_output.task
    for instance in task.instances:
        reqtype = instance.request_type
        requests[reqtype].append(instance)

# Lines 562-598: Execute requests by type
for reqtype, reqs in requests.items():
    eval_logger.info(f"Running {reqtype} requests")
    # Batching happens inside the LM's request method
    resps = getattr(lm, reqtype)(reqs)

    # Store responses in instances
    for x, resp in zip(reqs, resps):
        x.resps.append(resp)
```

**Batching Behavior**:
- Requests are grouped by type (`loglikelihood`, `generate_until`, etc.)
- The LM class handles batching internally based on `batch_size`
- Each Instance gets its response stored in `instance.resps`

---

## 6. Filter Pipeline

**File**: `lm_eval/evaluator.py` (line 600) and `lm_eval/api/filter.py`

### 6.1 Applying Filters

```python
# Line 600: Apply filters to all responses
task.apply_filters()
```

### 6.2 Filter Ensemble

**File**: `lm_eval/api/filter.py` (lines 45-56)

```python
def apply(self, instances: List[Instance]) -> None:
    resps, docs = zip(*((inst.resps, inst.doc) for inst in instances))
    resps, docs = list(resps), list(docs)

    for f in self.filters:
        # Apply filters in sequence (pipeline)
        resps = f().apply(resps, docs)

    # Store filtered responses with filter ensemble name as key
    for inst, resp in zip(instances, resps):
        inst.filtered_resps[self.name] = resp
```

**Example**: For GSM8K, the "strict-match" filter extracts the answer using regex `"#### (\\-?[0-9\\.\\,]+)"`.

---

## 7. Per-Sample Metric Computation

**File**: `lm_eval/evaluator.py` (lines 602-661)

### 7.1 Computing Metrics for Each Document

```python
# Lines 602-661: Collect per-sample metrics
for filter_key in task.instances[0].filtered_resps.keys():
    doc_iterator = task.doc_iterator(rank=rank, limit=limit, world_size=world_size)

    for doc_id, doc in doc_iterator:
        # Get all instances for this document
        requests = instances_by_doc_id[doc_id]

        # Call task.process_results() with filtered responses
        metrics = task.process_results(
            doc,
            [req.filtered_resps[filter_key] for req in requests]
        )

        # Store per-sample metric values
        for metric, value in metrics.items():
            task_output.sample_metrics[(metric, filter_key)].append(value)
```

### 7.2 process_results() Implementation

**For multiple_choice tasks** (`lm_eval/api/task.py`, lines 1600-1658):

```python
def process_results(self, doc, results):
    # results = list of log-likelihoods for each choice
    gold = self.doc_to_target(doc)  # Ground truth index

    # Compute per-sample accuracy
    pred = np.argmax(results)  # Predicted choice
    acc = 1.0 if pred == gold else 0.0

    # Length-normalized accuracy
    completion_len = np.array([len(i) for i in self.doc_to_choice(doc)])
    pred_norm = np.argmax(results / completion_len)
    acc_norm = 1.0 if pred_norm == gold else 0.0

    return {
        "acc": acc,
        "acc_norm": acc_norm,
    }
```

**For generate_until tasks** (`lm_eval/api/task.py`, lines 1667-1740):

```python
def process_results(self, doc, results):
    gold = self.doc_to_target(doc)
    result = results[0]  # Generated text

    result_dict = {}
    for metric in self._metric_fn_list.keys():
        # Call metric function with references and predictions
        result_score = self._metric_fn_list[metric](
            references=[gold],
            predictions=[result],
            **self._metric_fn_kwargs[metric]
        )
        result_dict[metric] = result_score

    return result_dict
```

**Critical Point**: Metrics are computed **per-sample** (one document at a time), NOT batched. Each call to `process_results()` handles a single document.

---

## 8. Aggregation Phase

**File**: `lm_eval/evaluator_utils.py` (lines 105-129)

### 8.1 Aggregating Per-Sample Metrics

```python
def calculate_aggregate_metric(self, bootstrap_iters=100000) -> None:
    for (metric, filter_key), items in self.sample_metrics.items():
        # Get aggregation function for this metric
        try:
            agg_fn = self.task.aggregation()[metric]
        except KeyError:
            # Default to mean if not specified
            agg_fn = mean

        # Apply aggregation function to all per-sample values
        metric_key = f"{metric},{filter_key}"
        self.agg_metrics[metric_key] = agg_fn(items)
        self.sample_len = len(items)

        # Calculate standard error if requested
        if isinstance(bootstrap_iters, int):
            stderr_fn = stderr_for_metric(metric=agg_fn, bootstrap_iters=...)
            if stderr_fn is not None:
                self.agg_metrics[f"{metric}_stderr,{filter_key}"] = stderr_fn(items)
```

### 8.2 Common Aggregation Functions

**File**: `lm_eval/api/metrics.py`

```python
@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)

@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))

@register_aggregation("bleu")
def bleu(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    return sacrebleu.corpus_bleu(preds, refs).score
```

**Aggregation Types**:
1. **Simple aggregations**: `mean`, `median` - operate on scalar values
2. **Corpus-level aggregations**: `bleu`, `chrf`, `ter` - need all references and predictions
3. **Transform aggregations**: `perplexity` - transform the mean of log-likelihoods

---

## 9. Standard Error Calculation

**File**: `lm_eval/api/metrics.py` (lines 506-577)

### 9.1 Bootstrap Standard Error

```python
def bootstrap_stderr(f, xs, iters):
    """
    Bootstrap estimate of standard error using resampling
    - f: aggregation function
    - xs: per-sample metric values
    - iters: number of bootstrap iterations (default 100,000)
    """
    res = []
    chunk_size = min(1000, iters)

    with mp.Pool(mp.cpu_count()) as pool:
        for bootstrap in tqdm(pool.imap(...), total=iters // chunk_size):
            # Resample with replacement and compute metric
            res.extend(bootstrap)

    # Return sample standard deviation of bootstrap distribution
    return sample_stddev(res)
```

### 9.2 Closed-Form Standard Error

```python
def mean_stderr(arr):
    """Standard error for mean: σ / √n"""
    return sample_stddev(arr) / math.sqrt(len(arr))

def stderr_for_metric(metric, bootstrap_iters):
    """Return stderr function for a metric"""
    if bootstrap_iters <= 0:
        return None

    # Metrics that use bootstrap
    bootstrappable = [median, matthews_corrcoef, f1_score,
                     perplexity, bleu, chrf, ter, nanmean]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    # Metrics with closed-form stderr
    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}
    return stderr.get(metric, None)
```

**Key Points**:
- **Bootstrap** is used for complex metrics (BLEU, perplexity, etc.)
- **Closed-form** is used for simple metrics (mean, accuracy)
- Default: 100,000 bootstrap iterations (can be disabled with `--bootstrap_iters 0`)

---

## 10. Group Metrics

**File**: `lm_eval/evaluator.py` (lines 700-788)

### 10.1 Aggregating Across Task Groups

```python
# Lines 700-788: Group aggregation
for group, task_list in reversed(task_hierarchy.items()):
    for task_name in task_list:
        task_obj = task_dict[task_name]

        # Aggregate metrics across subtasks
        for metric in task_obj.aggregation().keys():
            # Collect metric values from all subtasks
            group_metrics = [subtask.agg_metrics[metric]
                           for subtask in subtasks]

            # Apply group aggregation (usually weighted mean)
            agg_fn = task_obj.group_aggregation().get(metric, mean)
            task_obj.agg_metrics[metric] = agg_fn(group_metrics)
```

**Example**: MMLU has multiple subtasks (mmlu_anatomy, mmlu_astronomy, etc.). The group metric is the mean across all subtasks.

---

## 11. Concrete Examples

### 11.1 Example: ARC-Easy (Multiple Choice)

**Task Config**: `lm_eval/tasks/arc/arc_easy.yaml`

```yaml
task: arc_easy
output_type: multiple_choice
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.text}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
```

**Workflow**:
1. **Request Building**: For each question, create instances for each choice
2. **Model Execution**: Compute log-likelihood for each choice
3. **Filtering**: No filters (raw log-likelihoods used)
4. **Per-Sample Metrics**:
   - `acc = 1.0 if argmax(lls) == gold else 0.0`
   - `acc_norm = 1.0 if argmax(lls/len) == gold else 0.0`
5. **Aggregation**: `mean(acc_values)` across all questions
6. **Stderr**: `mean_stderr(acc_values)` (closed-form)

### 11.2 Example: GSM8K (Generative)

**Task Config**: `lm_eval/tasks/gsm8k/gsm8k.yaml`

```yaml
task: gsm8k
output_type: generate_until
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    regexes_to_ignore: [",", "\\$", "(?s).*#### ", "\\.$"]
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
```

**Workflow**:
1. **Request Building**: For each question, create a generate_until request
2. **Model Execution**: Generate text until stop sequence
3. **Filtering**:
   - Apply regex to extract answer: `"#### (\\-?[0-9\\.\\,]+)"`
   - Take first match
4. **Per-Sample Metrics**:
   - `exact_match(prediction, reference, ignore_case=True, regexes_to_ignore=[...])`
   - Returns 1.0 if match, 0.0 otherwise
5. **Aggregation**: `mean(exact_match_values)` across all questions
6. **Stderr**: `mean_stderr(exact_match_values)` (closed-form)

---

## 12. Key Takeaways

### 12.1 When Are Metrics Computed?

1. **Metric initialization**: During task loading (from YAML config)
2. **Per-sample computation**: After model execution and filtering, in `process_results()`
3. **Aggregation**: After all documents are processed, in `calculate_aggregate_metric()`
4. **Group aggregation**: After all tasks in a group are complete

### 12.2 On How Many Samples?

- **Per-sample metrics**: Computed **one document at a time** (not batched)
- **Aggregation**: Applied to **all per-sample values** for the entire dataset
- **Batching**: Only happens during model execution (LM batches requests internally)

### 12.3 How Does Aggregation Work?

1. **Per-sample values** are collected in `task_output.sample_metrics[(metric, filter_key)]`
2. **Aggregation function** is retrieved from `task.aggregation()[metric]`
3. **Aggregation** is applied: `agg_fn(all_sample_values)`
4. **Result** is stored in `task_output.agg_metrics[f"{metric},{filter_key}"]`

### 12.4 Important Design Patterns

1. **Passthrough Metrics**: Many metrics (acc, exact_match) are "passthrough" - they return the per-sample value as-is, and aggregation happens separately
2. **Corpus-Level Metrics**: Some metrics (BLEU, F1) need all samples together, so they're implemented as aggregation functions
3. **Filter Ensembles**: Multiple filter pipelines can be applied, creating separate metric values for each filter
4. **Metric Registration**: Metrics are registered with decorators (`@register_metric`, `@register_aggregation`)

### 12.5 Data Flow Summary

```
YAML Config → Task Initialization → Metric Registration
                                          ↓
Dataset → build_all_requests() → Instances (one per document)
                                          ↓
Instances → LM.request() → Responses (batched internally)
                                          ↓
Responses → FilterEnsemble.apply() → Filtered Responses
                                          ↓
(doc, filtered_resps) → process_results() → Per-Sample Metrics
                                          ↓
Per-Sample Metrics → calculate_aggregate_metric() → Aggregated Metrics
                                          ↓
Aggregated Metrics → stderr_for_metric() → Standard Errors
                                          ↓
Task Metrics → Group Aggregation → Final Results
```

---

## 13. Files Reference Summary

| Component | File | Key Lines |
|-----------|------|-----------|
| CLI Entry | `lm_eval/__main__.py` | 301-474 |
| Main Evaluator | `lm_eval/evaluator.py` | 70-788 |
| Task Base Classes | `lm_eval/api/task.py` | 1-1892 |
| Metric Functions | `lm_eval/api/metrics.py` | 1-627 |
| Metric Registry | `lm_eval/api/registry.py` | 76-91 |
| Filter System | `lm_eval/api/filter.py` | 1-200 |
| TaskOutput | `lm_eval/evaluator_utils.py` | 22-139 |
| Instance | `lm_eval/api/instance.py` | 1-38 |

---

## Conclusion

This analysis provides a complete understanding of the metrics computation workflow in lm-evaluation-harness. The system is designed to be:

- **Flexible**: Metrics can be easily added via registration decorators
- **Extensible**: Custom metrics and aggregations can be defined in task configs
- **Efficient**: Batching happens at the model execution level
- **Modular**: Clear separation between per-sample computation and aggregation
- **Robust**: Filter pipelines allow for flexible post-processing of model outputs

The key insight is that **metrics are computed per-sample** (one document at a time) in `process_results()`, then **aggregated across all samples** in `calculate_aggregate_metric()`. This design allows for maximum flexibility in defining both simple metrics (like accuracy) and complex corpus-level metrics (like BLEU).

