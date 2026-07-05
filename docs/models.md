# NLI Model Trade-offs

Different NLI models offer different speed/accuracy trade-offs. Choose based on your latency and accuracy needs.

## Available Models

Only the **default** row is measured on the committed benchmark
(`benchmarks/run_full_eval.py`: 95.0% synthetic hallucination-catch F1, 4.6%
false-positive rate, p50 22.5 ms on an M1 Max). The other rows are indicative
size/latency guidance — swap the model and re-run the eval to get real numbers
for your hardware before relying on them.

| Model | Alias | Approx. size | Approx. latency | Best For |
|-------|-------|------|---------|----------|
| **DeBERTa-v3 Base** | `default` | ~1.2 GB | ~22 ms | ✓ Recommended; best balance (benchmarked) |
| **DeBERTa-v3 Large** | explicit URL | ~1.8 GB | ~45 ms | High accuracy (legal, finance); GPU helps |
| **MiniLM L6** | `lightweight` | ~80 MB | ~3 ms | Mobile/edge; speed priority |
| **MiniCheck-DeBERTa** | `minicheck` | ~1.4 GB | ~30 ms | Fact-checking-tuned; opt-in |

## When to Choose

**DeBERTa-v3 Base (default)**
```python
result = verify(question, answer, context)
```
- Best balance of latency and accuracy
- Handles paraphrases well
- Recommended for most use cases

**Lightweight (speed priority)**
```python
result = verify(question, answer, context, nli_model="lightweight")
```
- 3ms per sentence; sub-3KB memory
- Misses ~6% of hallucinations
- Good for real-time, high-volume workloads

**Large (high accuracy)**
```python
result = verify(question, answer, context, nli_model="cross-encoder/nli-deberta-v3-large")
```
- +1.2% accuracy; 3.5x slower
- Needs GPU
- For legal/financial where false negatives are costly

**Vectara (hallucination-focused)**
```python
result = verify(question, answer, context, nli_model="vectara")
```
- Specialized for hallucination detection
- Good on contradictions
- Similar speed to base

## Latency-Aware Selection

Use `latency_budget_ms` to auto-select models:

```python
result = verify(
    question, answer, context,
    nli_model="lightweight",      # Fast baseline
    use_llm_judge=True,           # Upgrade uncertain cases
    latency_budget_ms=500,        # Total budget cap
    llm_client=judge,
)
```

If lightweight isn't confident, escalate to LLM judge.
