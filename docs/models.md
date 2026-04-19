# NLI Model Trade-offs

Different NLI models offer different speed/accuracy trade-offs. Choose based on your latency and accuracy needs.

## Available Models

| Model | Alias | Size | Latency | F1 | Best For |
|-------|-------|------|---------|-----|----------|
| **DeBERTa-v3 Base** | `default` | 700 MB | ~12ms | 91.3% | ✓ Recommended; best balance |
| **DeBERTa-v3 Large** | explicit URL | 1.8 GB | ~45ms | 92.5% | High accuracy (legal, finance) |
| **MiniLM L6** | `lightweight` | 80 MB | ~3ms | 85.0% | Mobile/edge; speed priority |
| **Vectara eval** | `vectara` | 400 MB | ~8ms | 88% | Specialized for hallucination |

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
