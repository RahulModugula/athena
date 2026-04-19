# Threshold Tuning Guide

The `trust_threshold` parameter controls what Athena flags as unsupported. Higher thresholds = fewer false positives but more hallucinations slip through.

## Domain Recommendations

| Domain | Threshold | Notes |
|--------|-----------|-------|
| **Legal / Financial** | 0.85–0.95 | High stakes: liability risk. Better to flag for review. |
| **Medical / Safety** | 0.80–0.90 | Patient/user safety critical. Be conservative. |
| **Product docs** | 0.70–0.80 | Errors hurt usability. Moderate strictness. |
| **Customer support** | 0.65–0.75 | Inconvenient but not critical. Allow some slack. |
| **Blog / content** | 0.60–0.70 | Readers expect casual tone and minor imprecision. |

## How to Choose

1. **Start with domain baseline** from table above
2. **Validate on 50–100 real examples** — check false positives match your tolerance
3. **Consider the cost:**
   - False positive (flag correct answer) vs false negative (miss error)
   - If false positives hurt more → lower threshold
   - If false negatives hurt more → raise threshold

Example: Support chatbot

```python
result = verify(
    question="What's the warranty?",
    answer="It's 2 years from purchase.",
    context=knowledge_base,
    trust_threshold=0.68,  # Relaxed; err on side of caution
)
```

## With LLM Judge

Combine with LLM judge for edge cases:

```python
result = verify(
    ...,
    trust_threshold=0.70,
    use_llm_judge=True,  # Escalates uncertain cases to LLM
    llm_client=judge,
)
```

This lets you be strict (0.70) while delegating borderline decisions to an LLM.
