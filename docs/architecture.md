# Architecture

```
Answer → Sentence Splitter → [Sentence₁, Sentence₂, ...]
                                    ↓
                        ┌───────────────────────────┐
                        │  For each sentence:        │
                        │  1. NLI entailment vs ctx  │
                        │  2. Lexical overlap vs ctx │
                        │  3. (optional) LLM judge   │
                        │  4. Calibrate → trust      │
                        └───────────────────────────┘
                                    ↓
                    VerificationResult (per-sentence + overall)
```

## Verification Pipeline

### 1. Sentence Splitting

The answer is split into sentences using regex-based boundary detection (with optional NLTK support). Each sentence is verified independently.

### 2. NLI Entailment Scoring

Each sentence is paired with the concatenated context chunks and scored by a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base` by default). The model outputs entailment probability — how likely the context logically supports the sentence.

### 3. Lexical Overlap

Token-level F1 is computed between each sentence and the best-matching context chunk. This catches surface-level grounding even when the NLI model is uncertain.

### 4. LLM-as-Judge (Optional)

For borderline cases (NLI scores 0.3–0.7), an LLM can be used as an additional judge. The LLM receives the context and sentence and returns a structured judgment. This adds an API cost but improves accuracy on ambiguous cases.

### 5. Trust Score Calibration

Signals are combined using a weighted ensemble:
- NLI entailment: 55%
- Lexical overlap: 25%
- LLM-as-judge: 20% (when enabled; weight redistributed to NLI/overlap when disabled)

The resulting trust score (0.0–1.0) is classified as SUPPORTED (≥0.75), PARTIAL (≥0.50), UNSUPPORTED (≥0.30), or CONTRADICTED (<0.30).

## Package Structure

```
athena_verify/
├── __init__.py          # Public API: verify(), verify_async(), etc.
├── core.py              # Main verify() implementation
├── models.py            # VerificationResult, SentenceScore, Chunk
├── nli.py               # NLI model loading and entailment scoring
├── overlap.py           # Token F1 overlap computation
├── calibration.py       # Trust score calibration and classification
├── llm_judge.py         # Optional LLM-as-judge scoring
├── parser.py            # Sentence splitting
└── integrations/
    ├── langchain.py     # VerifyingLLM wrapper
    └── llamaindex.py    # VerifyingPostprocessor wrapper
```
