# athena-verify — Next Iteration Plan

> Written 2026-06-28, after the false-positive-calibration + real-world-benchmark
> work landed on `fix/false-positive-calibration`. This file is the handoff for
> the next iteration.

## Where we are

**Shipped this iteration (branch `fix/false-positive-calibration`):**
- Fixed the false-positive root causes (anaphora windowing, full-chunk premise,
  model-agnostic NLI labels, abbreviation-aware splitter). Faithful FP rate
  **16.9% → 4.6%** (base) / 3.4% (large); synthetic F1 **91.3% → 95.0%**.
- Contradiction-aware grounding rescue + numeric gate (recover faithful
  paraphrases without passing number swaps / contradictions).
- Check-worthiness filter (skip questions / refusals / "passages don't mention X").
- Optional **MiniCheck-DeBERTa-v3-Large** backend (`nli_model="minicheck"`), no
  new deps. Benchmarked: does **not** beat default NLI on RAGTruth-QA, kept opt-in.
- Real-world benchmarks run (RAGTruth QA, HaluEval QA) + honest README/RESULTS.
- Packaging: `pip install athena-verify` works cold; build + twine check pass.
- Agent circuit-breaker demo + GIF; measured-improvements plot.
- All green: ruff, mypy --strict, 162 tests.

**Honest positioning (decided):** "best *practical* zero-shot local hallucination
guardrail." Zero-shot real-world accuracy is competitive (RAGTruth QA balanced
acc **0.71**, HaluEval QA **0.69** — GPT-3.5-to-GPT-4 prompted range), and athena
wins decisively on local/offline, provider-neutral, latency (~25 ms), per-claim
spans, and the agent circuit-breaker. We do **not** claim accuracy-SOTA; that
requires training on the benchmark (which is what LettuceDetect does).

## The ceiling we hit, and why

Zero-shot sentence-level NLI + aggregation caps at ~0.47 RAGTruth-QA response-F1.
LettuceDetect reaches 0.70 by **fine-tuning a ModernBERT token-classifier on the
RAGTruth training split**. A model swap (MiniCheck) did not break the ceiling —
the limitation is the zero-shot, sentence-then-aggregate paradigm on abstractive
RAG answers, not the specific checkpoint.

## Next iteration — prioritized

### P0 — Optional trained backend (the "beat in-domain SOTA" path)
Train a small detector on RAGTruth (LettuceDetect's recipe) and ship it as an
**opt-in** backend (`nli_model="trained"`), keeping zero-shot as the default so
the any-domain pitch holds.
- Base: `answerdotai/ModernBERT-base` (Apache, 8k ctx) token-classification, or
  fine-tune `MiniCheck-DeBERTa` on RAGTruth pairs.
- Data: RAGTruth train (15,090 responses) → token/sentence labels from spans.
- Target: match/approach LettuceDetect 0.79 overall / 0.70 QA response F1.
- Needs a GPU (CPU training is impractical). Add `scripts/train_detector.py`.
- Risk: domain overfit; report cross-dataset (HaluEval) numbers honestly.

### P1 — Squeeze zero-shot accuracy further (cheap, keeps default)
- **Embedding-cosine backoff** (task #9): gated SBERT cosine rescue in the
  neutral band for low-lexical-overlap paraphrases ("olive oil is drizzled on
  top"). Measure on synthetic + RAGTruth; only ship if it helps both.
- **Top-k premise retrieval** before NLI (BM25/embedding) — the research's
  FEVER-style sentence selection; may lift precision on long contexts.
- **FENICE-style claim decomposition** + per-claim judge — the research's named
  path to actually beat LettuceDetect zero-shot; larger effort, uncertain.

### P1 — Launch execution (non-modeling)
- Publish to PyPI: tag `v0.1.0` (trusted publishing workflow is ready).
- Publish the VS Code extension / verify the quickstart on a clean VM.
- Open the PR for `fix/false-positive-calibration`; squash-merge to main.
- README polish: ensure `assets/benchmarks.png` + GIF render on GitHub.

### P2 — Robustness / DX
- LLM-judge retry/backoff + circuit-breaker on API timeouts.
- Per-sentence latency histograms; wire the OTel/Langfuse exporters that exist
  but aren't called.
- Tune-on-train threshold shipped as the default response-level rule (currently
  the per-sentence default is calibrated on synthetic; expose a
  `response_threshold` knob informed by the RAGTruth sweep).

## Benchmark harnesses (for reproduction)
RAGTruth: clone `github.com/ParticleMedia/RAGTruth`, join `source_info.jsonl` +
`response.jsonl` on `source_id`, filter `task_type=="QA"`, gold = `labels`
non-empty. HaluEval: `RUCAIBox/HaluEval` `qa_data.json` (right vs hallucinated
answer). Both use example/response-level metrics; tune threshold on train/dev,
report on test.
