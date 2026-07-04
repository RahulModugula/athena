# What I learned building a zero-shot RAG hallucination detector

I spent the last few months building [athena-verify](https://github.com/RahulModugula/athena), an open-source runtime guardrail that checks whether an LLM answer is actually grounded in the context you retrieved, sentence by sentence, locally, with no API calls. This is the honest write-up: what worked, where it hits a ceiling, and the two things I got wrong on the way.

None of the numbers here are cherry-picked. Every figure is reproducible from `benchmarks/` in the repo, and where athena loses to a fine-tuned model, I say so.

## The problem

RAG systems make things up, and they do it with total confidence. Retrieval looks fine, the answer reads fine, but one sentence quietly invents a number or flips a negation, and there is no signal until a user catches it. The existing tools mostly tell you this after the fact: Ragas, TruLens, and DeepEval are batch evaluators that report a faithfulness rate over a test set. They do not stand between the model and the user at request time, and they do not tell you *which sentence* is the problem.

I wanted something boring and deployable: a function you call on a single answer, in-request, that returns a per-sentence verdict and the source span behind it. No document ingestion, no chunking, no database, no fine-tuning on my data.

## The approach: split, score, aggregate

The core is deliberately simple so it runs anywhere:

1. Split the answer into sentences.
2. Split the context into sentences too.
3. Score each answer sentence against the context with a natural-language-inference (NLI) cross-encoder, taking the best supporting match.
4. Add a lexical-overlap signal and combine into a trust score.
5. Optionally escalate borderline sentences to a local LLM judge.

The one implementation detail that mattered more than anything else: **score against individual context sentences, not the whole blob.** When you feed the NLI model one long context string, it sees information beyond the claim and labels almost everything "neutral." Scoring each context sentence as its own premise and taking the max recovers real entailment signal, and it is the difference between a model that can read a number and one that can't.

I use DeBERTa-v3 as the cross-encoder. It is small (~1.2 GB), runs at about 22 ms p50 on an M1 Max, and it is zero-shot — not trained or tuned on any hallucination dataset.

## Mistake #1: I optimized for catching hallucinations and ignored the false-positive rate

The first version caught hallucinations beautifully and flagged faithful sentences constantly. On my synthetic set the false-positive rate on genuinely-supported sentences was **17%**. A guardrail that cries wolf on one in six good sentences is worse than useless — people turn it off.

The root cause was not the model. It was three specific failure modes:

- **Anaphora.** A sentence like "It also caps annual liability at $5M" scores as unsupported because "It" has no antecedent in isolation. Fix: when a sentence opens with a referent, score it together with its predecessor so the antecedent is restored.
- **Faithful paraphrases scoring as neutral.** NLI often lands a fully-supported rephrase in the neutral band. Fix: a rescue step that only fires when the claim is *not contradicted* by the most on-topic context, most of its words are grounded, and — critically — every number in it appears in the context. That numeric gate is what stops the rescue from waving through "$1M" when the context says "$2M."
- **Meta sentences.** "The passages do not mention X" and refusals are not claims to verify. A check-worthiness filter skips them.

Those three changes took the false-positive rate from **16.9% to 4.6%** on the base model (3.4% on the large one) without letting hallucinations through — synthetic hallucination-catch F1 actually went *up*, from 91.3% to **95.0%**. The lesson: for a guardrail, the false-positive rate is the product. Everyone benchmarks recall; the thing that decides whether anyone keeps it on is precision on clean text.

## Mistake #2: I assumed synthetic numbers meant something

95% F1 on a synthetic set that I generated is a nice number and it proves almost nothing, because I wrote both the hallucinations and the ground truth. So I ran the real benchmarks the field uses, zero-shot, and reported them straight:

| Benchmark | Metric | athena (zero-shot, local, ~25 ms) | Reference |
|---|---|---|---|
| RAGTruth QA | balanced accuracy | **0.71** | LettuceDetect 0.70 F1, but fine-tuned on RAGTruth |
| HaluEval QA | accuracy | **0.69** | GPT-3.5 ≈ 0.62, GPT-4 ≈ 0.85 (prompted, API) |

On RAGTruth's imbalanced, 18%-positive response-level F1, athena scores about **0.47** — and I put that in the README too, because that is the honest number and class imbalance is why balanced accuracy is the fair metric. LettuceDetect reports ~79% F1 on RAGTruth, but it is a ModernBERT detector fine-tuned on RAGTruth's own training split. That accuracy is real and it is domain-specific: it does not transfer to your corpus. Athena trades in-domain accuracy for working on any corpus with zero training.

Here is the thing I did not want to admit and now think is the most useful conclusion: **a zero-shot NLI-then-aggregate pipeline has a ceiling on abstractive RAG answers, and swapping the checkpoint doesn't break it.** I tried MiniCheck-DeBERTa as a drop-in backend; it did not beat the default on RAGTruth QA. The limitation is the paradigm, not the model. If you want to top an in-domain benchmark, you fine-tune on that benchmark, which is exactly what the SOTA detectors do.

## Where this actually leaves a zero-shot library in 2026

By mid-2026 the "local span-level detector" lane is crowded — LettuceDetect keeps shipping better trained detectors, and inference stacks are starting to bake groundedness gates in directly. Competing on raw detection F1 against a model fine-tuned on the exact benchmark is a losing game, and pretending otherwise is how you get taken apart in the comments.

So I stopped framing athena as "another detector" and started treating detection as the cheap first layer under two things that are genuinely underserved:

- **A revision step, not just a score.** Almost every tool flags; very few propose the corrected sentence in the same pass. `suggest_revisions=True` returns the fix, not just the flag.
- **A framework-agnostic circuit-breaker for agents.** In a multi-step agent, one step's output is the next step's input, so a fabricated fact cascades and you find out at the end. `verify_step()` returns a pass/halt verdict against a step's evidence so you can stop the chain the moment it stops being grounded — in any agent loop, not welded to one serving stack.

```python
from athena_verify import verify_step

step = verify_step(claim=reasoning_step, evidence=retrieved_chunks, threshold=0.5)
if step.action == "halt":
    raise RuntimeError(f"ungrounded step blocked, trust={step.trust_score:.2f}")
```

The detector is zero-shot and provider-neutral by design, so the same code path runs on GPT, Claude, Llama, or Qwen output, and the NLI backend is swappable — a stronger trained detector can slot in underneath the same revision and circuit-breaker layer.

## What's next

- An opt-in trained backend (LettuceDetect's recipe) for people who want in-domain SOTA F1, keeping zero-shot as the default so the any-corpus pitch holds.
- Cheap zero-shot accuracy gains that keep the default honest: gated embedding-cosine rescue for low-overlap paraphrases, top-k premise retrieval before NLI.
- A short benchmark write-up on honest zero-shot grounding, with the RAGTruth and HaluEval harnesses published for reproduction.

The synthetic and real-world benchmarks are both in the repo, reproducible with one command each. If you run athena on your own data and it breaks, I want the failure case — that is the fastest way this gets better.

---

*All numbers are from real runs with deterministic seeds on an Apple M1 Max. See [benchmarks/RESULTS.md](https://github.com/RahulModugula/athena/blob/main/benchmarks/RESULTS.md) for full methodology and reproduction.*
