# The Athena-Verify Launch Plan

> Supersedes `WORLDCLASS_PLAN.md` (2026-04-17). This is the active plan.
> Based on code audit + web research on competitors, dev pain points, and viral dev-tool launch patterns in April 2026.

## TL;DR

You have an **80% solid product** with a **20% credibility problem** that would get you dunked on in the first HN comment. There is a genuine, defensible wedge for you to own — but a library called **LettuceDetect** (KRLabs, Feb 2025, MIT, ModernBERT, 79.2% F1 on RAGTruth) is the direct threat you probably don't know about, and if your launch post doesn't address it head-on, the first commenter will. Good news: your product differentiates on **layered architecture + revision loop + hybrid NLI/LLM**, none of which LettuceDetect or HHEM ship.

The plan below is **~3 weeks of focused work** → launch → hiring-grade signal.

---

## Part 1: The brutal truth about the current state

### What's genuinely strong
- Core `verify()` API is clean, typed, has streaming + async + batch variants.
- Quickstart example is well-chosen (legal contract with number-substitution — the exact failure mode NLI is weak at, so the demo lands).
- Hybrid NLI + LLM-judge architecture is the right shape.
- `suggest_revisions=True` — the code is there to output corrections, not just detect.
- Test coverage breadth is decent (123 tests).

### Launch blockers (must-fix before anyone sees this)
1. **Real-world benchmarks are unrun.** `run_ragtruth.py`, `run_halueval.py`, `run_factscore.py` exist and are wired — but `benchmarks/results/` only contains `full_eval.json` from the synthetic set. README cites **100 synthetic cases**. First HN comment: *"why no RAGTruth?"* Game over.
2. **README numbers don't match the committed JSON.** README says hybrid 81%. `full_eval.json` shows 52.9%. Someone will notice.
3. **Not on PyPI.** `pip install athena-verify` fails. Version 0.1.0, no tags, no publish workflow.
4. **Integration examples are commented-out stubs.** `examples/langchain_example.py` and `llamaindex_example.py` are skeleton comments. Reddit/HN expect copy-paste-runs-working examples.
5. **Suggest-revisions feature exists in code but is off by default, undocumented, untested, undemo'd.** This is your wow-factor feature sitting in the dark.

### Silent gaps (not blockers, but weak)
- No demo GIF or video.
- OTel/Langfuse trace methods exist on result objects but are never called from `verify()`.
- `LLMClient` type isn't exported from `__init__.py` even though it's in the public type signature.
- No Grafana/Datadog exporter — which r/LocalLLaMA devs specifically want.
- No span-level "show me which source sentence supports this claim" — HN devs are asking for this.

---

## Part 2: The competitive reality (April 2026)

| Competitor | Runtime? | Sentence-level? | Local+Free? | Revision loop? | Your edge |
|---|---|---|---|---|---|
| **LettuceDetect** (MIT, Feb 2025) | Yes | Span-level | Yes | **No** | **Revisions + hybrid fallback + integrations** |
| **Vectara HHEM v2.1** (Apache) | Yes | Passage only | Yes | No | **Granularity + revisions** |
| **Patronus Lynx** (CC-BY-NC) | Yes | Yes | 8B needs GPU | No | **MIT license + speed** |
| **Ragas** (Apache) | **No, offline** | Claim-level | Depends | No | **Runtime** |
| **TruLens** (MIT) | No | No | Depends | No | Runtime + simplicity |
| **Guardrails AI ProvenanceLLM** | Yes | Coarse | Requires Cohere | No | **Local + no API key** |
| **Cleanlab TLM** | Yes | Yes | **No, SaaS** | No | **Local** (also: just acquired by Handshake — uncertain future) |
| **Galileo Luna-2** | Yes | Yes | **No, SaaS** | No | **Local** (also: Cisco acquiring April 9) |
| **NeMo Guardrails** | Yes | No | Yes | No | Simplicity |

### The defensible wedge — own this sentence
> *"The only MIT-licensed library that does **sentence-level, runtime, hybrid NLI + local-LLM-judge** verification with **built-in revision suggestions**, pip-installs in 60 seconds, and runs entirely on a MacBook."*

Four adjectives together that **no single competitor ships**:
1. Sentence-level (HHEM is passage).
2. Hybrid architecture — cheap NLI fast path + local LLM fallback on ambiguous cases (HHEM is classifier-only; Lynx is LLM-only; Cleanlab is cloud-only).
3. Revision suggestions in the same pipeline — *literally no one else does this as a runtime feature.*
4. MIT (Lynx is non-commercial; Galileo/Cleanlab are closed).

### What developers are literally begging for
- **LangChain Issue #33191** (open since Oct 2025) explicitly asks for a `HallucinationDetector` that does `context-NLI` at runtime. Athena is the fix. **Quote this in your launch.**
- **LlamaIndex Issue #21213** wants post-RAG verification with per-claim citations.
- **Ragas Issue #2161** — users begging to escape the OpenAI dependency for evals.
- Universal complaint across HN/Reddit: *"Ragas tells me faithfulness=0.87 but not which sentence is lying."* This is your headline.

---

## Part 3: Three fundamental product improvements before you launch

These are not polish. Each changes what the product **is** in a way that determines whether the launch clicks or fizzles.

### Improvement #1 — Ship real-world benchmarks on RAGTruth and HaluEval
**Why:** Credibility. Synthetic-only = amateur. RAGTruth + HaluEval = serious.
**What:** Actually run the existing `run_ragtruth.py` and `run_halueval.py`. Commit `benchmarks/results/ragtruth.json` and `halueval.json`. Update README table to include:

| Benchmark | Athena NLI | Athena Hybrid | LettuceDetect | HHEM v2.1 | GPT-4 judge | Ragas faith. |
|---|---|---|---|---|---|---|
| RAGTruth QA | ? | ? | 79.2% | ~70% | ~64% | ~55% |
| HaluEval QA | ? | ? | ? | ? | ? | ? |

**Honesty move:** include the row where you lose (e.g., if LettuceDetect beats you on span-level F1, say so — then point to revisions + hybrid as where you win). This is the exact honesty lever that drives HN top comments.

**Effort:** 2–3 days (download datasets, run scripts, write up).

### Improvement #2 — Turn `suggest_revisions` into the headline feature
Right now it's hidden behind a flag. Make it the headline:

```python
from athena_verify import verify

result = verify(question=..., answer=..., context=..., suggest_revisions=True)

for unsupported in result.unsupported:
    print(f"❌ {unsupported.text}")
    print(f"✅ {unsupported.suggested_fix}")   # <-- this is the magic
```

No other open-source competitor ships this. This is the "auto-fix hallucination" demo that makes the Reddit GIF pop.

**Effort:** 1 day — make it the default in examples, add a dedicated section in README, record the GIF around this.

### Improvement #3 — Add a `VerifyingLLM` retry loop for LangChain
LangChain #33191 specifically asks for *"trigger automated fallback (retry, re-retrieval)."* You already detect. You already revise. Close the loop:

```python
chain = VerifyingLLM(llm, retriever=retriever, max_retries=2, on_unsupported="re-retrieve")
```

If verification fails, re-retrieve with the unsupported claims as additional queries and retry. You'd be the **first OSS runtime verifier with a self-healing loop** — and you can quote LangChain issue #33191 word-for-word in the launch post.

**Effort:** 2–3 days.

---

## Part 4: The wow-factor additions (stretch, but each is a tweet-magnet)

Pick 2 of these based on time:
- **Terminal GIF in README** (15 sec: hallucinated answer in → red/green highlighted output). **Non-negotiable.** ~2 hours.
- **Source-span highlighting** — not just "this answer sentence failed" but "this source sentence *did* support this claim." Directly addresses the HN complaint. ~1 day.
- **Grafana/OpenTelemetry exporter** — wire the existing `.to_otel_span()` method into `verify()`. r/LocalLLaMA wants this. ~1 day.
- **Ollama integration badge** — you say "LM Studio" everywhere; add Ollama as a first-class LLM judge client. 30 min + testing.
- **Public HuggingFace Space demo** — paste answer + context, get colored verification. This replaces the "playground" launch-pattern advice; hosting on HF is free and doesn't go down. ~half day.

**Don't do:** hosted dashboard, managed SaaS, web UI. Devs don't want these; enterprise incumbents already cover that lane.

---

## Part 5: The 3-week execution plan

### Week 1 — Fix the launch blockers
- [ ] Day 1–2: Download RAGTruth + HaluEval, run `run_ragtruth.py` + `run_halueval.py`, commit results. Reconcile README numbers with actual JSON.
- [ ] Day 3: Rewrite `langchain_example.py` and `llamaindex_example.py` as runnable scripts (not comments). Test each end-to-end. Add `requirements-example.txt`.
- [ ] Day 4: PyPI publish. Add `[project.scripts]` CLI entry point (`athena-verify verify --answer ... --context ...`). Tag v0.1.0 on GitHub. Set up GH Actions for future releases.
- [ ] Day 5: Make `suggest_revisions=True` the default in all examples. Update README with a dedicated "Auto-fix hallucinations" section.

### Week 2 — Fundamental improvements + differentiation
- [ ] Day 6–8: Build `VerifyingLLM` self-healing loop for LangChain. Reference issue #33191 in the PR description and in README.
- [ ] Day 9: Record the 15-second terminal GIF. Put it above the fold in README.
- [ ] Day 10: Ship one wow-factor: source-span highlighting OR HuggingFace Space demo.

### Week 3 — Polish, pre-launch assets, launch
- [ ] Day 11: Final benchmark table vs. LettuceDetect, HHEM, Ragas, GPT-4 judge. Include a row you lose.
- [ ] Day 12: Write the technical blog post (the "here's what I learned building this" version — this is the hiring-signal one).
- [ ] Day 13: Open 3 "good first issue" labeled issues. Pin the repo on your GH profile. Update your X/LinkedIn bios with the repo URL.
- [ ] Day 14: Dress rehearsal. Run quickstart from a fresh Python env. Fix whatever breaks.
- [ ] Day 15 (Tuesday): Launch Show HN at 8:30am ET.
- [ ] Day 15 (Tuesday), 6pm ET: r/LocalLLaMA post.
- [ ] Day 16: r/MachineLearning `[P]` post.
- [ ] Day 17: Blog post cross-posted (as a separate HN link, not another Show HN).

**Stagger, don't blast** — simultaneous multi-channel launches split the comment thread and nobody hits the front page.

---

## Part 6: The launch posts (drafts)

### Show HN title (pick one)
1. **Show HN: Athena-verify – runtime RAG hallucination detection, sentence-level, MIT** ← recommended
2. **Show HN: Athena – the `HallucinationDetector` LangChain has been missing**

### Show HN body (~150 words)
> Athena-verify is an open-source Python library that catches RAG hallucinations at runtime, sentence-by-sentence, running entirely on a MacBook. Three lines of code, no API key.
>
> It splits the LLM answer, scores each sentence against retrieved context using DeBERTa-v3 NLI (~20ms) + lexical overlap, and escalates borderline cases to a local LLM judge via LM Studio/Ollama. On RAGTruth QA it hits [X]% F1; hybrid mode catches number substitutions ("$2M → $1M") at 73% F1 where NLI alone gets 7%.
>
> What's different from Ragas/TruLens: it runs at **request time**, not batch. What's different from HHEM: **sentence granularity** plus **automated revision suggestions**. What's different from Patronus Lynx: **MIT license** and no GPU required.
>
> LettuceDetect beats it on span-level F1; athena wins on revisions + the hybrid fast-path. Both honest numbers in the README.
>
> `pip install athena-verify`
> https://github.com/RahulModugula/athena

### r/LocalLLaMA title
> **I built a local MIT hallucination detector for RAG — sentence-level NLI + LM Studio judge, no API key, RAGTruth/HaluEval results inside**

### r/LocalLLaMA body (~250 words)
Lead with the r/LocalLLaMA-native hooks: privacy, local, zero API. Screenshot of the terminal GIF. Benchmark table. End with a specific ask: *"Would love to test with other local LLM judges — right now I've only tried gemma-4-31b-it, qwen2.5-32b, and llama-3.3-70b via LM Studio."*

### The quote-post bait (post these replies to known accounts)
- Jason Liu (@jxnl): *"Built a sentence-level RAG verifier inspired by Instructor's surgical API philosophy — curious what you'd change about the verify() signature."*
- Jerry Liu (@jerryjliu0): *"Shipped a LlamaIndex VerifyingPostprocessor — would love your feedback on the integration shape, specifically around streaming."*
- Omar Khattab: *"Three hybrid verification signals (NLI + lexical + LLM) combined via calibrated weights. I'd value your read on the calibration choices."*

Reply-with-specific-question converts at ~10× naked mention rate.

---

## Part 7: Maximizing hiring signal specifically

1. **Write the launch as a technical post, not a product post.** The hiring-signal winners (Instructor/jxnl, Marker/VikParuchuri) read as *"here's what I learned,"* not *"here's my product."* Put "why we chose DeBERTa-v3 over plain GPT-4-judge" *above* the install command.
2. **Pin the repo** on your GitHub profile.
3. **Put the repo URL in your X and LinkedIn bios the day before launch.**
4. **Respond to every issue and HN comment within 20 minutes for the first 2 hours.** HN's ranking explicitly rewards comment velocity. Recruiters also read your response style.
5. **Maintainer-quality response for the first month.** Every issue answered within 24 hours. Every PR reviewed within 48. Recruiters *read the issues tab*.
6. **Get one known account to quote-post you** via the reply-with-technical-question tactic above.
7. **Register the repo on Awesome-RAG and Awesome-LLMOps lists within 48 hours of launch** — PR to each.
8. **Submit to weekly newsletters** (Ben's Bites, TLDR AI, The Batch) the morning of the launch.

---

## Part 8: What I would explicitly NOT do

- **Don't** build a web UI or hosted dashboard. That lane is owned by Langfuse/Arize/Datadog and saturated.
- **Don't** try to beat LettuceDetect on raw F1. Win on product surface (revisions, hybrid, integrations), not benchmarks.
- **Don't** launch all 4 channels the same day. Stagger.
- **Don't** include emojis in the Show HN post. The pattern is dry-technical.
- **Don't** make it a 5-domain / medical / financial / research mega-benchmark (what `WORLDCLASS_PLAN.md` suggested). Depth in one domain + SOTA benchmarks (RAGTruth, HaluEval) wins over breadth.
- **Don't** delay for a research paper. ArXiv is nice-to-have; a clean Show HN with honest benchmarks is worth 10× more at the hiring stage.

---

## The single biggest risk

**LettuceDetect.** If a commenter surfaces it and you don't already have a benchmark comparison + honest positioning in the README, you lose the thread. Solution: bench against it on day 1, include it in the comparison table, be explicit that it beats you on span-level F1 and that you win on revisions + hybrid + integrations. That's a defensible story. Silence is not.
