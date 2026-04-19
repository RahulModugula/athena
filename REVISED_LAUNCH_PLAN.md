# The Athena-Verify Revised Plan (April 17, 2026)

> Supersedes `LAUNCH_PLAN.md`. Driven by April 2026 research: competitive landscape scan, use-case market research, and full codebase audit (30 findings).
>
> **Headline pivot: your `suggest_revisions` moat is dead.** Azure AI Content Safety shipped "Groundedness Detection + Correction" with GPT-4o-backed auto-rewrite; Vectara shipped a standalone "Hallucination Corrector" with an open benchmark; the HalluClean paper (arXiv 2511.08916) shipped a plan-execute-revise framework. Revision is now commodity. You need a new headline feature.

---

## Part 0 — The revised thesis

### Old positioning (Feb 2026 plan)
> *"MIT-licensed sentence-level runtime hybrid NLI + local-LLM-judge verifier with built-in revision suggestions."*

### New positioning (April 2026, defensible)
> *"The local-first, provider-neutral, latency-bounded, sentence-level verification **library** — not an API, not a platform. Runs fully offline on a MacBook, works identically on GPT, Claude, Gemini, Llama, or Qwen, with sub-100ms p95 per sentence and per-claim source-span citations."*

Four words together no competitor ships as a library in April 2026:

1. **Local-first** — Azure, Vertex, Anthropic Citations, Patronus, Cleanlab, Galileo are all cloud/API.
2. **Provider-neutral** — Azure Groundedness only works with GPT-4o; Vertex grounding only with Gemini; Anthropic Citations only with Claude.
3. **Latency-bounded** — no open-source verifier exposes a hard `latency_budget_ms` knob; LatentAudit (arXiv 2604.05358) proved this is possible, no one has packaged it.
4. **Per-claim source-span citations** — Vectara and Azure return corrections but not per-claim span offsets into the original source. This replaces `suggest_revisions` as the wow feature.

Everything downstream reflects this pivot.

---

## Part 1 — Competitive landscape update (Feb → April 2026)

| Change | Impact | Evidence |
|---|---|---|
| Azure shipped Groundedness + Correction API | **Kills revision moat** for Azure-bound users | [learn.microsoft.com groundedness](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness) |
| Vectara shipped Hallucination Corrector + open benchmark | **Kills revision moat** + forces comparison | [vectara.com/blog/vectaras-hallucination-corrector](https://www.vectara.com/blog/vectaras-hallucination-corrector) |
| HalluClean (arXiv 2511.08916) academic plan-execute-revise | Direct academic analog of your revision layer | arxiv.org/html/2511.08916v5 |
| LatentAudit (arXiv 2604.05358) white-box residual-stream monitoring | Opens the "latency-bounded" position | arxiv.org/abs/2604.05358 |
| Cisco announced intent to acquire Galileo (April 9 2026) | Luna line goes behind Splunk/enterprise wall | [blogs.cisco.com acquisition](https://blogs.cisco.com/news/cisco-announces-the-intent-to-acquire-galileo) |
| Anthropic Citations API GA (on Anthropic + Vertex) | Erodes "need external verifier" pain for Claude users | [claude.com/blog/introducing-citations-api](https://claude.com/blog/introducing-citations-api) |
| LettuceDetect — **still no v2 shipped** | Good news: still beatable on integrations + UX | — |
| HHEM 2.3 (internal), HHEM-2.1-Open still public | No new public weights to beat | [vectara.com next-gen leaderboard](https://www.vectara.com/blog/introducing-the-next-generation-of-vectaras-hallucination-leaderboard) |

**New benchmarks to run alongside RAGTruth + HaluEval:**
- **FaithBench** (arXiv 2410.13210) — now a standard.
- **HalluLens** (arXiv 2504.17550) — 2025 broad benchmark.
- **RAGBench** — referenced in competitor literature.

---

## Part 2 — Target markets: beyond RAG developers

Research identified 10+ segments with documented pain. Three of them genuinely reshape the product, not just marketing copy.

### Tier 1 — Reshape the product

**1. Legal tech (citation verification).** Court-submissions database logs **1,227 documented cases of hallucinated citations** globally; Sixth Circuit sanctioned two attorneys $30K in March 2026; Stanford CodeX measures 30–45% fabricated-citation rates on general LLMs. 79% of lawyers use AI tools (ABA TechReport 2025). BriefCatch shipped "RealityCheck" at Legalweek 2026 for this exact reason. **What it requires:** a `citation_mode` — given `Smith v. Jones, 123 F.3d 456`, fetch the opinion, verify the quoted passage is actually present. Bluebook/pinpoint parsing. Local execution is a hard requirement (attorney-client privilege).

**2. AI-agent cascade prevention.** OWASP ASI08 names cascading failures as a top-10 risk; deep-research agent studies show **>57% of errors occur in early steps and cascade**; literature specifically calls for "confidence scoring at pipeline checkpoints" and "circuit breaker patterns" — which is exactly what sentence-level runtime verification is. **What it requires:** a `verify_step(claim, evidence) -> pass | fail | halt` primitive that is a circuit-breaker (not an eval), LangGraph/CrewAI integrations as first-class, and a fast-path that skips the LLM judge for inline per-step use.

**3. Coding agents / slopsquatting.** LLMs hallucinate package names **18–21%** of the time; **200,000+ fake packages** catalogued in code-LLM outputs; **43% of hallucinated packages repeat across 10 queries** — attackers pre-register them. Active in-the-wild case January 2026: `react-codeshift` npm conflation. This is a **security-budget** problem, not an eval-budget problem. **What it requires:** a code-aware mode — AST/import extraction, npm/PyPI/crates registry validation, API-signature check against typed docs. Arguably a standalone sub-product; flag for consideration.

### Tier 2 — Reachable with current product + marketing

**4. Enterprise internal knowledge bases** (Glean/Notion AI-alternatives on private data). Sentence-level citations + local hosting are direct product fits.
**5. Voice AI / phone agents** (Retell, Vapi, Bland). Per-utterance verification at <200ms maps to turn-by-turn voice. Requires the new latency-bounded mode.
**6. Financial research / earnings Q&A.** FINRA 2026 guidance mandates audit trails; **4 of 6 leading models fabricate financial data** on incomplete docs. Needs numeric-verification improvements (NLI is weak on numbers).
**7. News / journalism AI.** BBC/EBU tested four major chatbots — **>50% of answers had significant issues, 13% of quotes fabricated**. Quote-match mode (fuzzy string match layered on NLI) would fit.

### Tier 3 — Noted, not priority

**8. Medical/clinical AI** — 64.1% hallucination rate in summaries, but slow sales cycle, needs medical-NLI fine-tune and FDA docs. Good commercial-edition play, too heavy for OSS solo.
**9. Academic research assistants, government, insurance, customer-support chatbots** — real pain but either crowded or low willingness-to-pay.

### Synthesis
Move positioning from "eval tool for RAG devs" to **"runtime verification library for high-stakes answer/agent loops"** with three concrete vertical hooks (legal citations, agent circuit-breaker, voice latency). Keep the generic RAG-dev top-of-funnel, but don't compete there head-on against LettuceDetect on raw F1.

---

## Part 3 — Codebase audit findings (30, grouped)

Full per-finding detail in the audit; summary here. **Fix the 7 blockers before launch.**

### Blockers (must fix before anyone runs the code)

| # | Location | Issue |
|---|---|---|
| B1 | `nli.py:66-73` | Softmax hardcodes `exp_vals[0]` assuming 3-class output + entailment is class 0. Crashes on model variance. |
| B2 | `core.py:91` | All context chunks concatenated into one NLI premise; silent truncation at 512 tokens on realistic RAG contexts. |
| B3 | `parser.py:76` | Regex sentence split breaks on "Dr. Smith" / "U.S. policy" — common legal/medical failure mode. |
| B4 | `examples/*_example.py` | Both LangChain and LlamaIndex examples are comment-only stubs. `pip install && python examples/langchain_example.py` fails. |
| B5 | `benchmarks/results/` | Only synthetic `full_eval.json` committed. README cites 81%; JSON shows 52.9%. RAGTruth/HaluEval scripts exist but unrun. |
| B6 | `pyproject.toml` | Not published to PyPI; no release tags; no GH Actions. `pip install athena-verify` fails. |
| B7 | `models.py:63` vs `core.py` | `supporting_spans` field defined but never populated. Advertised feature missing. |

### Major (ship soon after launch)

- `llm_judge.py:136` — no retry/backoff/circuit-breaker on LLM API timeouts.
- `core.py verify_batch_async` — one API failure fails the whole batch (no per-item error isolation).
- `integrations/langchain.py:144` — `__getattr__` passthrough breaks IDE type checking.
- `integrations/__init__.py` — empty; forces `from athena_verify.integrations.langchain import …` instead of top-level.
- `models.py:145-184` — `.to_otel_span()` / `.to_langfuse_trace()` exist but `verify()` never calls them.
- No structured logging of verification decisions, no per-sentence latency histograms, no NLI model caching across calls.
- No CLI entry point (`athena-verify verify --answer … --context …`).
- No end-to-end tests with real models; all tests mock NLI.
- No security/privacy documentation (context is sent to Anthropic/OpenAI if LLM judge enabled).
- No tuning guide for `trust_threshold`.
- `suggest_revisions` feature is off by default and untested — but **per the pivot, demote this, don't promote**.
- `calibration.py:54` weight redistribution when LLM judge absent is silent — add `num_signals` to `SentenceScore` for auditability.
- `parser.py:88` NLTK fallback exists but undocumented/unused.

### Nice-to-have
- `LLMClient` not in `__all__`; `verify_stream` yields incomplete intermediate objects; no property tests for `[0.0, 1.0]` trust-score invariant; no adversarial tests (long contexts, CJK, degenerate sentences); no model-selection guide.

---

## Part 4 — The revised 3 fundamental product improvements

The prior plan's three were: (1) benchmarks, (2) `suggest_revisions` headline, (3) `VerifyingLLM` retry loop.

**New list, reflecting April 2026 reality:**

### Improvement #1 — Ship real-world benchmarks (unchanged, still urgent)
Run `run_ragtruth.py`, `run_halueval.py`, add FaithBench + HalluLens. Commit JSON. Update README with honest table vs LettuceDetect 79.2%, HHEM-2.1-Open, Ragas faithfulness, GPT-4 judge. **Include a row you lose.**

**Effort:** 2–3 days.

### Improvement #2 — Per-claim source-span citations (**replaces** `suggest_revisions` as headline)

The `supporting_spans` field (`models.py:63`) is defined but empty. Ship it. The demo becomes:

```python
result = verify(question=..., answer=..., context=chunks)

for s in result.sentences:
    print(f"{s.text}")
    for span in s.supporting_spans:
        print(f"  ← chunk[{span.chunk_idx}] offset {span.start}-{span.end}: {span.text!r}")
```

This directly addresses the universal HN complaint: *"Ragas tells me faithfulness=0.87 but not which sentence is lying or which source supports it."* Azure and Vectara ship corrections; neither ships span-level source citations in the open-source library space. **This is the new defensible demo GIF.**

**Effort:** 1 day for basic chunk-level + sentence-level mapping; 2–3 days for precise character-offset spans.

### Improvement #3 — Latency-bounded mode with hard p95 budget (new)

First-class knob no competitor has:

```python
result = verify(..., latency_budget_ms=50)     # pure NLI + lexical, skip LLM judge
result = verify(..., latency_budget_ms=500)    # allow borderline LLM escalation
result = verify(..., latency_budget_ms=None)   # always escalate borderline
```

Positions athena-verify as *the* library for voice AI, agent circuit-breakers, and high-QPS RAG — three of the top four segments identified in market research. LatentAudit (April 2026) proved this is technically possible; no one has shipped it as a library knob.

**Effort:** 1–2 days (the mechanism is an early-exit in `core.py` around the LLM-judge escalation).

### Improvement #4 — `VerifyingLLM` self-healing loop (kept from prior plan)

LangChain issue #33191 still open. If verification fails, re-retrieve with unsupported claims as additional queries and retry. Quote the issue in the PR.

**Effort:** 2–3 days.

---

## Part 5 — Stretch bets (pick 2, not 4)

In priority order; pick 2 based on time:

- **Legal citation-mode** — Bluebook parsing + quoted-passage verification against case text. Ships the legal-tech wedge. 3–5 days.
- **`verify_step()` primitive + LangGraph/CrewAI integration** — ships the agent circuit-breaker wedge. 2–3 days.
- **HuggingFace Space public demo** — free, doesn't go down. Half a day.
- **Terminal GIF (non-negotiable)** — 2 hours.
- **Ollama-native client** — first-class instead of LM-Studio-only. 30 min + testing.
- **OTel/Langfuse auto-instrumentation** — wire the existing `.to_otel_span()` into `verify()`. 2–3 hours.

**Don't do:** hosted dashboard, web UI, SaaS, medical-NLI fine-tune, research paper.

---

## Part 6 — Revised 3-week execution

### Week 1 — Blockers + the pivot

- Day 1: Fix blockers B1 (NLI softmax), B2 (per-chunk NLI), B3 (NLTK sentence split default).
- Day 2–3: Run RAGTruth + HaluEval + FaithBench. Commit results. Rewrite README table honestly.
- Day 4: Ship `supporting_spans` (Improvement #2). This is the new headline.
- Day 5: Rewrite `langchain_example.py` + `llamaindex_example.py` as runnable scripts. Add `requirements-examples.txt`. Publish to PyPI. Tag v0.1.0. GH Actions CI.

### Week 2 — Differentiation

- Day 6: Ship `latency_budget_ms` mode (Improvement #3). This is the voice AI / agent wedge.
- Day 7–8: Ship `VerifyingLLM` self-healing loop (Improvement #4).
- Day 9: Wire OTel/Langfuse auto-instrumentation + structured decision logs.
- Day 10: Pick 1 of two stretch bets — **citation-mode (legal wedge)** or **verify_step() (agent wedge)**. Default recommendation: `verify_step()` — smaller scope, bigger category.

### Week 3 — Polish + launch

- Day 11: Terminal GIF. Above-the-fold in README. HuggingFace Space demo (optional).
- Day 12: Technical blog post (*"how we built a local RAG verifier — what we learned"*, not a product post — reads as hiring signal, like Instructor/Marker).
- Day 13: Open 3 "good first issue" labels. Pin repo on GH profile. Update X/LinkedIn bios.
- Day 14: Fresh-venv dress rehearsal. Fix whatever breaks.
- Day 15 (Tuesday): Show HN 8:30am ET + r/LocalLLaMA 6pm ET.
- Day 16: r/MachineLearning `[P]`.
- Day 17: Blog post cross-post.

---

## Part 7 — Revised Show HN body (reflecting new moat)

> Athena-verify is an open-source Python library that catches RAG hallucinations at runtime, sentence-by-sentence, **entirely offline**, in **any LLM stack**. Three lines of code, no API key, no telemetry.
>
> It scores each sentence with DeBERTa-v3 NLI + lexical overlap (~20ms) and optionally escalates borderline cases to a local LLM judge via LM Studio or Ollama. Exposes a `latency_budget_ms` knob — the first open-source verifier that does.
>
> Each result includes **per-claim source spans**: not just *"this sentence failed"*, but *"this source sentence, offset 412–537, supports that claim."* On RAGTruth QA it hits [X]% F1; on FaithBench [Y]%. LettuceDetect still beats it on span-level F1; athena wins on latency bounds, provider-neutrality, and spans-in-library.
>
> Every hyperscaler detector today is locked to one stack (Azure→GPT-4o, Vertex→Gemini, Anthropic Citations→Claude, Vectara Corrector→Vectara). This one isn't.
>
> `pip install athena-verify`
> https://github.com/RahulModugula/athena

---

## Part 8 — What to explicitly NOT do

- **Don't** lead with `suggest_revisions`. Azure + Vectara + HalluClean already ship it. Demote to "also available."
- **Don't** try to beat LettuceDetect on raw span F1. Win on latency bounds + spans-in-library + integrations.
- **Don't** build a medical-NLI fine-tune or legal ontology now. Flag as commercial-edition.
- **Don't** build a web UI, hosted dashboard, or SaaS. Langfuse/Arize/Datadog own that lane.
- **Don't** launch 4 channels the same day. Stagger.
- **Don't** wait for the arXiv paper. Ship the code and blog post.
- **Don't** write the post with emojis.

---

## Part 9 — Single biggest risk

**Hyperscaler bundling.** Azure Groundedness + Correction is free with Azure OpenAI; Vertex grounding is free with Gemini. For the ~60% of enterprise RAG that sits on Azure or Vertex, they'll reach for the bundled option first. Your mitigation: be the library that works **everywhere else** (OpenAI direct, Anthropic direct, local Llama, Mistral, Qwen, Ollama) — which is the majority of serious RAG developers on HN and r/LocalLLaMA. Don't position as a hyperscaler replacement; position as the library the hyperscaler detectors **can't** reach.

---

## Appendix — Ordered codebase task list (for execution)

1. **B1** — Fix NLI softmax (`nli.py:66-73`) — 1h
2. **B3** — NLTK sentence split default (`parser.py:76`) — 2h
3. **B2** — Per-chunk NLI (`core.py:91`) — 4h
4. **B7** — `supporting_spans` population (`core.py`, `models.py:63`) — 1 day
5. **B4** — Runnable LangChain + LlamaIndex examples — 1 day
6. **B5** — Run RAGTruth + HaluEval + FaithBench; commit JSON; reconcile README — 2–3 days
7. **B6** — PyPI publish + CLI + GH Actions CI — 1 day
8. **Improvement #3** — `latency_budget_ms` mode — 1–2 days
9. **Improvement #4** — `VerifyingLLM` self-healing loop — 2–3 days
10. **OTel/Langfuse auto-instrumentation** + structured logs — 3–4h
11. Retry/backoff on LLM judge — 3–4h
12. Per-item error isolation in `verify_batch_async` — 2h
13. Export `LLMClient` + `integrations` top-level — 30m
14. Security/privacy doc + tuning guide + model-selection guide — 4–6h
15. Adversarial + property tests — 4–5h

**Stretch (pick 1–2):**
16. `verify_step()` + LangGraph/CrewAI integration — 2–3 days
17. Legal citation-mode (Bluebook + quote-in-opinion) — 3–5 days
18. HuggingFace Space demo — half day
19. Ollama-native first-class client — 1h







---                                                                                                                                
  Session 1 — B7 + Improvement #2: populate supporting_spans (the new headline)                                                   
                                                                                                                                     
  Populate the `supporting_spans` field on `SentenceScore`. Right now it's                                                           
  declared in athena_verify/models.py:63 but never set in athena_verify/core.py,                                                     
  so the advertised "per-claim source spans" feature is missing.                                                                     
                                                                                                                                     
  For each sentence, record the chunk(s) that supported it: chunk_idx,                                                               
  character offsets (start, end) into the original chunk text, and the                                                               
  supporting substring. Use the per-unit NLI scores already computed in                                                              
  core.py around line 91 — pick units above an entailment threshold (start                                                           
  with 0.5) and map each unit back to its parent chunk's char offsets.                                                               
                                                                                                                                     
  Update models.py to type the field as a list[SupportingSpan] dataclass                                                             
  instead of list[dict] so it shows up properly in IDE/JSON output.                                                                  
                                                                                                                                     
  Add a test in tests/ that calls verify() on a 2-chunk context and asserts                                                          
  each supported sentence has at least one span with valid offsets that                                                              
  slice back to the substring.                                                                                                       
      
  ---                                                                                                                                
  Session 2 — Improvement #3: latency_budget_ms knob
                                                                                                                                     
  Add a `latency_budget_ms: int | None = None` parameter to verify() and
  verify_async() in athena_verify/core.py.                                                                                           
      
  Behavior:                                                                                                                          
    - None (default): current behavior — escalate borderline to LLM judge.
    - <= 100: skip LLM judge entirely, return NLI+lexical only.                                                                      
    - >100: track elapsed time; only escalate borderline sentences if                                                                
      remaining budget covers an LLM call (use a rolling avg from                                                                    
      llm_judge_avg_ms in metadata, or a 2000ms conservative default).                                                               
                                                                                                                                     
  Record per-sentence elapsed_ms and a `budget_exceeded: bool` in                                                                    
  VerificationResult.metadata. Add tests that assert latency_budget_ms=50                                                            
  never invokes the LLM client (mock and assert call_count == 0).                                                                    
                                                                                                                                     
  ---                                                                                                                                
  Session 3 — B5: real benchmarks + honest README                                                                                    
      
  Run benchmarks/run_ragtruth.py and benchmarks/run_halueval.py end-to-end
  on the actual datasets (not synthetic). Commit the JSON outputs to                                                                 
  benchmarks/results/. Add benchmarks/run_faithbench.py following the same                                                           
  pattern. Run it and commit results.                                                                                                
                                                                                                                                     
  Then rewrite the benchmark table in README.md with real numbers vs                                                                 
  LettuceDetect 79.2% F1, HHEM-2.1-Open, Ragas faithfulness, GPT-4 judge.                                                            
  Include at least one row where athena loses — the plan explicitly                                                                  
  requires this for credibility.                                                                                                     
                                                                                                                                     
  If a dataset requires download/auth, document the steps in                                                                         
  benchmarks/RESULTS.md, but still run what you can locally.                                                                         
                                                                                                                                     
  --- 
  Session 4 — B3 + B4: sentence splitting + runnable examples                                                                        
                                                                                                                                     
  Two fixes:
                                                                                                                                     
  1. In athena_verify/parser.py, make split_sentences() use NLTK Punkt by                                                            
     default (it currently uses a regex that breaks on "Dr. Smith" /
     "U.S."). Keep the regex as a fallback when NLTK isn't installed.                                                                
     Add nltk to pyproject.toml dependencies (or as an optional extra                                                                
     with auto-fallback). Add tests for "Dr. Smith said hello. The U.S.                                                              
     policy is X." that assert 2 sentences, not 4.                                                                                   
                                                                                                                                     
  2. Rewrite examples/langchain_example.py and examples/llamaindex_example.py                                                        
     so `python examples/langchain_example.py` actually runs end-to-end                                                              
     with a tiny in-memory retriever and a mocked or real LLM (use                                                                   
     FakeListLLM from langchain_core for langchain; MockLLM for                                                                      
     llamaindex). Add examples/requirements.txt.                                                                                     
                                                                                                                                     
  ---                                                                                                                                
  Session 5 — B6: PyPI + CLI + release
                                                                                                                                     
  Three things:
                                                                                                                                     
  1. Add an `athena-verify` CLI entry point in pyproject.toml:                                                                       
       athena-verify verify --answer "..." --context file.txt [--json]
     Implement in athena_verify/cli.py using argparse. Output a colored                                                              
     sentence-by-sentence trust score table by default; --json for                                                                   
     machine-readable.                                                                                                               
                                                                                                                                     
  2. Make sure .github/workflows/ci.yml runs tests on push and that                                                                  
     there's a release workflow that publishes to PyPI on git tag v*.
     Use trusted publishing (OIDC), not API tokens. Don't add tokens.                                                                
                                                                                                                                     
  3. Bump version to 0.1.0 in pyproject.toml. DO NOT tag or publish — I'll                                                           
     do that manually. Just leave it ready.                                                                                          
                                                                                                                                     
  --- 
  Session 6 — Improvement #4: VerifyingLLM self-healing loop                                                                         
      
  Extend athena_verify/integrations/langchain.py VerifyingLLM with a
  retry/re-retrieve loop:                                                                                                            
                                                                                                                                     
    VerifyingLLM(llm, retriever=retriever, max_retries=2,                                                                            
                 on_unsupported="re-retrieve")                                                                                       
                                                                                                                                     
  When verification fails and on_unsupported="re-retrieve":
    - Take each unsupported sentence's text as a new query                                                                           
    - Re-retrieve via the passed retriever                                                                                           
    - Append the new chunks to context, regenerate the answer                                                                        
    - Re-verify, up to max_retries                                                                                                   
                                                                                                                                     
  Reference LangChain issue #33191 in the docstring and in                                                                           
  docs/architecture.md. Add tests with a mock retriever that returns
  better chunks on the second call and assert the loop terminates with                                                               
  a passing verification.                                                                                                            
                                                                                                                                     
  ---                                                                                                                                
  Session 7 — polish bundle                                                                                                          
      
  Five smaller fixes, one PR:
                                                                                                                                     
  1. athena_verify/llm_judge.py:136 — add tenacity-style retry with                                                                  
     exponential backoff (3 attempts, 1s/2s/4s) on httpx errors and                                                                  
     timeouts. Wrap with a circuit breaker that opens after 5                                                                        
     consecutive failures for 30s.                                                                                                   
                                                                                                                                     
  2. athena_verify/core.py verify_batch_async — wrap each item in                                                                    
     try/except so one failure doesn't fail the batch; return per-item
     error in that item's metadata.                                                                                                  
                                                                                                                                     
  3. athena_verify/__init__.py — export LLMClient at the top level. Make                                                             
     `from athena_verify.integrations import langchain, llamaindex` work                                                             
     (currently athena_verify/integrations/__init__.py is empty).                                                                    
                                                                                                                                     
  4. Wire the existing models.py to_otel_span() / to_langfuse_trace()                                                                
     methods — call them from verify() when env vars                                                                                 
     ATHENA_OTEL_ENABLED=1 / ATHENA_LANGFUSE_ENABLED=1 are set.                                                                      
                                                                                                                                     
  5. Cache the NLI model load in nli.py with functools.lru_cache keyed on                                                            
     model_name so repeated verify() calls don't re-load the model.                                                                  
                                                                                                                                     
  --- 
  Session 8 — docs                                                                                                                   
      
  Add three short docs to docs/:
    - security.md: what data leaves the machine when LLM judge is on                                                                 
      (Anthropic / OpenAI / local), how to stay fully offline.                                                                       
    - tuning.md: how to pick trust_threshold for different domains                                                                   
      (legal=0.85, support=0.65, etc) with reasoning.                                                                                
    - models.md: NLI model trade-offs (deberta-v3-large vs base vs                                                                   
      lightweight), latency vs F1 table from the benchmarks.                                                                         
                                                                                                                                     
  Link all three from README. Each doc <300 words.                                                                                   
                                                                                                                                     
  ---                                                                                                                                
  Session 9 — stretch (pick ONE before launch)
                                                                                                                                     
  Plan recommends verify_step():
                                                                                                                                     
  Add a verify_step() primitive in athena_verify/core.py:
                                                                                                                                     
    result = verify_step(claim: str, evidence: str | list[str],                                                                      
                         threshold: float = 0.7) -> StepResult                                                                       
    # StepResult: passed: bool, trust_score: float, action: "continue"|"halt"                                                        
                                                                                                                                     
  Build a LangGraph node wrapper in
  athena_verify/integrations/langgraph.py that fits as a graph node                                                                  
  returning either the original state or a halt signal. Add a CrewAI                                                                 
  tool wrapper in athena_verify/integrations/crewai.py.                                                                              
                                                                                                                                     
  Add examples/agent_circuit_breaker.py showing a 3-step LangGraph agent                                                             
  where verify_step halts on a fabricated intermediate claim.                                                                        
                                                                                                                                     
  --- 
  Don't bother prompting — do these manually                                                                                         
                                                                                                                                     
  - Terminal GIF (record yourself, don't ask Claude)
  - HuggingFace Space (one-off web setup)                                                                                            
  - Pinning repo / bio updates                                                                                                       
  - Picking the launch Tuesday                                                                                                       
                                                                                                                                     
  Suggested order if you want to ship in one week: 1 → 2 → 3 → 5 → 4 → 6 → 7. Sessions 8 and 9 are nice-to-have but not blocking.   


  All 7 blockers fixed:                  
  - B1 (NLI softmax): fixed — now uses   
  exp_vals[1] (entailment class 1)       
  correctly                             
  - B2 (per-chunk NLI): fixed — loops per
   chunk in core.py:146                  
  - B3 (NLTK parser): fixed — NLTK punkt 
  as default                             
  - B4 (examples): langchain_example.py, 
  llamaindex_example.py, requirements.txt
   all present                           
  - B6 (CLI/GH Actions/v0.1.0): cli.py,  
  ci.yml/release.yml, version set — not  
  published per plan                     
  - B7 (supporting_spans): fully         
  populated and typed                    
                                         
  Sessions 1, 2, 4, 5, 6, 7, 8, 9 all 
  shipped (verify_step,                  
  latency_budget_ms, VerifyingLLM,
  polish, docs, etc.)                    
      
  ---
  Three things still incomplete before 
  you push the button:                   
                      
  1. B5 (benchmarks) — half done.        
  faithbench_50.json exists but ragtruth 
  and halueval have no real-data results.
   The README comparison table has no    
  FaithBench score vs LettuceDetect and
  no row where athena loses — which the
  plan explicitly requires for
  credibility.
  2. tests/test_supporting_spans.py is 
  untracked — it was written but never   
  committed.
  3. 9 commits not pushed to origin — the
   code is locally ahead, remote is      
  stale. Push before any launch.
                                         
  Manual items still on you (per plan's  
  "don't ask Claude"):
  - Terminal GIF                         
  - HuggingFace Space
  - Pinning repo / bio updates
  - git tag v0.1.0 + pip publish
  - Pick your launch Tuesday             
                                         
  The one real blocker is B5 — the       
  RAGTruth/HaluEval scripts exist but the
   datasets need download/auth. Run what
  you can, document the rest in          
  benchmarks/RESULTS.md, and update the
  README comparison table with honest
  numbers before launching.