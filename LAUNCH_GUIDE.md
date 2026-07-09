# athena-verify — Launch Guide

> The single canonical launch + strategy + marketing doc (2026-07-08). Everything
> in one place: where the product stands, the mid-2026 market, how to position,
> the exact remaining to-do, the full marketing plan, paste-ready copy, and how to
> turn the launch into reputation / funding / O-1A evidence.
>
> Merges and replaces `STRATEGY.md` and the old `LAUNCH.md`. The only other living
> doc is [`PLAN_NEXT.md`](PLAN_NEXT.md) (engineering roadmap).
>
> **Tone rule for everything you post:** sound like a person, not a press release.
> No hype words, no "SOTA", no exclamation marks, no em dashes. HN and Reddit
> punish marketing tone and reward honesty — especially admitting where it loses.

---

## Part 0 — TL;DR and current state

The product is ~90% launch-ready — much further along than the old April plans
assume. Calibration landed (false-positive rate 16.9% → 4.6%), the real RAGTruth
and HaluEval numbers are run and honest, examples run, the GIF and plot ship, 164
tests are green, the build is publishable. A live demo is now built in `demo/`.

**The one hard launch blocker left is PyPI publish** (`pip install athena-verify`
is still 404). Second is deploying the demo. Both are in Part 4 — about 10 minutes
of your time total.

**The strategic shift since the April plans:** the "local, MIT, span-level
detector" lane filled in H1 2026. LettuceDetect escalated (v0.2.2 on 2026-07-05,
generative + CPU models, a July SOTA paper, ~582 stars) and vLLM's HaluGate
(Dec 2025) shipped nearly athena's exact architecture inside the dominant serving
stack. Leading with "local zero-shot NLI detector" now reads as behind-the-curve.

**The defensible move: sell athena as the verification layer on top of any
detector — led by the agent circuit-breaker.** Two independent research passes
converge on the same conclusion: `verify_step()` is the single most differentiated,
least-copyable feature. Competitors flag; almost none halt an agent chain, and the
ones that gate are welded to one serving stack. That is the open lane.

### What's shipped and solid
- Typed `verify()` with async / batch / streaming; `verify_step()` circuit-breaker;
  `verified_completion()`; `suggest_revisions`; `latency_budget_ms` knob; CLI.
- False-positive calibration: anaphora windowing, contradiction-aware rescue,
  numeric gate, check-worthiness filter. Faithful FP rate 16.9% → 4.6% (base);
  synthetic hallucination-catch F1 91.3% → 95.0%.
- Honest real-world benchmarks: RAGTruth QA 0.71 balanced accuracy, HaluEval QA
  0.69, zero-shot, with the losing F1 row (0.47) shown and explained. Reproducible.
- Per-claim `supporting_spans`; LangChain / LlamaIndex / LangGraph / CrewAI
  integrations with runnable (mock-based) examples.
- OpenTelemetry: `verify()` emits a real span (Grafana/Datadog/Jaeger) when
  `ATHENA_OTEL_ENABLED=1` and the `[otel]` extra is installed; safe log fallback.
- Circuit-breaker GIF + measured-improvements plot. ruff + mypy --strict green.

---

## Part 1 — The market in mid-2026 (verified July 2026)

| Project | License | Local | Granularity | Status |
|---|---|---|---|---|
| **LettuceDetect** (KRLabs) | MIT | Yes | Token/span | v0.2.2 (2026-07-05), ~582★. v2 generative (Qwen-2B) + mmBERT + CPU "TinyLettuce"; covers code/tool-output/agentic; SOTA paper (arXiv 2607.00895). Escalating. |
| **vLLM HaluGate** | Open (vLLM) | In-process | Token+span | Dec 2025. Check-worthiness gate → NLI → span merge — nearly athena's exact design, real-time, but welded to the vLLM stack. |
| **Vectara HHEM** | 2.1-Open | Yes | Passage | Mature factual-consistency baseline. Not sentence/span. |
| **Patronus Lynx** | Open (Llama) | Heavy | Answer | Reference open LLM-judge; large, not zero-shot, not span. |
| **Bespoke-MiniCheck-7B** | Llama-3.1 (non-MIT) | Yes | Sentence | Tops LLM-AggreFact ~77%; 7B, restrictive license. |
| **Paladin-mini** | Open | Yes | Claim | New (arXiv 2506.20384), 3.8B Phi-4-mini grounding. |
| **LibreEval 1.0** (Arize) | Open | Yes | Answer | 2026; largest open RAG-hallucination dataset + cheap fine-tuned detector. |
| **Ragas / TruLens / DeepEval** | Open | LLM-judge | Claim | Canonical offline eval; not zero-shot local runtime. |
| **Cleanlab TLM** | Cloud | No | Score | Acqui-hired by Handshake (Jan 2026); standalone future uncertain. |
| **Galileo Luna** | Cloud | No | Platform | Acquired by Cisco (closed 2026-05-22); enterprise-gated now. |

**What changed since the April plans:**
- LettuceDetect went from a beatable peer to a well-resourced moving target with
  academic backing. **Do not compete on detection F1** — a zero-shot NLI model
  loses to a detector fine-tuned on the exact benchmark. That's fine; not our lane.
- A near-clone of athena's architecture shipped in the most-used inference server
  (HaluGate). Validates the design, removes "novel architecture" as a claim, but
  HaluGate is stack-locked — that's our opening.
- Both commercial competitors from the rumor list left the open field
  (Cleanlab→Handshake, Galileo→Cisco). Minor tailwind.

**Demand signal, verified:** LangChain **#33191** ("HallucinationDetector", asking
for a pluggable runtime NLI hallucination module) was **closed as "not planned."**
Real demand + the core framework declining to own it = a validated wedge for a
drop-in library. People are already hand-rolling detect-and-revise loops.

---

## Part 2 — Positioning (the decision that matters most)

> **athena-verify is the runtime verification and self-healing layer for RAG and
> agents — not another detector.** It flags ungrounded sentences with per-claim
> source spans, proposes the corrected sentence, and halts an agent chain the
> moment a step stops being grounded. Zero-shot and provider-neutral by default,
> with a swappable detection backend so a stronger trained detector
> (LettuceDetect, HHEM, MiniCheck) can slot in under the same revision +
> circuit-breaker layer.

Three pillars, in order of defensibility:

1. **Framework-agnostic agent circuit-breaker (the lead).** `verify_step()` returns
   pass/halt against a step's evidence, in any agent loop. LettuceDetect's own July
   paper disclaims agent-trajectory monitoring; HaluGate's gate is welded to vLLM.
   Maps to a named, growing pain: OWASP ASI08 cascading failures; >57% of agent
   errors originate early and cascade.
2. **Detect → revise, not just score.** `suggest_revisions=True` returns the fix.
   Cloud APIs (Azure, Vectara) added correction; local OSS libraries still mostly
   only flag. A supporting differentiator, not the headline.
3. **Backend-agnostic + honest zero-shot detection + per-claim spans.** The
   any-corpus, no-fine-tune, library-not-platform pitch for the LangChain #33191
   crowd. Lead with balanced accuracy (0.71); publish the losing F1 (0.47) with the
   class-imbalance explanation. Honesty is the credibility play.

**Do not claim:** "SOTA detection," "beats LettuceDetect on F1," "novel
architecture," or build a hosted dashboard/SaaS/web UI (saturated, off-thesis).

---

## Part 3 — Who stars this, hooks, and vertical wedges

### Audiences, ranked by star leverage
1. **Local-LLM believer (r/LocalLLaMA)** — number one launch audience.
   Hook: 100% local, no API keys, nothing leaves your box, same on GPT/Claude/Llama/Qwen.
2. **RAG app builder (indie/startup).** Hook: catch hallucinations before your users do, three lines, any model.
3. **AI agent builder.** Hook: `verify_step()` halts a chain the moment a step stops being grounded. Most differentiated, least copyable.
4. **Platform / ML engineer.** Hook: local, provider-neutral, ~25ms, zero API cost, nothing leaves your infra.
5. **Regulated-industry dev (legal/medical/finance).** Hook: per-claim source spans, runs air-gapped.

### Vertical wedges (real, quotable pain — reshape the roadmap, not just copy)
1. **Agent-cascade prevention** (the circuit-breaker's home). OWASP ASI08; >57%
   early-step error cascade; literature explicitly asks for "circuit breaker
   patterns at pipeline checkpoints." Lead vertical.
2. **Legal citation verification.** 1,227+ documented hallucinated-citation cases;
   a Sixth Circuit $30K sanction (March 2026); 30–45% fabricated-citation rates.
   Local execution is mandatory (privilege). A future `citation_mode` is a strong
   commercial-edition wedge.
3. **Voice AI / high-QPS RAG.** Per-utterance verification at <200ms maps to the
   `latency_budget_ms` knob no competitor exposes.

Tier-2 (reachable now): enterprise KBs, financial Q&A, journalism quote-match.
Tier-3 (too heavy for solo OSS): medical/clinical.

---

## Part 4 — Remaining to-do before launch (your ~10 minutes)

Everything else is done and pushed. These two require your accounts.

### 1. Publish to PyPI (the only hard blocker)
- Go to https://pypi.org/manage/account/publishing/ → add a **pending publisher**:
  - Project `athena-verify` · Owner `RahulModugula` · Repo `athena` · Workflow
    `release.yml` · Environment *(leave blank)*
- Trigger the release:
  ```
  git tag v0.1.0 && git push origin v0.1.0
  ```
  `release.yml` builds and publishes automatically. Then verify in a clean venv:
  ```
  python -m venv /tmp/t && /tmp/t/bin/pip install athena-verify
  ```

### 2. Deploy the live demo (highest-leverage traction asset)
- https://huggingface.co/new-space → SDK **Gradio**, hardware **CPU basic**.
- Upload the three files from `demo/`: `app.py`, `requirements.txt`, `README.md`
  (its frontmatter configures the Space and pre-bakes the NLI model).
- It boots, pre-warms, and is live. Paste the Space URL above the fold in the README.

### Repo-readiness checklist (mostly done — confirm before posting)
- [ ] `pip install athena-verify` works cold on macOS **and** Linux (do #1 first).
- [ ] README above the fold: one-line value prop → badges → demo GIF → 2-minute
      copy-paste quickstart → benchmark table → live demo link.
- [x] Live no-signup demo (`demo/`) — deploy it (#2).
- [x] Reproducible benchmarks (`benchmarks/RESULTS.md`).
- [ ] Three or four "good first issue"-labeled issues.
- [ ] Warm up the Reddit account 2–4 weeks out: lurk, upvote, comment without
      links, then post. Never use alt accounts to vote — bannable and auto-detected.

### Optional pre-launch polish (a few days, nice-to-have)
- Add a Colab alongside the Space; Ollama-native judge client.
- A tiny "receipts" image in the README showing per-claim spans output.

---

## Part 5 — Where to post, ranked (with mid-2026 calibrations)

**Set expectations honestly.** A launch is a ~24-hour pulse, not a growth engine.
Median AI-tool HN launch ≈ 120 stars/24h, ~289/7d, and the "Show HN" tag gives no
statistical edge once you control for score/timing/reputation (arXiv 2511.04453).
The repo converts clicks to stars, not the venue. **What wins is one sharp,
shareable hook** — here, the efficiency + circuit-breaker angle: *"CPU-real-time,
zero-shot RAG verification with an agent circuit-breaker — no GPU, no API key."*
Concentrate everything into one **12–17 UTC weekday (Tue–Thu) window** to spike
star velocity; weekend effect is negligible.

### Sequence
Show HN first → maker's comment within 5 min → same-day X thread + r/LocalLLaMA
30–60 min later. Then clear 2–3 hours to answer every reply quickly and graciously
(agree with critics first, then address — the discussion is what keeps you ranked).

### Reddit, in order (space across the week, fresh copy each time, link in body)
1. **r/LocalLLaMA** — lead here (770k, best single sub).
2. **r/Rag** — exact bullseye.
3. **r/AI_Agents** — self-promotion flair; **lead with the circuit-breaker** (most native fit).
4. r/ChatGPTCoding — AI tooling welcome.
5. r/LLMDevs — post as a resource/tools writeup, not a launch.
6. r/Python — Showcase flair; what it is and how it's built.
7. r/MachineLearning — `[P]` flair, benchmark first, zero marketing tone, weekends.
8. r/LangChain — problem-first, framed as detecting hallucinations in a RAG pipeline.
9. r/artificial, r/OpenAI — only if framed around reliability.

**Skip r/programming** — it banned all AI/LLM content (late 2025); a post gets
removed and can hurt your account. Verify each sub's live sidebar rules/flair while
logged in before posting.

### X/Twitter, same morning
Hook tweet → demo GIF → 2–3 value tweets → proof tweet with numbers → repo link +
pip install → soft "star it if useful, feedback welcome." X in 2026 reportedly
rewards author replies and boosts external blog links, so reply diligently and link
the blog post. Tag 3–5 people who actually work in RAG, not celebrities:
@jerryjliu0, @llama_index, @LangChainAI, @rlancemartin, @HamelHusain (evals angle),
@simonw, @_akhaliq.

### DevHunt > Product Hunt
DevHunt (dev-tool-specific weekly board) converts better per visitor for dev tools.
Product Hunt has declined for indie/dev tools ("credibility, not customers") — treat
PH as an optional badge only, don't build the launch around it.

---

## Part 6 — Copy you can paste

Fill in `<LINK>` with the repo URL. Keep the honest limitation lines — they are why
people trust it.

### Show HN title (pick one)
- Show HN: Athena, a local verification layer for RAG and agents with a hallucination circuit-breaker ← recommended
- Show HN: Athena, halt an agent chain the moment a step stops being grounded, runs locally
- Show HN: Athena, local zero-shot RAG verification with per-claim source spans, no API key

### Show HN body
Athena is an open source library that verifies whether an LLM answer is actually grounded in the context you retrieved, sentence by sentence, and then does two things a detector alone doesn't: it hands back the corrected sentence, and it gives you a circuit-breaker that halts an agent chain the moment a step stops being grounded. It runs locally on a small NLI model, so there are no API calls and nothing leaves your machine, and it works with any model provider.

You pass in the question, the answer, and the retrieved chunks. You get back a per-sentence trust score, the list of unsupported sentences, and the exact source span that supported each claim. For agents, verify_step() returns a pass or halt against a step's evidence so a bad fact can't cascade downstream.

The honest version of the benchmarks, since that is the first thing anyone asks. It is zero-shot, so it is not trained on any hallucination dataset. On RAGTruth QA it gets about 0.71 balanced accuracy. LettuceDetect reports higher F1 on RAGTruth, but it is fine-tuned on the RAGTruth training split, so that accuracy is domain specific (and RAGTruth's original span labels are known to under-annotate, so take everyone's RAGTruth F1 with a grain of salt). Athena trades some in-domain accuracy for working on any corpus with no training. On HaluEval QA it lands around 0.69 accuracy, roughly GPT-3.5 to GPT-4 prompting territory, except it is local and about 25ms per check.

What I think is actually new here: it is a library and not a hosted API or a serving-stack plugin, it is provider neutral, and the detection backend is swappable — the revision step and the agent circuit-breaker sit on top, so you can point a stronger trained detector at the same interface. The circuit-breaker for agents is the piece nothing else ships as a drop-in.

Where it loses: heavily paraphrased claims with almost no shared words, and it is not going to beat a model that was fine-tuned on your exact benchmark. Happy to get into the approach or the failure cases.

Repo and install are in the README: <LINK>

### Hacker News first comment (post within five minutes)
I built this because I kept shipping RAG apps that hallucinated with total confidence. Retrieval looked fine, the answer read fine, but one sentence quietly invented a number or flipped a negation and there was no signal until someone caught it.

The approach is deliberately boring so it can run anywhere. Split the answer into sentences, split the context into sentences, and score each answer sentence against the context with an NLI model, taking the best supporting match. The interesting part was cutting false positives: faithful paraphrases often score as neutral even though they are supported, so there is a rescue step that only fires when the claim is not contradicted, every number in it appears in the context, and most of its words are grounded. That got the false positive rate on my synthetic set from about 17 percent down to about 5 percent.

The honest limitation is that a zero-shot approach like this will not top a benchmark that someone fine-tuned a model on. I wrote up the real RAGTruth and HaluEval numbers in the README rather than only the flattering synthetic ones. Would love to hear what breaks it on your data.

### r/LocalLLaMA post
Title: I built a fully local RAG hallucination detector. Zero-shot, no API calls, runs on your own machine.

I have been building RAG apps for a while and the thing that kept burning me was confident hallucinations. Retrieval works, the answer sounds right, but one sentence quietly makes something up and you get no signal until a user catches it.

So I built athena-verify. It is an open source Python library (MIT) that checks an LLM answer against your retrieved context sentence by sentence and tells you which sentences are actually supported. It runs 100 percent locally on a small NLI model. No API keys, nothing leaves your machine, and it behaves the same whether you are on GPT, Claude, Llama, or Qwen.

Three lines:

    from athena_verify import verify
    result = verify(question=q, answer=answer, context=chunks)
    for s in result.unsupported:
        print("unsupported:", s.text)

You also get the exact context span that did or did not support each sentence, so you can show your users the receipts.

Being upfront about where it stands, because this sub can smell hype from a mile away. It is zero-shot, I did not train it on any hallucination benchmark. On RAGTruth QA it gets about 0.71 balanced accuracy. LettuceDetect scores higher there, but it is fine-tuned on RAGTruth's own training set, so that number does not carry over to your data. On HaluEval QA it is around 0.69 accuracy, roughly GPT-3.5 to GPT-4 prompting range, except local and about 25ms. On my own synthetic set it is 95 percent F1 at a 4.6 percent false positive rate, but synthetic numbers are easy so I do not lead with them.

There is also a verify_step() primitive for agents that halts a chain the moment a step stops being grounded, which has saved me from cascading failures in multi-step pipelines.

Repo: <LINK>
Install: pip install athena-verify

Would genuinely love feedback from people running local RAG. Especially curious what breaks it on your data.

### r/Rag post
Title: Tired of RAG answers quietly making things up, so I built a local detector that flags which sentences are not grounded

Same body as the r/LocalLLaMA post, first paragraph tightened to the RAG framing. Lead with the problem, keep the honest numbers, link in the body.

### r/AI_Agents post (use the self-promotion flair)
Title: A circuit breaker for agent hallucinations: verify each step against its evidence and halt before a bad fact cascades

In a multi-step agent, one step's output becomes the next step's input. If a step invents a fact, everything downstream is built on it and you usually find out at the end. I built a small open source library that checks each step against its evidence and returns a pass or halt, so you can stop the chain the moment it stops being grounded.

    from athena_verify import verify_step
    step = verify_step(claim=reasoning_step, evidence=chunks, threshold=0.5)
    if step.action == "halt":
        raise RuntimeError(f"ungrounded step blocked, trust={step.trust_score:.2f}")

It runs locally, no API calls, works with any model. It is zero-shot, so honest numbers: about 0.71 balanced accuracy on RAGTruth QA, around 0.69 on HaluEval QA, roughly GPT-3.5 to GPT-4 prompting range but local and about 25ms. Repo and install in the README: <LINK>. Curious how it does on your agent traces.

### X or Twitter thread
1. I kept shipping RAG apps that hallucinated with total confidence. So I built a verifier that catches it in real time, sentence by sentence, running fully local, and can halt an agent chain the moment a step stops being grounded. Open source, MIT. Here is how it works.

2. You give it the question, the answer, and your retrieved chunks. It scores each answer sentence against the context and flags the ones that are not supported. Three lines, no API keys, nothing leaves your machine.

3. It is provider neutral. GPT, Claude, Llama, Qwen, does not matter. And it returns the exact source span for each claim, so you can show users why a sentence was trusted or not.

4. Honest numbers, because everyone asks. Zero-shot, no training on any hallucination set. RAGTruth QA around 0.71 balanced accuracy, HaluEval QA around 0.69. GPT-3.5 to GPT-4 prompting range, except local and about 25ms.

5. The part I am most excited about is verify_step() for agents. It halts a multi-step chain the moment a step stops being grounded, before a bad fact reaches the final answer.

6. Free and MIT. pip install athena-verify. Repo: <LINK>. Try it on your own data and tell me what breaks.

### LinkedIn
Lead with the founder story in the first two lines, attach a short slide PDF (problem, solution, demo, numbers), put the GitHub link in the first comment. Three hashtags max.

---

## Part 7 — Week-one amplification

Stagger these, write native copy for each, do not mirror-dump.

- **dev.to:** publish to your own blog first, then cross-post with a canonical URL.
  Tags: showdev, python, llm, opensource.
- **Newsletters with real submission doors (verified active):**
  - PyCoder's Weekly — pycoders.com/submissions (best-fit low-friction Python door).
  - Console.dev — email hello@console.dev; meet console.dev/selection-criteria.
  - The Changelog News — changelog.com/news/submit (great for genuine OSS).
  - Latent Space — guest-post form; coverage is warm-intro only, no cold email.
  - Big reach, no open door (be newsworthy on HN/GitHub, don't "submit"): TLDR AI,
    The Rundown AI, AlphaSignal (auto-surfaces trending repos — a strong launch
    auto-features), Ben's Bites (active, pivoted; pitch by relationship).
- **Awesome-lists (submit day one, no star gate):**
  - `Danielskry/Awesome-RAG` — the prominent RAG-tools list.
  - `tensorchord/Awesome-LLMOps` — obs/eval categories fit.
  - Gated for later: `EthicalML/awesome-production-machine-learning` (≥500★);
    `steven2358/awesome-generative-ai` (≥1000★, "Discoveries" tier below that).
  - `kyrolabs/awesome-langchain` auto-closes brand-new repos — submit once there's history.
  - Read each CONTRIBUTING first.
- **Discords/Slacks (help first, then share):** Hugging Face, Ollama, LangChain
  Slack, LlamaIndex.
- Build one runnable example app and submit it to `awesome-llm-apps` (wants apps,
  not links).

---

## Part 8 — What actually drives stars, and demo best practices

- **Hook > polish.** Every verified viral OSS-AI case rode one sharp, shareable
  thing (browser-use on timing, Docling on acute PDF pain, Kokoro-82M on a pure
  benchmark hook). Athena's hook is efficiency + the circuit-breaker.
- **Benchmark honesty is now a credibility asset.** Trust in public benchmarks
  collapsed in 2025–26 (Leaderboard Illusion; Karpathy's "loss of trust"; Stack
  Overflow: only 3% of devs highly trust AI-tool accuracy). Publishing weaknesses
  and false-negative rates *reads as credible*. Add the RAGTruth caveat — its
  original span annotations under-annotate (re-annotation raised flagged spans
  86→865 on one 408-example subset, arXiv 2603.27752) — and you signal expert care.
- **Demo (Gradio Space on CPU Basic, free).** Encoder-class detectors run
  real-time on CPU, so no ZeroGPU/PRO needed. Cold-start: bake weights at build via
  `preload_from_hub` (already in `demo/README.md`) + a warmup inference on boot
  (already in `app.py`). Pre-load one clearly-hallucinated and one clearly-faithful
  example so first visitors see it both catch an error and pass a true statement.
  Use a traffic-light verdict **and** a 0–1 score (never color alone — WCAG).
- **README (empirical):** value prop → hero GIF → one-command quickstart → feature
  table → curated badges. "Assessment" badges (build/coverage/downloads) beat
  vanity badges. Ignore recycled "10-second/90%" stats — no primary source.
- **Don't chase or buy stars.** Star-velocity thresholds are folklore, and a 2026
  RCT found displaying stars has zero causal effect on downloads (see Part 10).

---

## Part 9 — Failure modes that kill a launch

- Over-claiming or unreproducible benchmarks. Drop superlatives, ship a rerunnable
  eval, show a table not a claim.
- Asking friends to upvote or using alt accounts. Bannable on both platforms,
  mostly auto-detected. Announce it and let votes happen.
- Broken `pip install` or a demo behind a waitlist. Test in a clean env, link a
  live no-signup demo.
- Posting to the wrong sub. Skip r/programming. Match each sub's flair and rules.
- Hiding that you are the author. Disclose upfront and lead with substance.
- Pasting identical copy everywhere at once. Stagger and rewrite each time.
- Getting defensive with critics. Agree first, then address. The back-and-forth is
  what ranks you.

---

## Part 10 — Reputation, funding, and O-1A evidence

The goal is not raw stars — it's durable credibility (top-ML-engineer signal, dev
trust, funding and O-1 leverage). That changes what to optimize.

*This is factual research to shape strategy, not legal advice. Before you file
anything, get an immigration attorney. The controlling rules are 8 CFR 214.2(o) and
the USCIS Policy Manual; everything else is secondary.*

### Honest depth beats hype
The blog post ("what I learned building this," including where zero-shot hits a
ceiling and where athena loses) is the highest-value reputation asset. Serious
people reward the honesty a benchmark-gamer can't fake. Keep the losing rows in.

### Build for dependents, not stars (evidence-backed)
A 2026 RCT found displaying star counts has **zero causal effect on downloads**
(arXiv 2603.07919), stars↔downloads correlation is only 0.14–0.47, and ~6M fake
stars are catalogued — so USCIS officers, attorneys, and investors all discount
them. What actually counts:
- **PyPI download percentiles** vs comparable libraries (frame as a ranking, never
  raw counts).
- A **populated GitHub/PyPI dependents graph** — get real projects to
  `import athena_verify`.
- **Named-enterprise production letters.**
- Getting **ranked #1 in a defined set** (top of a RAG-hallucination-detection
  awesome-list or a public leaderboard). This "defined-set" move is exactly what
  made Alexey Inkin's EB-1A case work (he lobbied FlutterGems to create a category
  his project could top). Copy the defined-set + downstream-letters move, not just
  the download stats. Instrument all of this from day one.

### How the O-1A actually works
One giant internationally recognized award, or document at least three of eight
criteria — but three is necessary, not sufficient. USCIS then does a discretionary
"final merits determination" (are you in the small percentage at the very top with
sustained acclaim). Strong tech cases get denied at this second step, so aim to
cleanly document **four or more** criteria and prove impact, not existence.

The eight: nationally recognized awards; membership requiring outstanding
achievement; published material about you in major/trade media; judging others'
work; original contributions of major significance; authorship of scholarly
articles; a critical role for a distinguished organization; a high salary.

### The Jan 2025 update (get this right)
USCIS policy update PA-2025-02 (effective Jan 8 2025) amended the **O-1A** guidance
(Policy Manual Vol 2, Part M, Ch 4): added evidentiary examples for critical and
emerging technologies including AI, and reaffirmed the comparable-evidence
mechanism. Two honest caveats — getting this wrong in a petition is worse than not
citing it: (1) it is **O-1A only** — it did not change EB-1A (Vol 6, Part F, Ch 2),
so don't cite it in an EB-1A filing; (2) the USCIS text does **not** literally say
"GitHub," "open source," or "stars" — "a software repository with real-world impact
= an original contribution" is how attorneys apply comparable evidence, not a
verbatim quote. Frame it as comparable evidence of impact and verify current
wording against uscis.gov/policy-manual (RFEs rose through 2026; the framework may
tighten).

### Where athena fits, criterion by criterion
- **Original contributions of major significance** — athena's home. See "build for
  dependents" above; adoption evidence + independent expert letters citing hard
  numbers, not stars.
- **Authorship of scholarly articles** — an arXiv preprint alone is weak (no peer
  review). Aim for a peer-reviewed workshop/conference paper (ACL/NeurIPS/ICML
  eval or RAG workshop). A benchmark study on honest zero-shot hallucination
  detection is a real paper. Post arXiv now for citations, then get the reviewed
  version.
- **Judging the work of others** — highest leverage, lowest barrier, start now.
  Accept review invitations, join a program committee, judge a recognized hackathon.
  Keep the invite + a letter with counts. Reviewing PRs on your own repo doesn't
  count.
- **High salary** — pure paperwork. Comp docs + a comparison report (BLS, OFLC wage
  library, Levels.fyi) showing top 5–10% for your role and region.
- **Press about you** — IEEE Spectrum, The New Stack, VentureBeat, InfoQ,
  TechCrunch about you and your work (not paid, not just your employer). A strong
  launch is the natural trigger.
- **Critical role** — own a core system at a well-known/VC-backed org, or maintain
  a widely-depended-on OSS project.

### The 12–18 month plan
Target a clean four-criteria case: original contributions, authorship, judging,
high salary — with press and critical role as upgrades.
- **Months 0–2:** salary comparison; start the OSS impact dossier (downloads,
  dependents, adoption, commit logs proving authorship); post the arXiv paper;
  begin building relationships for expert letters.
- **Months 1–6:** land two or three judging/review roles; submit the reviewed
  workshop paper.
- **Months 3–12:** pursue launch-tied press; document your critical role; apply for
  a competitive grant or award.
- **Months 9–18:** finalize five to eight expert letters (majority from independent
  experts who never worked with you); secure a conference talk; file once four
  criteria are cleanly documented.

### The one effort that serves all three goals
The **opt-in trained backend + benchmark paper** (from `PLAN_NEXT.md`) is the single
strongest item: a citable paper (authorship), a real adoption story (original
contribution), and a press/talk hook at once. It closes the in-domain F1 gap for
people who want it while keeping zero-shot the default. Ship it as a "v0.2, now with
a trained backend and a paper" second wave. Maintainer-grade responsiveness in the
launch window and first month matters too — recruiters and investors read the issues
tab, not just the star graph.

---

## Part 11 — Targets (recalibrated to mid-2026 reality)

- **First 48 hours:** front page of Show HN, top of r/LocalLLaMA for the day. A
  realistic star band is ~120–300 if the repo is ready and the hook lands; treat
  anything above as upside, and don't optimize for the number — optimize for the
  first `pip install` working and for first dependents.
- **First two weeks:** into `Awesome-RAG` and `Awesome-LLMOps`, one newsletter
  mention, and — more important than stars — the first two or three projects that
  `import athena_verify`.
- **First quarter:** the trained backend + arXiv note shipped as a second wave, one
  talk submitted, first adoption by a named project, PyPI downloads trending in a
  reportable percentile.
