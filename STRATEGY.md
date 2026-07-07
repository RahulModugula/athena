# athena-verify — Strategy

> The single canonical strategy doc (2026-07-08). Supersedes `WORLDCLASS_PLAN.md`,
> `LAUNCH_PLAN.md`, and `REVISED_LAUNCH_PLAN.md`.
>
> Two living companions: [`LAUNCH.md`](LAUNCH.md) is the launch-day copy, channel
> playbook, and O-1 evidence plan; [`PLAN_NEXT.md`](PLAN_NEXT.md) is the technical
> roadmap. This file is the layer above both: where the product stands, what the
> market actually looks like in mid-2026, how we position, and what to do in what
> order.

---

## TL;DR

The product is ~90% launch-ready — much further along than the April plans assume.
The calibration work landed (false-positive rate 16.9% → 4.6%), the real RAGTruth
and HaluEval numbers are run and honest, examples run, the GIF and plot ship, 162
tests are green. **The one hard launch blocker left is PyPI publish** (currently
404); the highest-leverage missing asset is a **live no-signup demo** (now built
in `demo/`).

The strategic shift since the April plans: **the "local, MIT, span-level
detector" lane filled in H1 2026.** LettuceDetect escalated (v0.2.2 on 2026-07-05,
new generative + CPU models, a July SOTA paper, ~582 stars) and vLLM's HaluGate
(Dec 2025) shipped nearly athena's exact architecture inside the dominant serving
stack. Leading with "local zero-shot NLI detector" now reads as behind-the-curve.

**The defensible move: stop selling athena as a detector and sell it as the
verification layer on top of any detector — led by the agent circuit-breaker.**
Two independent research passes (April and July) converge on the same conclusion:
`verify_step()` is the single most differentiated, least-copyable feature.
Competitors flag; almost none halt an agent chain, and the ones that gate are
welded to one serving stack. That is the open lane.

---

## Part 1 — Where the product actually stands (2026-07-08)

Reconciling the stale April audits against the current tree.

### Shipped and solid
- Clean typed `verify()` with async / batch / streaming variants; `verify_step()`
  circuit-breaker; `verified_completion()`.
- False-positive calibration: anaphora windowing, contradiction-aware rescue,
  numeric gate, check-worthiness filter. Faithful FP rate **16.9% → 4.6%** (base),
  3.4% (large). Synthetic hallucination-catch F1 **91.3% → 95.0%**.
- Real-world benchmarks run and committed honestly: **RAGTruth QA 0.71 balanced
  accuracy, HaluEval QA 0.69**, zero-shot, with the losing F1 row (0.47) shown
  and explained. `benchmarks/RESULTS.md` is reproducible.
- Per-claim `supporting_spans` populated (was the empty-field gap in the April audit).
- Integrations present: LangChain `VerifyingLLM`, LlamaIndex `VerifyingPostprocessor`,
  LangGraph `VerifyStepNode`, CrewAI. Examples are runnable (mock-based, no keys).
- `latency_budget_ms` knob, CLI entry point, MiniCheck opt-in backend.
- Circuit-breaker GIF + measured-improvements plot. 162 tests, ruff + mypy --strict green.
- Build is publishable (`twine check` passes on the committed wheel + sdist);
  `release.yml` is a tag-triggered PyPI trusted-publishing workflow.

### Remaining gaps, ranked
1. **PyPI publish (hard blocker).** `pip install athena-verify` → 404. No git tags.
   Everything else is downstream of this — a broken first install kills a launch.
2. **Live demo** — built now in `demo/` (Gradio Space); needs to be deployed and
   linked above the fold. This is the single highest-leverage traction asset:
   most people will not `pip install` a 1.2 GB download to try it.
3. **Positioning drift across docs** — README/LAUNCH lead with "detector"; the
   defensible lead is the circuit-breaker layer (see Part 3). Needs a deliberate
   hero-copy decision, not a silent rewrite.
4. **Stale asset hygiene** — `docs/blog_post.md` rewritten to current numbers;
   `docs/models.md` de-fabricated; README comparison de-"TBD"-ed. Done in this pass.
5. **OTel/Langfuse exporters exist but `verify()` never calls them**; no
   auto-instrumentation. Post-launch polish.

---

## Part 2 — The market in mid-2026 (what changed, verified July 2026)

| Project | License | Local | Granularity | Status mid-2026 |
|---|---|---|---|---|
| **LettuceDetect** (KRLabs) | MIT | Yes | Token/span | **v0.2.2 (2026-07-05), ~582★.** v2 generative (Qwen-2B) + mmBERT + CPU "TinyLettuce"; now covers code/tool-output/agentic; SOTA paper (arXiv 2607.00895, 2026-07-01). Escalating, not stalling. |
| **vLLM HaluGate** | Open (vLLM Semantic Router) | Yes, in-process | Token+span | **Dec 2025.** Check-worthiness gate → NLI → span merge — nearly athena's exact architecture, real-time, but welded to the vLLM stack. |
| **Vectara HHEM** | 2.1-Open (open) | Yes | Passage-level | Mature factual-consistency baseline. Not sentence/span. |
| **Patronus Lynx** | Open (Llama) | Heavy | Answer-level | Reference open LLM-judge; large footprint, not zero-shot, not span. |
| **Bespoke-MiniCheck-7B** | Llama-3.1 (non-MIT) | Yes (Ollama) | Sentence/claim | Tops LLM-AggreFact ~77%; 7B, restrictive license. |
| **Paladin-mini** | Open | Yes | Claim | New (arXiv 2506.20384, June 2026), 3.8B Phi-4-mini grounding. |
| **LibreEval 1.0** (Arize) | Open | Yes | Answer-level | 2026; largest open RAG-hallucination dataset + cheap fine-tuned detector. |
| **Ragas / TruLens / DeepEval** | Open | LLM-judge | Claim | Canonical offline eval; not zero-shot local runtime. |
| **Cleanlab TLM** | Cloud | No | Score | **Acqui-hired by Handshake (Jan 2026);** standalone future uncertain. |
| **Galileo Luna** | Cloud | No | Platform | **Acquired by Cisco (closed 2026-05-22);** now enterprise-gated. |

**What changed since the April plans:**
- LettuceDetect went from a beatable peer to a well-resourced moving target with
  academic backing. **Do not compete on detection F1** — a zero-shot NLI model
  will likely lose to a detector fine-tuned on the exact benchmark. That's fine;
  it's not our lane.
- A near-clone of athena's architecture shipped inside the most-used inference
  server (HaluGate). This validates the design and removes "novel architecture"
  as a claim — but HaluGate is stack-locked, which is our opening.
- Both commercial competitors from the April rumor list left the open field
  (Cleanlab→Handshake, Galileo→Cisco). Minor tailwind.

**Demand signal, verified:** LangChain **#33191** ("HallucinationDetector",
opened Oct 2025, asking for a pluggable runtime NLI hallucination module) was
**closed as "not planned."** Real demand + the core framework explicitly
declining to own it = a validated wedge for a drop-in library. People are already
hand-rolling detect-and-revise loops (multiple DIY repos and TDS write-ups),
which is exactly athena's shape.

---

## Part 3 — Positioning (the decision that matters most)

### The reframe
> **athena-verify is the verification and self-healing layer for RAG and agents —
> not another detector.** It flags ungrounded sentences with per-claim source
> spans, proposes the corrected sentence, and — the part nothing else ships as a
> drop-in — halts an agent chain the moment a step stops being grounded. Zero-shot
> and provider-neutral by default, with a swappable detection backend so a
> stronger trained detector (LettuceDetect, HHEM, MiniCheck) can slot in
> underneath the same revision + circuit-breaker layer.

### Why this survives the first HN/Reddit comment
Three pillars, in order of defensibility:

1. **Framework-agnostic agent circuit-breaker (the lead).** `verify_step()`
   returns pass/halt against a step's evidence, in any agent loop. LettuceDetect's
   own July-2026 paper explicitly disclaims agent-trajectory monitoring; HaluGate's
   gate is welded to vLLM. A provider-neutral, drop-in circuit-breaker for any
   agent (LangGraph, CrewAI, custom) is the clearest white space. This also maps
   to a named, growing pain: OWASP ASI08 cascading failures; agent studies show
   >57% of errors originate in early steps and cascade.
2. **Detect → revise, not just score.** `suggest_revisions=True` returns the fix
   in the same pass. Cloud APIs (Azure, Vectara) added correction; local OSS
   libraries (Ragas, TruLens, HHEM, LettuceDetect) still mostly only flag. Revision
   is a supporting differentiator, not the headline.
3. **Backend-agnostic + honest zero-shot detection + per-claim spans.** The
   any-corpus, no-fine-tune, library-not-platform pitch for the LangChain #33191
   crowd who got told "not planned." Lead with balanced accuracy (0.71), publish
   the losing F1 (0.47) with the class-imbalance explanation. Honesty is the
   credibility play — it is what earns respect from serious ML people, and it is
   what a fine-tuned-benchmark-gaming competitor can't buy.

### What NOT to claim
- Not "SOTA detection," not "beats LettuceDetect on F1," not "novel architecture."
- Not a hosted dashboard / SaaS / web UI — that lane is saturated (Langfuse,
  Arize, Datadog, and now Cisco/Splunk via Galileo).

---

## Part 4 — Vertical wedges (from the April market research, still valid)

Keep the generic RAG-dev top-of-funnel, but these three verticals reshape the
product roadmap and give concrete, quotable pain for launch copy:

1. **Agent-cascade prevention** (the circuit-breaker's home). OWASP ASI08; >57%
   early-step error cascade; literature explicitly asks for "circuit breaker
   patterns at pipeline checkpoints." This is the lead vertical.
2. **Legal citation verification.** 1,227+ documented hallucinated-citation cases
   globally; a Sixth Circuit $30K sanction (March 2026); 30–45% fabricated-citation
   rates. Local execution is a hard requirement (privilege). A future `citation_mode`
   (fetch opinion → verify quoted passage present) is a strong commercial-edition wedge.
3. **Voice AI / high-QPS RAG.** Per-utterance verification at <200 ms maps to the
   `latency_budget_ms` knob no competitor exposes.

Tier-2 (reachable with current product): enterprise KBs, financial Q&A (needs the
numeric gate we already have), journalism quote-match. Tier-3 (noted, too heavy
for solo OSS): medical/clinical.

---

## Part 5 — Execution plan (prioritized)

### Now (unblock the launch) — days, not weeks
- [ ] **Publish to PyPI.** Configure the trusted publisher on pypi.org for
      `athena-verify`, then `git tag v0.1.0 && git push origin v0.1.0` to trigger
      `release.yml`. Verify `pip install athena-verify` in a clean venv on macOS + Linux.
- [ ] **Deploy the live demo.** Push `demo/` to a Hugging Face Space (Gradio, CPU
      basic). Link it above the fold in the README. Pre-warm confirmed in `app.py`.
- [ ] **Decide the hero copy.** Approve the Part 3 reframe (circuit-breaker lead,
      "verification layer") and update the README's first two lines + LAUNCH.md
      Show HN title accordingly. This is a positioning decision, not a mechanical edit.

### Pre-launch polish — a few days
- [ ] Add HaluGate + LettuceDetect-v2 to the competitive table honestly (treat them
      as backends / stack-locked, not rivals). Quote LangChain #33191 in the launch post.
- [ ] Wire the existing `.to_otel_span()` into `verify()` (r/LocalLLaMA + platform-eng ask).
- [ ] Optional: Ollama-native judge client; add a Colab alongside the Space.
- [ ] Fresh-VM dress rehearsal of the quickstart.

### Launch — follow LAUNCH.md, with these mid-2026 calibrations
Set expectations honestly: **a launch is a ~24-hour pulse, not a growth engine.**
Median AI-tool HN launch ≈ 120 stars/24h, ~289/7d, and the "Show HN" tag itself
gives *no* statistical edge once you control for score/timing/reputation
(arXiv:2511.04453). **What wins is one sharp, shareable hook, not the venue.**
Athena's hook is the efficiency/benchmark angle: *"CPU-real-time, zero-shot RAG
hallucination detection with an agent circuit-breaker — no GPU, no API key."*

- **Sequence:** Show HN first (maker's comment within 5 min), then same-day X
  launch thread + r/LocalLLaMA (770k, best single sub) 30–60 min later. Concentrate
  into one **12–17 UTC weekday window** to spike star velocity; weekend effect is
  negligible. Stagger r/RAG, r/AI_Agents, r/LLMDevs, and a single r/MachineLearning
  `[P]` post (weekend) across the week with fresh copy each time.
- **Lead r/AI_Agents with the circuit-breaker** — most native fit there.
- **Skip r/programming** (banned all AI/LLM content, late 2025).
- **DevHunt > Product Hunt** for dev tools; PH has declined for indie/dev tools
  ("credibility, not customers") — treat PH as an optional badge only.
- **X in 2026** reportedly rewards author replies and now boosts external blog
  links — thread + reply diligently, and link the blog post.
- **Seed, don't spam:** Latent Space (newsletter + Discord, on-topic), Ollama and
  Hugging Face Discords (help first). Newsletters with real submission doors:
  **PyCoder's Weekly** (form, best Python fit), **Console.dev** (email),
  **The Changelog News** (form), **Latent Space** (guest form, warm-intro only).
  TLDR AI / Ben's Bites (still active, pivoted) have no open door — you get in by
  being newsworthy on HN/GitHub.
- **Awesome-lists (no star gate — submit day one):** `Danielskry/Awesome-RAG`
  (the prominent RAG-tools list) and `tensorchord/Awesome-LLMOps`. Gated for later:
  `EthicalML/awesome-production-machine-learning` (≥500★),
  `steven2358/awesome-generative-ai` (≥1000★; "Discoveries" tier below that).
  `kyrolabs/awesome-langchain` auto-closes brand-new repos — submit once there's history.
- **Benchmark-honesty asset:** trust in public benchmarks collapsed in 2025–26
  (Leaderboard Illusion, Karpathy's "loss of trust," Stack Overflow: only 3% of devs
  highly trust AI-tool accuracy). Publishing weaknesses + false-negative rate now
  *reads as credible*. Add the specific RAGTruth caveat — its original span
  annotations had heavy false negatives (re-annotation raised flagged spans 86→865
  on one 408-example subset, arXiv:2603.27752) — and you signal expert-level care.

### Second wave (the credibility / funding / O-1 multiplier) — weeks after
- [ ] **Opt-in trained backend** (LettuceDetect's recipe on RAGTruth) shipped as
      `nli_model="trained"`, keeping zero-shot default. Closes the in-domain F1 gap
      for people who want it and produces a real "v0.2, now with a trained backend
      and a paper" story.
- [ ] **Benchmark write-up** on honest zero-shot grounding → arXiv now for
      citations, then a reviewed workshop version (ACL/NeurIPS/ICML eval or RAG
      workshop). This is the single strongest item for both traction and the O-1
      "authorship" + "original contribution" criteria.

---

## Part 6 — Aligning the launch with the reputation / funding / O-1 goal

The goal is not raw stars — it's durable credibility (top-ML-engineer signal, dev
trust, funding and O-1 leverage). That changes what to optimize:

- **Honest depth beats hype.** The blog post ("what I learned building this,"
  including where zero-shot hits a ceiling and where athena loses) is the highest-
  value reputation asset. Serious people reward the honesty a benchmark-gamer can't
  fake. Keep the losing rows in.
- **Build for dependents, not stars.** This is now evidence-backed, not opinion:
  a 2026 RCT found displaying star counts has **zero causal effect on downloads**
  (arXiv:2603.07919), stars↔downloads correlation is only 0.14–0.47, and ~6M fake
  stars are catalogued — so USCIS officers, attorneys, and investors all discount
  them. What actually counts for the O-1A "original contribution" criterion and for
  funding diligence: **PyPI download percentiles** vs comparable libraries (frame as
  a ranking, never raw counts), a **populated GitHub/PyPI dependents graph** (get
  real projects to `import athena_verify`), **named-enterprise production letters**,
  and getting **ranked #1 in a *defined set*** (top of a RAG-hallucination-detection
  awesome-list or a public leaderboard). Instrument all of this from day one.
  - *Caveat corrected from LAUNCH.md:* the Jan 2025 USCIS update (PA-2025-02) is
    **O-1A only** and does not literally name "GitHub/open source" — it strengthened
    comparable-evidence for AI/emerging tech. Favorable, but frame it as comparable
    evidence of impact and verify current wording. (Fixed in LAUNCH.md §8.)
  - *Reference playbook:* Alexey Inkin's approved EB-1A petition — but note his own
    lesson matches our risk exactly: raw download stats were **insufficient**; he
    won by getting institutional validation and engineering a "defined set" his
    project could rank #1 in (he lobbied FlutterGems to create a category). Copy the
    *defined-set + downstream-letters* move, not just the download stats.
- **The trained-backend + paper is the one effort that serves all three goals at
  once** — a citable paper (authorship), a real adoption story (original
  contribution), and a press/talk hook. Prioritize it as the second wave.
- **Maintainer-grade responsiveness** in the launch window and first month:
  recruiters and investors read the issues tab and your response style, not just
  the star graph.

Full O-1 criterion-by-criterion plan and the launch-day hour-by-hour: LAUNCH.md
sections 4–8.

---

## Part 7 — What to explicitly NOT do

- Don't compete on detection F1 against a benchmark-fine-tuned model. Different lane.
- Don't build a hosted dashboard / SaaS / web UI. Saturated and off-thesis.
- Don't lead with "zero-shot NLI detector" — 2024-era framing in a 2026 market.
- Don't launch all channels the same day; stagger and rewrite copy each time.
- Don't ship any number you can't reproduce. Every claim → a rerunnable script.
- Don't delay the launch for the paper; ship honest benchmarks now, paper as wave two.
