# athena-verify Launch Guide

Everything needed to launch athena-verify and pull real traction: who it is for, what to build first, where to post, exactly what to say (copy you can paste), the hour-by-hour launch sequence, and how to turn the launch into evidence for an O-1A visa.

The copy in this file is written to sound like a person, not a press release. Keep it that way when you post. No hype words, no "SOTA", no exclamation marks, no em dashes. Reddit and Hacker News punish marketing tone and reward honesty, especially admitting where the thing loses.

---

## 0. The one thing that actually matters

Star count is decided by your README and your demo, not by which site you post on. One study of 188k Show HN posts found the HN score explains only about 8 percent of star variance. So the launch venue gets you clicks; the repo converts clicks to stars. Prep the repo first, then launch. A front page Show HN plus a strong r/LocalLLaMA post can realistically mean 300 to 1000 stars in the first 48 hours if the repo is ready, and roughly nothing if the first `pip install` fails.

Rough math to set expectations: about 1.4 stars per HN upvote, and about 92 percent of the star gain arrives within 48 hours. Plan your energy around those two days.

---

## 1. Who stars this, and the hook for each

Ranked by how much star leverage each group gives you.

1. **The local LLM believer (r/LocalLLaMA).** Runs everything on their own hardware, does not want anything touching a cloud API, and gives stars generously to genuinely local tools. Athena being zero-shot, fully local, and provider neutral is made for them. This is the number one launch audience.
   Hook: it runs 100 percent on your machine, no API keys, nothing leaves your box, works the same on GPT, Claude, Llama, or Qwen.

2. **The RAG app builder (indie and startup).** Shipping a RAG chatbot and scared of it making things up with total confidence.
   Hook: catch hallucinations before your users do, three lines, any model.

3. **The AI agent builder.** Building multi-step or multi-agent systems and worried about one bad fact cascading into everything downstream.
   Hook: verify_step() halts a chain the moment a step stops being grounded. This is the most differentiated and least copyable feature.

4. **The platform or ML engineer at a company.** Needs runtime guardrails but cannot ship data to a paid cloud detector like Patronus or Galileo.
   Hook: local, provider neutral, about 25ms, zero API cost, nothing leaves your infra.

5. **The regulated industry developer (legal, medical, finance).** Needs auditable, offline, per-claim citations.
   Hook: per-claim source spans, runs air gapped.

---

## 2. Prep the repo before posting anything (2 to 4 weeks out)

Do not skip this. A broken first impression cannot be undone.

- [ ] `pip install athena-verify` works in a clean virtualenv on macOS and Linux. Test it on a fresh machine or container. This is the single most common launch killer.
- [ ] README above the fold, in this order: name and one line value prop, then a row of badges (stars, license, build passing, Python version), then the demo GIF, then a copy-paste quickstart that reaches a working result in under two minutes, then the benchmark table, then a live demo link.
- [ ] **A live demo with no signup wall.** This is the biggest missing piece right now and the highest leverage thing to build. A Hugging Face Space (Gradio) or a Google Colab where someone pastes a question, an answer, and context and sees the per-sentence result. HN's Show HN rules literally require something people can try without a signup barrier.
- [ ] The benchmark numbers must be reproducible. Ship the eval script and a Colab that regenerates them. Never write a superlative you cannot back with a rerunnable number.
- [ ] LICENSE, CONTRIBUTING, and three or four issues labeled "good first issue".
- [ ] Warm up the Reddit account. Reddit auto-detects the vast majority of manipulation and a fresh account posting a promo link can get shadow removed. Spend two to four weeks: lurk and upvote, then comment without links, then post. Never use alt accounts to vote on your own thing. It is bannable on both Reddit and HN and it is mostly auto-detected.

---

## 3. Improvements worth making before launch

These move the needle on traction, in priority order.

1. **Live demo (Colab plus a Hugging Face Space).** Highest leverage. Most people will not `pip install` a 400MB-plus download to try it. Let them click and paste.
2. **Trim the cold-start download friction.** First `verify()` pulls the NLI model. Document the size, offer a lightweight model alias, and pre-warm in the Colab so the demo is instant.
3. **A tiny "receipts" visual in the README.** A short code block or image showing the per-claim source spans output, since that is a real differentiator most tools do not have.
4. **The RAGTruth-trained backend and an arXiv note (from PLAN_NEXT.md).** This is the big one. It gives you a defensible in-domain number, a citable paper, and a talk. It helps traction and it is the strongest single item for the O-1 case. Worth doing even if it lands a few weeks after the first launch as a "v0.2, now with a trained backend and a paper" second wave.

---

## 4. Where to post, ranked

### Hacker News, Show HN
Post first. Then Reddit 30 to 60 minutes later, then the X thread the same morning. Aim for a US morning on a Tuesday through Thursday for raw developer traffic. Post your maker's comment within five minutes of submitting, then clear two to three hours to answer every reply quickly and graciously. Agree with critics first, then address them. The discussion is what keeps you on the front page.

### Reddit, in order
Post to one primary sub first, then space the rest across the week. Do not paste the same text everywhere on the same day; write fresh copy each time or spam filters will catch it. Put the GitHub link in the post body or your own first comment, not as a bare link post.

1. r/LocalLLaMA. Lead here. Local is native to this crowd.
2. r/Rag. Exact bullseye audience.
3. r/AI_Agents. Use the self-promotion flair, frame it as agent guardrails.
4. r/ChatGPTCoding. AI tooling is welcome.
5. r/LLMDevs. Post as a resource or tools writeup, not a launch.
6. r/Python. Use the Showcase flair, explain what it is and how it is built.
7. r/MachineLearning. Use the [P] flair, benchmark first, zero marketing tone.
8. r/LangChain. Problem first only, framed as detecting hallucinations in a RAG pipeline.
9. r/artificial and r/OpenAI, only if framed around reliability.

Do not post to r/programming. It banned all AI and LLM content in 2026 and a post there gets removed and can hurt your account.

Verify each sub's live sidebar rules and flair names while logged in before posting. Reddit changed a lot in 2025 and secondhand rule lists go stale.

### X or Twitter, same morning
Hook tweet, then the demo GIF, then two or three tweets of value, then a proof tweet with the numbers, then the repo link and pip install, then a soft "star it if it is useful, feedback welcome." Tag three to five people who actually work in RAG rather than celebrities. Good targets: @jerryjliu0, @llama_index, @LangChainAI, @rlancemartin, @HamelHusain for the evals angle, @simonw, @_akhaliq.

### Product Hunt
Optional same-day amplifier for backlinks and social proof. Not a primary star driver for a dev tool. Do not build the launch around it.

---

## 5. Copy you can paste

Fill in <LINK> with the repo URL. Keep the honest limitation lines in. They are why people trust it.

### Show HN title (pick one)
- Show HN: Athena, a local zero-shot hallucination detector for RAG pipelines
- Show HN: Athena, detect RAG hallucinations offline, no labels, no fine-tuning
- Show HN: Athena, span-level hallucination detection for RAG, no LLM judge needed

### Show HN body
Athena is an open source library that checks whether an LLM answer is actually grounded in the context you retrieved, sentence by sentence. It runs locally on a small NLI model, so there are no API calls and nothing leaves your machine, and it works with any model provider.

You pass in the question, the answer, and the retrieved chunks. You get back a per-sentence trust score, the list of unsupported sentences, and the exact source span that supported each claim.

The honest version of the benchmarks, since that is the first thing anyone asks. It is zero-shot, so it is not trained on any hallucination dataset. On RAGTruth QA it gets about 0.71 balanced accuracy. LettuceDetect reports higher F1 on RAGTruth, but it is fine-tuned on the RAGTruth training split, so that accuracy is domain specific. Athena trades some in-domain accuracy for working on any corpus with no training. On HaluEval QA it lands around 0.69 accuracy, which is roughly GPT-3.5 to GPT-4 prompting territory, except it is local and about 25ms per check.

What I think is actually new here: it is a library and not a hosted API, it is provider neutral, it returns per-claim source spans instead of one number, and it has a circuit-breaker primitive for agents that halts a chain the moment a step stops being grounded.

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

Same body as the r/LocalLLaMA post works here, with the first paragraph tightened to the RAG framing. Lead with the problem, keep the honest numbers, put the link in the body.

### r/AI_Agents post (use the self-promotion flair)
Title: A circuit breaker for agent hallucinations: verify each step against its evidence and halt before a bad fact cascades

In a multi-step agent, one step's output becomes the next step's input. If a step invents a fact, everything downstream is built on it and you usually find out at the end. I built a small open source library that checks each step against its evidence and returns a pass or halt, so you can stop the chain the moment it stops being grounded.

    from athena_verify import verify_step
    step = verify_step(claim=reasoning_step, evidence=chunks, threshold=0.5)
    if step.action == "halt":
        raise RuntimeError(f"ungrounded step blocked, trust={step.trust_score:.2f}")

It runs locally, no API calls, works with any model. It is zero-shot, so honest numbers: about 0.71 balanced accuracy on RAGTruth QA, around 0.69 on HaluEval QA, roughly GPT-3.5 to GPT-4 prompting range but local and about 25ms. Repo and install in the README: <LINK>. Curious how it does on your agent traces.

### X or Twitter thread
1. I kept shipping RAG apps that hallucinated with total confidence. So I built a detector that catches it in real time, sentence by sentence, running fully local. Open source, MIT. Here is how it works.

2. You give it the question, the answer, and your retrieved chunks. It scores each answer sentence against the context and flags the ones that are not supported. Three lines, no API keys, nothing leaves your machine.

3. It is provider neutral. GPT, Claude, Llama, Qwen, does not matter. And it returns the exact source span for each claim, so you can show users why a sentence was trusted or not.

4. Honest numbers, because everyone asks. Zero-shot, no training on any hallucination set. RAGTruth QA around 0.71 balanced accuracy, HaluEval QA around 0.69. GPT-3.5 to GPT-4 prompting range, except local and about 25ms.

5. The part I am most excited about is verify_step() for agents. It halts a multi-step chain the moment a step stops being grounded, before a bad fact reaches the final answer.

6. Free and MIT. pip install athena-verify. Repo: <LINK>. Try it on your own data and tell me what breaks.

### LinkedIn
Lead with the founder story in the first two lines, attach a short slide PDF (problem, solution, demo, numbers), and put the GitHub link in the first comment rather than the post body. Keep it to three hashtags.

---

## 6. Week one amplification

Stagger these, write native copy for each, do not mirror dump.

- dev.to: publish to your own blog first, then cross-post with a canonical URL. Tags: showdev, python, llm, opensource.
- Newsletters that take free submissions: PyCoders Weekly (pycoders.com/submissions), Ben's Bites, The Rundown AI, Data Elixir. Latent Space is the most relevant audience but is warm intro only, no cold email.
- awesome-list pull requests, read each CONTRIBUTING first. Start with Awesome-LLM, Awesome-RAG, Awesome-LLMOps, awesome-langchain. After you cross 500 to 1000 stars, submit to awesome-production-machine-learning (500 star minimum) and awesome-generative-ai (1000 star gate).
- Discord and Slack showcase channels: Hugging Face Discord, LangChain Slack community, LlamaIndex Discord, Ollama Discord.
- Build one runnable example app and submit it to awesome-llm-apps, which wants apps rather than links.

---

## 7. Failure modes that kill a launch

- Over-claiming or unreproducible benchmarks. Drop superlatives, ship a rerunnable eval, show a table not a claim.
- Asking friends to upvote or using alt accounts. Bannable on both platforms and mostly auto-detected. Announce it and let votes happen.
- Broken pip install or a demo behind a waitlist. Test in a clean environment, link a live no-signup demo.
- Posting to the wrong sub. Skip r/programming. Match each sub's flair and rules.
- Hiding that you are the author. Disclose it upfront and lead with substance.
- Pasting identical copy everywhere at once. Stagger and rewrite each time.
- Getting defensive with critics. Agree first, then address. The back and forth is what ranks you.

---

## 8. Turning this into O-1A visa evidence

This is factual research to shape strategy, not legal advice. Before you file anything, get an immigration attorney. The controlling rules are 8 CFR 214.2(o) and the USCIS Policy Manual, everything else is secondary.

### How the O-1A actually works
You either have one giant internationally recognized award, or you document at least three of eight criteria. Meeting three is necessary but not sufficient. USCIS then does a second discretionary step called the final merits determination, asking whether you are genuinely in the small percentage at the very top of the field with sustained acclaim. Strong tech cases get denied at this second step, so the goal is to cleanly document four or more criteria, not the bare three, and to prove impact rather than existence.

The eight criteria you can document: nationally recognized prizes or awards, membership in associations that require outstanding achievement, published material about you in major or trade media, judging the work of others, original contributions of major significance, authorship of scholarly articles, a critical role for a distinguished organization, and a high salary.

### The 2025 update that matters for you
In January 2025 USCIS added an explicit hook for open source to the original contributions criterion. It now lists as an example: contributions to repositories of software, data, designs, protocols, or other technical resources with evidence of significant scientific, scholarly, or business related impact. This did not exist before 2025. It is the closest thing to a green light for an OSS project like athena, and note the word impact. The repo existing is not the point, the field using it is.

### Where athena fits, criterion by criterion
- **Original contributions of major significance.** This is athena's home. Stars alone are never persuasive and are known to be gameable, so USCIS and good attorneys discount raw star counts. What actually counts is adoption evidence: PyPI download percentiles benchmarked against comparable libraries, the GitHub dependents graph showing other projects that depend on it, named companies using it in production, and independent expert letters that cite hard numbers. The public gold standard playbook here is Alexey Inkin's approved EB-1A petition, which is on GitHub. He used package download stats to prove scale, mined the dependents graph to find and get letters from real downstream users, and used commit logs to prove he personally built it. Copy that approach.
- **Authorship of scholarly articles.** An arXiv preprint alone is weak because there is no peer review. Aim for a peer reviewed workshop or conference paper with published proceedings, for example an ACL, NeurIPS, or ICML workshop on evaluation or RAG. A benchmark study on honest zero-shot hallucination detection is a real paper. Post the arXiv version now for citations, then get the reviewed version.
- **Judging the work of others.** Highest leverage, lowest barrier, start now. Accept peer review invitations from workshops and journals in this space, join a program committee, or judge a recognized hackathon. Keep the invitation plus a letter stating how many submissions you reviewed. Do not count reviews inside your own employer, and reviewing pull requests on your own repo does not count.
- **High salary.** Pure paperwork. Assemble your comp documents plus a comparison report using BLS, the OFLC wage library, and Levels.fyi showing you are in the top five to ten percent for your role and region.
- **Press about you.** Coverage in outlets like IEEE Spectrum, The New Stack, VentureBeat, InfoQ, or TechCrunch that is about you and your work, not a paid placement and not just about your employer. A strong launch is the natural trigger for this.
- **Critical role.** Document that you own a core system at a well known or venture backed org, or that you maintain a widely depended on OSS project.

### The 12 to 18 month plan
Target a clean four criteria case: original contributions, authorship, judging, and high salary, with press and critical role as upgrades.

- Months 0 to 2: assemble the salary comparison, start the OSS impact dossier (downloads, dependents, adoption, commit logs proving you are the author), post the arXiv paper, and begin building relationships for expert letters.
- Months 1 to 6: land two or three judging or review roles, and submit the reviewed workshop paper.
- Months 3 to 12: pursue launch tied press, document your critical role, and apply for a competitive grant or award.
- Months 9 to 18: finalize five to eight expert letters with a majority from independent experts who never worked with you, secure a conference talk, and file once four criteria are cleanly documented.

### The honest part
A popular repo alone is never enough, every source agrees on that. Pure engineers are structurally disadvantaged because they rarely have awards, formal memberships, or peer reviewed papers, so you have to manufacture the record from your real work. The number one reason tech cases fail is claiming significance without proving it, so every claim needs third party numbers behind it. The strongest thing you can do for this visa and for the project at the same time is the trained backend plus the benchmark paper in PLAN_NEXT.md. That single effort produces a citable paper, a real adoption story, and a press hook all at once.

---

## 9. Targets

- First 48 hours: front page of Show HN, top of r/LocalLLaMA for the day, 300 plus stars.
- First two weeks: 1000 plus stars, into three or four awesome-lists, one newsletter mention.
- First quarter: the trained backend and an arXiv note shipped as a second wave, one talk submitted, first adoption by a named project.
