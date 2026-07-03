---
title: athena-verify
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
preload_from_hub:
  - cross-encoder/nli-deberta-v3-base
---

<!-- preload_from_hub bakes the NLI weights into the image at build time so the
     Space cold-starts fast. Note: it ignores a custom HF_HOME and can't fetch
     private repos. app.py also runs a warmup inference on boot. -->

# athena-verify — live demo

Paste a question, an LLM answer, and the context chunks you retrieved. Each
sentence of the answer is scored against the context; unsupported sentences are
flagged in red, with the exact source span that did (or didn't) support each
claim. Runs the real [`athena-verify`](https://github.com/RahulModugula/athena)
library, zero-shot, on CPU — no API keys, nothing leaves this Space.

## Deploy your own copy

1. Create a new Space at https://huggingface.co/new-space (SDK: Gradio, CPU basic).
2. Upload `app.py` and `requirements.txt` from this folder.
3. The Space builds and pre-warms the NLI model on boot; first request is fast.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The DeBERTa-v3 NLI model (~1.2 GB) downloads on first use.
