# Security & Data Privacy

## What Data Leaves Your Machine

Athena-verify has two operating modes with different privacy implications:

### NLI-only (Default) — Fully Local
- **Your data:** Stays on your machine
- **Models:** DeBERTa-v3 cross-encoder (~1.2 GB) runs locally
- **API calls:** Zero
- **Best for:** Sensitive data, compliance-heavy workloads (healthcare, legal, finance)

```python
from athena_verify import verify

# Fully offline — nothing leaves your infrastructure
result = verify(
    question="...", 
    answer="...", 
    context=chunks,
)
```

### NLI + LLM Judge — API Calls Required
When you enable `use_llm_judge=True`, sentences are sent to an LLM for additional verification:

| Provider | Data Sent | Retention | Best For |
|----------|-----------|-----------|----------|
| **Anthropic** | sentence + context | 30 days (privacy mode off) | Enterprise with API contract |
| **OpenAI** | sentence + context | Log for abuse detection | Lighter workloads |
| **Local (LM Studio)** | Nothing | Stays local | Maximum control |

```python
# API calls made for borderline sentences
from athena_verify.llm_judge import AnthropicJudge

client = AnthropicJudge()
result = verify(..., use_llm_judge=True, llm_client=client)
```

## Staying Fully Offline

**Option 1: NLI-only (recommended)**
```python
result = verify(question="...", answer="...", context=chunks)
# use_llm_judge defaults to False
```

**Option 2: Local LLM judge**
```python
# Run LM Studio locally, then:
import requests

class LocalJudge:
    def complete(self, prompt: str) -> str:
        resp = requests.post("http://localhost:8000/completions", 
                           json={"prompt": prompt})
        return resp.json()["choices"][0]["text"]

result = verify(..., use_llm_judge=True, llm_client=LocalJudge())
```

## Compliance Notes

- **HIPAA:** Use NLI-only mode or self-hosted LLM judge
- **GDPR:** Be aware API calls may log prompt text; check vendor terms
- **SOC 2:** NLI-only mode has zero external dependencies
