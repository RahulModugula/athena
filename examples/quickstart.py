"""Quickstart example for athena-verify.

Run with: python examples/quickstart.py
"""

from athena_verify import verify

# Your retrieved context chunks (from any RAG pipeline)
context = [
    "The indemnification clause in Section 12.1 states that the liability cap "
    "is $2,000,000 per incident, with an aggregate annual cap of $5,000,000.",
    "The agreement was signed on January 15, 2024, by both parties.",
    "Either party may terminate this agreement with 90 days written notice.",
]

# The LLM-generated answer to verify
answer = (
    "The liability cap is $1,000,000 per incident. "
    "The agreement can be terminated with 30 days notice. "
    "It was signed on January 15, 2024."
)

# Run verification
result = verify(
    question="What are the key terms of the contract?",
    answer=answer,
    context=context,
)

# Print results
print(f"Overall trust score: {result.trust_score:.2f}")
print(f"Verification passed: {result.verification_passed}")
print(f"Number of sentences: {len(result.sentences)}")
print()

for sent in result.sentences:
    status_emoji = {
        "SUPPORTED": "✅",
        "PARTIAL": "🟡",
        "UNSUPPORTED": "🔴",
        "CONTRADICTED": "❌",
    }.get(sent.support_status, "❓")

    print(f"  {status_emoji} [{sent.support_status}] (trust: {sent.trust_score:.2f})")
    print(f"     \"{sent.text}\"")
    print()

if result.unsupported:
    print(f"\n⚠️  {len(result.unsupported)} unsupported claim(s) detected!")
    for sent in result.unsupported:
        print(f"   - \"{sent.text}\"")
else:
    print("\n✅ All claims are supported by the context.")
