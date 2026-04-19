"""LangChain integration example for athena-verify.

Demonstrates how to use VerifyingLLM with a LangChain RetrievalQA chain.
This example uses FakeListLLM for demonstration without requiring API keys.

Prerequisites:
    pip install athena-verify langchain langchain-core

Run with: python examples/langchain_example.py
"""

from langchain_core.documents import Document
from langchain.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain_core.llms.fake import FakeListLLM

from athena_verify.integrations.langchain import VerifyingLLM


class SimpleInMemoryRetriever(BaseRetriever):
    """Simple in-memory retriever for demonstration."""

    documents = [
        Document(
            page_content="The indemnification cap is $1 million per incident.",
            metadata={"source": "contract.pdf"},
        ),
        Document(
            page_content="The warranty period is 12 months from purchase date.",
            metadata={"source": "warranty.pdf"},
        ),
        Document(
            page_content="The liability clause limits total damages to $5 million.",
            metadata={"source": "contract.pdf"},
        ),
    ]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Return all documents for this demo (simple matching)."""
        return self.documents


def example_with_fake_llm():
    """Example using VerifyingLLM with a FakeListLLM for demonstration.

    This example does not require any API keys and demonstrates the
    integration pattern without external dependencies.
    """
    # Set up a simple in-memory retriever
    retriever = SimpleInMemoryRetriever()

    # Create a FakeListLLM that returns predefined responses
    fake_llm = FakeListLLM(
        responses=[
            "The indemnification cap is $1 million per incident according to the contract.",
            "The warranty period is 12 months and the liability is capped at $5 million.",
        ]
    )

    # Wrap with VerifyingLLM for hallucination detection
    verifying_llm = VerifyingLLM(
        llm=fake_llm,
        trust_threshold=0.70,
        on_unsupported="flag",  # Options: "flag", "warn", "reject"
    )

    # Create retrieval QA chain
    chain = RetrievalQA.from_llm(
        llm=verifying_llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # Example queries
    queries = [
        "What is the indemnification cap?",
        "What is the warranty period?",
    ]

    print("Running LangChain + Athena example...\n")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        result = chain({"query": query})

        print(f"\nResponse: {result['result']}")

        # Access verification metadata
        if hasattr(verifying_llm, "last_verification") and verifying_llm.last_verification:
            verification = verifying_llm.last_verification
            print(f"Trust score: {verification.trust_score:.2f}")
            print(f"Verified claims: {len(verification.verified)}")
            print(f"Unsupported claims: {len(verification.unsupported)}")

        print("\nSource documents:")
        for doc in result.get("source_documents", []):
            print(f"  - {doc.page_content[:80]}...")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    example_with_fake_llm()
