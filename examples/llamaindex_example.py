"""LlamaIndex integration example for athena-verify.

Demonstrates how to use VerifyingPostprocessor with a LlamaIndex query engine.
This example uses mock components to run without requiring API keys.

Prerequisites:
    pip install athena-verify llama-index

Run with: python examples/llamaindex_example.py
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.response_synthesizers import SimpleSummarize

from athena_verify.integrations.llamaindex import VerifyingPostprocessor


def example_with_mock_components():
    """Example using VerifyingPostprocessor with mock LLM and index.

    This example demonstrates the integration pattern without requiring
    external API keys or complex setup.
    """
    # Create sample documents
    documents = [
        Document(
            text="The indemnification cap is $1 million per incident.",
            metadata={"source": "contract.pdf"},
        ),
        Document(
            text="The warranty period is 12 months from purchase date.",
            metadata={"source": "warranty.pdf"},
        ),
        Document(
            text="The liability clause limits total damages to $5 million.",
            metadata={"source": "contract.pdf"},
        ),
    ]

    # Build an index from the documents
    print("Building LlamaIndex from sample documents...")
    index = VectorStoreIndex.from_documents(documents)

    # Create a mock LLM that returns predefined responses
    mock_llm = MockLLM(
        max_tokens=512,
        response=(
            "Based on the documents, the indemnification cap is $1 million per incident, "
            "the warranty period is 12 months, and the total liability is capped at $5 million."
        ),
    )

    # Create the verifying postprocessor
    postprocessor = VerifyingPostprocessor(
        trust_threshold=0.70,
        flag_unsupported=True,
    )

    # Create query engine with verification
    print("Setting up query engine with verification...")
    engine = index.as_query_engine(
        llm=mock_llm,
        response_synthesizer=SimpleSummarize(),
        response_postprocessors=[postprocessor],
    )

    # Example queries
    queries = [
        "What is the indemnification cap?",
        "What is the warranty period?",
    ]

    print("\nRunning LlamaIndex + Athena example...\n")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        response = engine.query(query)

        print(f"\nResponse: {response}")

        # Access verification results from response metadata
        if response.metadata and "athena_verification" in response.metadata:
            v = response.metadata["athena_verification"]
            print(f"Trust score: {v.get('trust_score', 'N/A')}")
            print(f"Verified claims: {len(v.get('verified', []))}")
            print(f"Unsupported claims: {len(v.get('unsupported', []))}")
        else:
            print("(Verification metadata not available in response)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    example_with_mock_components()
