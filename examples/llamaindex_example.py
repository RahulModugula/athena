"""LlamaIndex integration example for athena-verify.

Demonstrates how to use VerifyingPostprocessor with a LlamaIndex query engine.

Prerequisites:
    pip install athena-verify llama-index

Run with: python examples/llamaindex_example.py
"""

from athena_verify.integrations.llamaindex import VerifyingPostprocessor


def example_with_query_engine():
    """Example using VerifyingPostprocessor with LlamaIndex.

    This requires a running LlamaIndex setup with an index.
    See the LlamaIndex docs for building your own index.
    """
    # This is a demonstration of the integration pattern.
    # In practice, you would:
    #
    # from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    # from athena_verify.integrations.llamaindex import VerifyingPostprocessor
    #
    # # Build your index
    # documents = SimpleDirectoryReader("./data").load_data()
    # index = VectorStoreIndex.from_documents(documents)
    #
    # # Add verification postprocessor
    # postprocessor = VerifyingPostprocessor(
    #     trust_threshold=0.70,
    #     flag_unsupported=True,
    # )
    #
    # # Create query engine with verification
    # engine = index.as_query_engine(
    #     response_postprocessors=[postprocessor],
    # )
    #
    # # Query
    # response = engine.query("What is the indemnification cap?")
    # print(response)
    #
    # # Access verification results
    # verification = postprocessor.last_verification
    # print(f"Trust score: {verification.trust_score}")
    # print(f"Unsupported: {len(verification.unsupported)}")
    #
    # # Or from response metadata
    # if "athena_verification" in response.metadata:
    #     v = response.metadata["athena_verification"]
    #     print(f"Trust: {v['trust_score']}")

    print("See the code comments for the full integration pattern.")
    print("Install llama-index to use this integration.")


if __name__ == "__main__":
    example_with_query_engine()
