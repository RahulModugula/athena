"""LangChain integration example for athena-verify.

Demonstrates how to use VerifyingLLM with a LangChain RetrievalQA chain.

Prerequisites:
    pip install athena-verify langchain langchain-openai

Run with: python examples/langchain_example.py
"""

from athena_verify.integrations.langchain import VerifyingLLM


def example_with_retrieval_qa():
    """Example using VerifyingLLM with LangChain RetrievalQA.

    This requires a running LangChain setup with a retriever.
    See the LangChain docs for setting up your own retriever.
    """
    # This is a demonstration of the integration pattern.
    # In practice, you would:
    #
    # from langchain_openai import ChatOpenAI
    # from langchain.chains import RetrievalQA
    #
    # llm = ChatOpenAI(model="gpt-4o")
    # verifying_llm = VerifyingLLM(
    #     llm=llm,
    #     trust_threshold=0.70,
    #     on_unsupported="flag",  # or "warn", "reject"
    # )
    #
    # chain = RetrievalQA.from_llm(
    #     llm=verifying_llm,
    #     retriever=your_retriever,
    #     return_source_documents=True,
    # )
    #
    # result = chain({"query": "What is the indemnification cap?"})
    #
    # # Access verification metadata
    # verification = verifying_llm.last_verification
    # print(f"Trust score: {verification.trust_score}")
    # print(f"Unsupported claims: {len(verification.unsupported)}")

    print("See the code comments for the full integration pattern.")
    print("Install langchain and langchain-openai to use this integration.")


if __name__ == "__main__":
    example_with_retrieval_qa()
