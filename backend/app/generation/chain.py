from collections.abc import AsyncIterator

import structlog
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.generation.prompts import RAG_PROMPT, format_context

logger = structlog.get_logger()


def get_llm(streaming: bool = False) -> ChatZhipuAI:
    return ChatZhipuAI(
        model=settings.llm_model,
        api_key=settings.zhipuai_api_key,
        temperature=0.1,
        streaming=streaming,
    )


async def generate_answer(question: str, chunks: list[dict]) -> str:
    llm = get_llm(streaming=False)
    context = format_context(chunks)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer: str = await chain.ainvoke({"question": question, "context": context})
    return answer


async def stream_answer(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    llm = get_llm(streaming=True)
    context = format_context(chunks)
    chain = RAG_PROMPT | llm | StrOutputParser()
    async for token in chain.astream({"question": question, "context": context}):
        yield token
