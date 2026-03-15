from collections.abc import AsyncIterator

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from app.config import settings
from app.generation.prompts import RAG_PROMPT, format_context

logger = structlog.get_logger()


def get_llm(streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "zhipuai":
        from langchain_community.chat_models import ChatZhipuAI

        return ChatZhipuAI(  # type: ignore[return-value]
            model=settings.zhipuai_model,
            api_key=settings.zhipuai_api_key,
            temperature=0.1,
            streaming=streaming,
        )
    elif settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[return-value]
            model=settings.openrouter_model,
            api_key=SecretStr(settings.openrouter_api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            streaming=streaming,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[return-value]
            model_name=settings.llm_model,
            api_key=SecretStr(settings.anthropic_api_key),
            temperature=0.1,
            streaming=streaming,
            timeout=None,
            stop=None,
        )


async def generate_answer(question: str, chunks: list[dict[str, object]]) -> str:
    llm = get_llm(streaming=False)
    context = format_context(chunks)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer: str = await chain.ainvoke({"question": question, "context": context})
    return answer


async def stream_answer(question: str, chunks: list[dict[str, object]]) -> AsyncIterator[str]:
    llm = get_llm(streaming=True)
    context = format_context(chunks)
    chain = RAG_PROMPT | llm | StrOutputParser()
    async for token in chain.astream({"question": question, "context": context}):
        yield token
