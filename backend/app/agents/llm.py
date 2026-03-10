from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from app.config import settings


def get_agent_llm(streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "zhipuai":
        from langchain_community.chat_models import ChatZhipuAI

        return ChatZhipuAI(
            model=settings.llm_model,
            api_key=settings.zhipuai_api_key,
            temperature=0.3,
            streaming=streaming,
        )
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
        streaming=streaming,
    )
