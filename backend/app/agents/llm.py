from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.config import settings


def get_agent_llm(streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "zhipuai":
        from langchain_community.chat_models import ChatZhipuAI

        return ChatZhipuAI(
            model=settings.zhipuai_model,
            api_key=settings.zhipuai_api_key,
            temperature=0.3,
            streaming=streaming,
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=settings.llm_model,
        api_key=SecretStr(settings.anthropic_api_key), timeout=None, stop=None,
        temperature=0.3,
        streaming=streaming,
    )
