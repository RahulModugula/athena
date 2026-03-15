from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.config import settings


def get_agent_llm(streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "zhipuai":
        from langchain_community.chat_models import ChatZhipuAI

        return ChatZhipuAI(  # type: ignore[return-value]
            model=settings.zhipuai_model,
            api_key=settings.zhipuai_api_key,
            temperature=0.3,
            streaming=streaming,
        )
    elif settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[return-value]
            model=settings.openrouter_model,
            api_key=SecretStr(settings.openrouter_api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            streaming=streaming,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[return-value]
            model_name=settings.llm_model,
            api_key=SecretStr(settings.anthropic_api_key),
            temperature=0.3,
            streaming=streaming,
            timeout=None,
            stop=None,
        )
