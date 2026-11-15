from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key,
        temperature=s.temperature,
    )
