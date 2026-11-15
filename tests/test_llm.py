from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import Settings
from app.core.llm import get_llm
from app.prompts.system import SYSTEM_PROMPT, build_chat_messages


def test_get_llm_uses_settings():
    s = Settings(_env_file=None, model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-x")
    llm = get_llm(s)
    assert llm.model_name == "gpt-4o"
    assert llm.openai_api_base == "https://api.openai.com/v1"


def test_system_prompt_nonempty_role():
    assert "客服" in SYSTEM_PROMPT and len(SYSTEM_PROMPT) > 20


def test_build_chat_messages_prepends_system():
    history = [HumanMessage("hi"), AIMessage("hello")]
    msgs = build_chat_messages(history, "帮我催一下")
    assert msgs[0].type == "system"
    assert msgs[1:3] == history
    assert msgs[-1] == HumanMessage(content="帮我催一下")
