from app.core.config import Settings

def test_settings_defaults(monkeypatch):
    # pymilvus 的 load_dotenv() 会把 .env 漏进 os.environ,在非字母序的测试顺序下
    # 覆盖这里的默认值断言;构造 Settings 前先把被测键从环境中摘掉(hermetic)。
    for key in (
        "model",
        "base_url",
        "api_key",
        "temperature",
        "max_history_tokens",
        "max_history_messages",
        "database_url",
        "rerank_model",
        "rerank_top_k",
        "hybrid_dense_top_k",
        "hybrid_sparse_top_k",
        "rerank_use_fp16",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.model == "qwen3:8b"
    assert s.base_url == "http://localhost:11434/v1"
    assert s.api_key == "ollama"
    assert s.temperature == 0.0
    assert s.max_history_tokens == 4000
    assert s.max_history_messages == 10
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_top_k == 10
    assert s.hybrid_dense_top_k == 50
    assert s.hybrid_sparse_top_k == 50
    assert s.rerank_use_fp16 is True


def test_get_settings_default_model():
    from app.core.config import get_settings

    assert get_settings().model == "qwen3:8b"

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MODEL", "gpt-4o")
    s = Settings(_env_file=None)
    assert s.model == "gpt-4o"


def test_database_url_default():
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.database_url.startswith("mysql+pymysql://")


def test_intent_model_defaults_follow_main(monkeypatch):
    from app.core.config import Settings

    for key in (
        "intent_model",
        "intent_model_base_url",
        "intent_model_api_key",
        "intent_small_model",
        "intent_escalate_threshold",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.intent_model == s.model
    assert s.intent_model_base_url == s.base_url
    assert s.intent_model_api_key == s.api_key
    assert s.intent_small_model is None
    assert s.intent_escalate_threshold == 0.6


def test_tool_executor_defaults(monkeypatch):
    from app.core.config import Settings

    for key in ("tool_timeout_seconds", "tool_retries", "mcp_write_tools"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.tool_timeout_seconds == 120.0
    assert s.tool_retries == 2
    assert s.mcp_write_tools == []
