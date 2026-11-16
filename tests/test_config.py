from app.core.config import Settings

def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.model == "qwen2.5:7b"
    assert s.base_url == "http://localhost:11434/v1"
    assert s.api_key == "ollama"
    assert s.temperature == 0.0
    assert s.max_history_tokens == 4000
    assert s.max_history_messages == 10

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MODEL", "gpt-4o")
    s = Settings(_env_file=None)
    assert s.model == "gpt-4o"


def test_database_url_default():
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.database_url.startswith("mysql+pymysql://")
