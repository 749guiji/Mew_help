from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    temperature: float = 0.0
    max_history_tokens: int = 4000
    max_history_messages: int = 10
    database_url: str = "mysql+pymysql://root:123456@127.0.0.1:3306/mew_help"
    milvus_uri: str = "./data/milvus.db"
    milvus_collection: str = "knowledge"
    embedding_model: str = "bge-m3"
    embedding_base_url: str = "http://localhost:11434"
    embedding_dim: int = 1024
    retrieval_top_k: int = 1
    retrieval_threshold: float = 0.5
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 10
    hybrid_dense_top_k: int = 50
    hybrid_sparse_top_k: int = 50
    rerank_use_fp16: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
