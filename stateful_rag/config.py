from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"

# Absolute path to chroma_db — works regardless of where app is launched from
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "DocMind"
    log_level: str = "INFO"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"

    # Embeddings
    embedding_model: str = "nomic-embed-text"

    # ChromaDB — absolute path so it works from any working directory
    chroma_persist_dir: str = str(CHROMA_DIR)

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()