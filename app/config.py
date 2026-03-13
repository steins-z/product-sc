"""MiroFish configuration — loads from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, populated from environment variables."""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Mock mode — when True, LLM calls return deterministic fake data
    use_mock_llm: bool = True

    # Chunking
    chunk_max_tokens: int = 1000
    chunk_overlap_tokens: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
