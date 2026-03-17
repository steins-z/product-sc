"""MiroFish configuration — loads from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, populated from environment variables."""

    # LLM provider — "openai" or "qwen" (DashScope OpenAI-compatible)
    llm_provider: str = "qwen"

    # API key (works for both OpenAI and Qwen/DashScope)
    llm_api_key: str = ""
    api_key: str = ""  # alias: also reads API_KEY from env

    # Base URL override — set for Qwen DashScope or other OpenAI-compatible APIs
    # Qwen default: https://dashscope.aliyuncs.com/compatible-mode/v1
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_base: str = ""  # alias: also reads API_BASE from env

    # Model name
    llm_model: str = "qwen-plus"

    # Legacy aliases (still read from env for backward compat)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Mock mode — when True, LLM calls return deterministic fake data
    use_mock_llm: bool = True

    # Database
    db_filename: str = "mirofish.db"

    # Chunking
    chunk_max_tokens: int = 1000
    chunk_overlap_tokens: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_api_key(self) -> str:
        """Return effective API key (prefer llm_api_key, then api_key, then openai_api_key)."""
        return self.llm_api_key or self.api_key or self.openai_api_key

    def get_model(self) -> str:
        """Return effective model name."""
        if self.llm_provider == "openai":
            return self.llm_model if self.llm_model != "qwen-plus" else self.openai_model
        return self.llm_model

    def get_base_url(self) -> str | None:
        """Return base URL — checks api_base, llm_base_url, or None for default OpenAI."""
        url = self.api_base or self.llm_base_url
        if self.llm_provider == "openai" and url == "https://dashscope.aliyuncs.com/compatible-mode/v1":
            return None  # use OpenAI default
        return url


settings = Settings()
