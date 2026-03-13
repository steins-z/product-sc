"""MiroFish configuration — loads from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, populated from environment variables."""

    # LLM provider — "openai" or "qwen" (DashScope OpenAI-compatible)
    llm_provider: str = "qwen"

    # API key (works for both OpenAI and Qwen/DashScope)
    llm_api_key: str = ""

    # Base URL override — set for Qwen DashScope or other OpenAI-compatible APIs
    # Qwen default: https://dashscope.aliyuncs.com/compatible-mode/v1
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Model name
    llm_model: str = "qwen-plus"

    # Legacy aliases (still read from env for backward compat)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Mock mode — when True, LLM calls return deterministic fake data
    use_mock_llm: bool = True

    # Chunking
    chunk_max_tokens: int = 1000
    chunk_overlap_tokens: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_api_key(self) -> str:
        """Return effective API key (prefer llm_api_key, fallback to openai_api_key)."""
        return self.llm_api_key or self.openai_api_key

    def get_model(self) -> str:
        """Return effective model name."""
        if self.llm_provider == "openai":
            return self.llm_model if self.llm_model != "qwen-plus" else self.openai_model
        return self.llm_model

    def get_base_url(self) -> str | None:
        """Return base URL — None for default OpenAI, DashScope URL for Qwen."""
        if self.llm_provider == "openai" and self.llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1":
            return None  # use OpenAI default
        return self.llm_base_url


settings = Settings()
