"""Central configuration, read from environment variables (with a .env file supported).

Every tunable value has a sensible default so the project runs out of the box;
only GROQ_API_KEY is required, and only for answer generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if present. Existing environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All runtime settings for the pipeline."""

    policy_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    llm_model: str
    llm_temperature: float
    groq_api_key: str | None

    def require_groq_key(self) -> str:
        """Return the Groq API key or raise a clear, secret-free error."""
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
            )
        return self.groq_api_key


def load_settings() -> Settings:
    """Build Settings from the environment and validate them."""
    settings = Settings(
        policy_dir=Path(os.getenv("POLICY_DIR", PROJECT_ROOT / "data" / "policies")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / ".chroma")),
        collection_name=os.getenv("CHROMA_COLLECTION", "policy_chunks"),
        chunk_size=_int_env("CHUNK_SIZE", 800),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 150),
        top_k=_int_env("TOP_K", 4),
        min_similarity=_float_env("MIN_SIMILARITY", 0.25),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.0),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
    )
    if settings.chunk_size <= 0:
        raise ConfigurationError("CHUNK_SIZE must be positive")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    if settings.top_k <= 0:
        raise ConfigurationError("TOP_K must be positive")
    return settings
