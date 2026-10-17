"""Settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "var" / "claimpilot.db"))))

    # "openai" | "bedrock" | "mock". "auto" picks openai when a key is set, otherwise mock.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4.1-mini"))
    openai_embedding_model: str = field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    bedrock_model: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "30")))

    # Claims above this amount always go to a senior handler, whatever the recommendation.
    senior_review_threshold: int = field(default_factory=lambda: int(_env("SENIOR_REVIEW_THRESHOLD", "300000")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def resolved_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openai" if os.environ.get("OPENAI_API_KEY") else "mock"


def get_settings() -> Settings:
    return Settings()
