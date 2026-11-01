"""Environment-driven configuration for the Temporal & State-Aware Semantic Memory Buffer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, real env vars win."""
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception:
        pass


@dataclass(frozen=True)
class Config:
    buffer_db: str
    gemini_api_key: str
    embed_model: str
    embed_dim: int
    gemini_base_url: str
    default_decay_lambda: float = 0.05
    request_timeout: float = 30.0

    @property
    def db_path(self) -> Path:
        return Path(self.buffer_db)

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            buffer_db=os.environ.get("SM_BUFFER_DB", "./memory_buffer.db"),
            gemini_api_key=os.environ.get("SM_GEMINI_API_KEY", ""),
            embed_model=os.environ.get("SM_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(os.environ.get("SM_EMBED_DIM", "768")),
            gemini_base_url=os.environ.get("SM_GEMINI_BASE_URL", GEMINI_BASE),
            default_decay_lambda=float(os.environ.get("SM_DEFAULT_DECAY_LAMBDA", "0.05")),
            request_timeout=float(os.environ.get("SM_REQUEST_TIMEOUT", "30.0")),
        )
