"""Settings loaded from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = _env(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _env_ids(name: str) -> set[int]:
    raw = _env(name, "") or ""
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            out.add(int(part))
    return out


@dataclass
class Settings:
    # Telegram
    telegram_token: str = ""
    allowed_user_ids: set[int] = field(default_factory=set)  # empty = anyone

    # Storage
    db_path: str = "papergap.db"
    reports_dir: str = "reports"
    keep_papers: bool = False  # delete harvested abstracts after each run to save disk

    # Paper sources
    openalex_api_key: str | None = None
    contact_email: str | None = None
    openalex_daily_budget_usd: float = 0.90  # free key = $1/day
    openalex_search_mode: str = "search"  # "search" or "filter"
    default_sources: tuple[str, ...] = ("openalex",)

    # Corpus size limits
    default_max_papers: int = 20_000
    hard_max_papers: int = 300_000
    min_corpus: int = 100
    max_corpus: int = 500_000

    # Specificity gate
    gate_min_score: int = 70
    gate_llm_review: bool = True

    # LLM
    llm_provider: str = "auto"  # auto | anthropic | openai | none
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    questions_per_run: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        sources = tuple(
            s.strip().lower()
            for s in (_env("DEFAULT_SOURCES", "openalex") or "openalex").split(",")
            if s.strip()
        )
        return cls(
            telegram_token=_env("TELEGRAM_BOT_TOKEN", "") or "",
            allowed_user_ids=_env_ids("ALLOWED_USER_IDS"),
            db_path=_env("DB_PATH", "papergap.db") or "papergap.db",
            reports_dir=_env("REPORTS_DIR", "reports") or "reports",
            keep_papers=_env_bool("KEEP_PAPERS", False),
            openalex_api_key=_env("OPENALEX_API_KEY"),
            contact_email=_env("CONTACT_EMAIL"),
            openalex_daily_budget_usd=_env_float("OPENALEX_DAILY_BUDGET_USD", 0.90),
            openalex_search_mode=(_env("OPENALEX_SEARCH_MODE", "search") or "search").lower(),
            default_sources=sources or ("openalex",),
            default_max_papers=_env_int("DEFAULT_MAX_PAPERS", 20_000),
            hard_max_papers=_env_int("HARD_MAX_PAPERS", 300_000),
            min_corpus=_env_int("MIN_CORPUS", 100),
            max_corpus=_env_int("MAX_CORPUS", 500_000),
            gate_min_score=_env_int("GATE_MIN_SCORE", 70),
            gate_llm_review=_env_bool("GATE_LLM_REVIEW", True),
            llm_provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_model=_env("LLM_MODEL", "claude-sonnet-5-5") or "claude-sonnet-5-5",
            openai_base_url=_env("OPENAI_BASE_URL"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL"),
            questions_per_run=_env_int("QUESTIONS_PER_RUN", 8),
        )
