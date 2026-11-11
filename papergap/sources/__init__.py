from __future__ import annotations

from ..config import Settings
from .arxiv import ArxivSource
from .base import BudgetExhausted, Paper, PaperSource, SourceError
from .europepmc import EuropePMCSource
from .openalex import OpenAlexSource

SOURCE_NAMES = ("openalex", "europepmc", "arxiv")


def make_source(name: str, settings: Settings) -> PaperSource:
    name = name.lower()
    if name == "openalex":
        return OpenAlexSource(
            api_key=settings.openalex_api_key,
            email=settings.contact_email,
            budget_usd=settings.openalex_daily_budget_usd,
            mode=settings.openalex_search_mode,
        )
    if name == "europepmc":
        return EuropePMCSource(email=settings.contact_email)
    if name == "arxiv":
        return ArxivSource()
    raise ValueError(f"Unknown source: {name}")


def novelty_source_name(settings: Settings, brief_sources: list[str]) -> str:
    """Pick the source used for novelty checks: OpenAlex if a key exists, else Europe PMC/arXiv."""
    if settings.openalex_api_key:
        return "openalex"
    for s in brief_sources:
        if s != "openalex":
            return s
    return "europepmc"


__all__ = [
    "BudgetExhausted", "Paper", "PaperSource", "SourceError", "make_source",
    "novelty_source_name", "SOURCE_NAMES",
]
