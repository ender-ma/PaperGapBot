"""Novelty check: search each proposed question's key terms together."""
from __future__ import annotations

import asyncio
import logging

from .sources import BudgetExhausted, PaperSource, SourceError

log = logging.getLogger(__name__)


def novelty_label(count: int) -> str:
    if count == 0:
        return "No direct match found"
    if count <= 5:
        return "Sparse prior work"
    if count <= 30:
        return "Some prior work: differentiate"
    return "Crowded: likely already studied"


async def check_novelty(source: PaperSource, questions: list[dict], delay_s: float = 0.2) -> None:
    """Adds q['novelty'] = {count, label, query, nearest[]} in place."""
    for q in questions:
        terms = [t for t in q.get("check_terms") or [] if t.strip()]
        if len(terms) < 2:
            q["novelty"] = {"label": "Not checked (too few terms)", "count": None, "query": terms, "nearest": []}
            continue
        try:
            count, papers = await source.quick_search(terms, n=3)
            q["novelty"] = {
                "count": count,
                "label": novelty_label(count),
                "query": terms,
                "source": source.name,
                "nearest": [{"title": p.title, "year": p.year, "url": p.url} for p in papers],
            }
        except (SourceError, BudgetExhausted) as exc:
            log.info("Novelty check failed: %s", exc)
            q["novelty"] = {"label": "Not checked (source unavailable)", "count": None, "query": terms, "nearest": []}
        await asyncio.sleep(delay_s)
