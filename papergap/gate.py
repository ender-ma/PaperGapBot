"""Runs the specificity gate: static checks + corpus-size probe + optional LLM review."""
from __future__ import annotations

import logging
from typing import Callable

from .brief import Brief, Check, GateReport, corpus_check, static_checks
from .config import Settings
from .llm import LLMClient, review_brief
from .mining.text import normalize_phrase
from .sources import PaperSource, SourceError, make_source

log = logging.getLogger(__name__)


async def evaluate_gate(
    brief: Brief,
    settings: Settings,
    llm: LLMClient | None = None,
    source_factory: Callable[[str, Settings], PaperSource] = make_source,
) -> GateReport:
    rep = GateReport(fingerprint=brief.fingerprint(), checks=static_checks(brief),
                     min_score=settings.gate_min_score)

    if "openalex" in brief.sources and not settings.openalex_api_key:
        only = brief.sources == ["openalex"]
        rep.checks.append(Check(
            "OpenAlex API key configured", not only,
            "OpenAlex needs a free key (openalex.org/settings/api) in OPENALEX_API_KEY, "
            "or switch: /edit sources europepmc" if only else
            "No OpenAlex key: OpenAlex will stop after the small keyless allowance.",
            weight=0, hard=only,
        ))

    if rep.hard_failures:
        return rep  # don't spend API calls on a brief that fails basic checks

    primary = brief.sources[0]
    src = source_factory(primary, settings)
    try:
        count = await src.count(brief)
        rep.corpus_count = count
        rep.corpus_source = primary
        rep.checks.append(corpus_check(count, settings.min_corpus, settings.max_corpus, primary))
        if count > settings.max_corpus or count > brief.max_papers * 3:
            try:
                kws = await src.sample_keywords(brief, 100)
                have = {normalize_phrase(t) for t in brief.all_query_terms()}
                rep.narrowing_options = [
                    k for k, _ in kws if normalize_phrase(k) not in have and len(k) <= 40
                ][:8]
            except SourceError as exc:
                log.info("Keyword sample failed: %s", exc)
    except SourceError as exc:
        rep.checks.append(Check(
            "Corpus size in range", False, f"Could not reach {primary}: {exc}", weight=10, hard=True,
        ))
    finally:
        await src.aclose()

    if llm and settings.gate_llm_review:
        review = await review_brief(llm, brief)
        if review:
            rep.llm_score = review.get("score")
            bits = [review.get("verdict") or ""]
            if review.get("missing"):
                bits.append("Missing: " + "; ".join(map(str, review["missing"]))[:300])
            if review.get("sharper_focus"):
                bits.append("Sharper focus: " + str(review["sharper_focus"])[:300])
            rep.llm_advice = "\n".join(b for b in bits if b)
    return rep
