"""One screening job: harvest -> mine -> write questions -> novelty check -> report."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from .brief import Brief
from .config import Settings
from .llm import LLMClient, template_questions, write_questions
from .mining.engine import KIND_LABELS, mine
from .novelty import check_novelty
from .report import write_files
from .sources import BudgetExhausted, PaperSource, SourceError, make_source, novelty_source_name
from .sources.openalex import OpenAlexSource
from .storage import Store

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


@dataclass
class JobProgress:
    stage: str = "Starting"
    frac: float = 0.0
    papers: int = 0
    target: int = 0
    started: float = field(default_factory=time.time)
    cancelled: bool = False

    def line(self) -> str:
        mins = int((time.time() - self.started) // 60)
        bar_n = int(self.frac * 10)
        bar = "▓" * bar_n + "░" * (10 - bar_n)
        papers = f" · {self.papers:,}/{self.target:,} papers" if self.target else ""
        return f"{self.stage}\n{bar} {int(self.frac * 100)}%{papers} · {mins} min"


def profile_prefs(profile: dict) -> dict:
    def kinds(text: str) -> list[str]:
        out = []
        for part in (text or "").replace(";", ",").split(","):
            p = part.strip().upper()
            if p in KIND_LABELS:
                out.append(p)
        return out

    return {"focus_kinds": kinds(profile.get("kinds", "")), "avoid_kinds": kinds(profile.get("avoid_kinds", ""))}


async def harvest(job_id: int, brief: Brief, settings: Settings, store: Store, progress: JobProgress,
                  source_factory: Callable[[str, Settings], PaperSource]) -> tuple[dict, list[str]]:
    stats: dict = {}
    notes: list[str] = []
    target = min(brief.max_papers, settings.hard_max_papers)
    progress.target = target
    have = store.count_papers(job_id)
    for name in brief.sources:
        remaining = target - have
        if remaining <= 0:
            break
        src = source_factory(name, settings)
        progress.stage = f"Fetching papers from {name}"
        try:
            if isinstance(src, OpenAlexSource):
                left = await src.remaining_budget()
                if left is not None:
                    src.budget_usd = min(src.budget_usd, max(0.0, left * 0.95))
                    stats["openalex_budget_left_usd"] = round(left, 3)
            async for batch in src.harvest(brief, remaining, should_stop=lambda: progress.cancelled):
                have += store.add_papers(job_id, batch)
                progress.papers = have
                progress.frac = min(1.0, have / max(target, 1))
        except BudgetExhausted as exc:
            notes.append(f"{exc} Continued with the {have:,} papers fetched so far.")
        except SourceError as exc:
            notes.append(f"{name} failed: {exc}")
            log.warning("Source %s failed: %s", name, exc)
        finally:
            if isinstance(src, OpenAlexSource):
                stats["openalex_cost_usd"] = round(src.spent_usd, 4)
            await src.aclose()
        if progress.cancelled:
            raise JobCancelled()
    return stats, notes


async def run_job(job_id: int, brief: Brief, profile: dict, prompt_extra: str, settings: Settings,
                  store: Store, progress: JobProgress, llm: LLMClient | None = None,
                  source_factory: Callable[[str, Settings], PaperSource] = make_source) -> dict:
    stats, notes = await harvest(job_id, brief, settings, store, progress, source_factory)
    n = store.count_papers(job_id)
    if n == 0:
        raise RuntimeError("No papers were retrieved. " + " ".join(notes))

    def prog(stage: str, frac: float) -> None:
        progress.stage = stage
        progress.frac = frac
        if progress.cancelled:
            raise JobCancelled()

    progress.target = 0
    result = await asyncio.to_thread(
        mine, lambda: store.iter_papers(job_id), n, brief, profile_prefs(profile), prog
    )
    notes += result.notes
    cands = result.candidates

    progress.stage, progress.frac = "Writing research questions", 0.92
    srcs = ", ".join(f"{k} {v:,}" for k, v in result.sources.items())
    corpus_line = (f"{result.n_docs:,} papers ({srcs}), published {result.year_min}-{result.year_max}; "
                   f"{len(cands)} top gap signals below out of {result.stats.get('candidates_total')} detected.")
    questions: list[dict] = []
    writer = "templates (no LLM configured)"
    if llm and cands:
        try:
            questions = await write_questions(llm, brief, profile, prompt_extra, cands, corpus_line,
                                              settings.questions_per_run)
            writer = llm.name
        except Exception as exc:  # noqa: BLE001
            notes.append(f"LLM question writing failed ({exc}); fell back to templates.")
            log.exception("LLM failed")
    if not questions:
        questions = template_questions(brief, cands, settings.questions_per_run)
        if llm:
            writer = "templates (LLM fallback)"

    progress.stage, progress.frac = "Checking novelty of each question", 0.96
    nsrc = source_factory(novelty_source_name(settings, brief.sources), settings)
    try:
        await check_novelty(nsrc, questions)
    finally:
        await nsrc.aclose()

    stats.update({
        "n_docs": result.n_docs, "n_vocab": result.n_vocab, "sources": result.sources,
        "year_min": result.year_min, "year_max": result.year_max, "question_writer": writer,
        "notes": notes, **result.stats,
    })
    gaps = [c.to_dict() for c in cands]
    job = store.get_job(job_id) or {"job_id": job_id}
    html_path, csv_path = write_files(settings.reports_dir, job_id, brief, job, stats, questions, gaps, notes)
    store.save_results(job_id, gaps, questions)
    store.update_job(job_id, status="done", finished_at=time.time(), stats=stats,
                     report_path=html_path, csv_path=csv_path)
    if not settings.keep_papers:
        store.delete_papers(job_id)
    progress.stage, progress.frac = "Done", 1.0
    return {"stats": stats, "gaps": gaps, "questions": questions, "notes": notes,
            "html": html_path, "csv": csv_path}


async def offline_report(papers: list, brief: Brief, settings: Settings, path: str,
                         llm: LLMClient | None = None) -> str:
    """Mine an in-memory list of papers and write an HTML report (used by the demo)."""
    result = await asyncio.to_thread(mine, lambda: iter(papers), len(papers), brief, {}, None)
    cands = result.candidates
    questions: list[dict] = []
    writer = "templates"
    if llm:
        try:
            questions = await write_questions(llm, brief, {}, "", cands, f"{result.n_docs} papers",
                                              settings.questions_per_run)
            writer = llm.name
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM failed in demo: %s", exc)
    if not questions:
        questions = template_questions(brief, cands, settings.questions_per_run)
    stats = {"n_docs": result.n_docs, "sources": {"synthetic demo": result.n_docs},
             "year_min": result.year_min, "year_max": result.year_max, "question_writer": writer,
             **result.stats}
    from .report import render_html

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(brief, {"job_id": "demo"}, stats, questions, [c.to_dict() for c in cands],
                            result.notes + ["DEMO: synthetic papers with planted gaps, not real literature."]))
    return path
