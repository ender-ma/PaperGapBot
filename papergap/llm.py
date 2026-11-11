"""LLM layer: Claude (default), any OpenAI-compatible endpoint, or none.

The LLM never searches or counts; it only turns machine-detected gap signals
into well-formed research questions, and must cite the signal ids it used.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from .brief import Brief
from .config import Settings
from .mining.engine import KIND_LABELS, GapCandidate

log = logging.getLogger(__name__)

SYSTEM_RULES = """You are a senior research strategist who turns literature-gap signals into novel, \
answerable research questions.

Rules:
1. Use ONLY the gap signals provided. Every question must cite at least one signal id (e.g. "G3") in \
"signal_ids". Prefer questions that combine signals (e.g. an unexplored connection in an under-studied setting).
2. Be specific: name the population / organism / system, the variables, and a concrete study design.
3. Fit the researcher's profile, skills and resources. Do not propose work they cannot do with what they have \
unless you say what extra resource is needed.
4. Be honest about false gaps: synonyms the search missed, work published outside the indexed sources, or \
signals that come from abstracts only. Put this in "risks".
5. Never invent citations, numbers or findings. Quote numbers only from the signals.
6. Output valid JSON only, no prose before or after."""

QUESTION_SCHEMA = """{"questions": [{
  "title": "short title, max 12 words",
  "question": "the research question, one or two sentences",
  "signal_ids": ["G1", "G4"],
  "why_gap": "why this looks unanswered, citing the signal numbers",
  "novelty_angle": "what would be new",
  "approach": "study design and key methods",
  "data_and_resources": "data, samples, tools or collaborators needed",
  "feasibility": "high | medium | low",
  "risks": "why this might not be a real gap / main pitfalls",
  "check_terms": ["2 to 4 short search terms to test novelty"]
}]}"""


class LLMClient:
    name = "llm"

    async def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicClient(LLMClient):
    name = "claude"

    def __init__(self, api_key: str, model: str):
        from anthropic import AsyncAnthropic  # imported lazily: optional dependency

        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatClient(LLMClient):
    """Works with any /chat/completions endpoint (OpenAI, OpenRouter, Groq, Ollama ...)."""

    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str | None, model: str):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    async def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"] or ""


def make_llm(settings: Settings) -> LLMClient | None:
    p = settings.llm_provider
    if p == "none":
        return None
    if p in ("auto", "anthropic") and settings.anthropic_api_key:
        try:
            return AnthropicClient(settings.anthropic_api_key, settings.llm_model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Anthropic client unavailable: %s", exc)
    if p in ("auto", "openai") and settings.openai_base_url and settings.openai_model:
        return OpenAICompatClient(settings.openai_base_url, settings.openai_api_key, settings.openai_model)
    return None


def extract_json(text: str):
    """Parse JSON from a model reply that may include fences or stray prose."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a: b + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def brief_block(brief: Brief) -> str:
    return "\n".join(f"{k}: {v}" for k, v in brief.summary_lines())


def profile_block(profile: dict) -> str:
    if not profile:
        return "(no profile given)"
    labels = {
        "stage": "Career stage", "skills": "Skills", "resources": "Resources / lab access",
        "goals": "Goals", "avoid": "Avoid", "kinds": "Preferred gap types",
    }
    return "\n".join(f"{labels.get(k, k)}: {v}" for k, v in profile.items() if v)


async def review_brief(llm: LLMClient, brief: Brief) -> dict | None:
    system = (
        "You are a strict research methodologist. Before an automated literature-gap search runs, "
        "judge whether this brief defines a researchable NICHE (not a whole field). Reply with JSON only: "
        '{"score": 1-10, "verdict": "one sentence", "missing": ["..."], "sharper_focus": "a sharper focus line"}'
    )
    try:
        raw = await llm.complete(system, "BRIEF\n" + brief_block(brief), max_tokens=600)
        data = extract_json(raw)
        if isinstance(data, dict) and "score" in data:
            data["score"] = int(max(1, min(10, int(data.get("score", 5)))))
            return data
    except Exception as exc:  # noqa: BLE001
        log.warning("Brief review failed: %s", exc)
    return None


def candidates_block(cands: list[GapCandidate], max_ev: int = 3) -> str:
    lines = []
    for c in cands:
        lines.append(f"[{c.gid}] {c.kind} ({KIND_LABELS.get(c.kind, c.kind)}) — {c.title}")
        stats = {k: v for k, v in c.stats.items() if v not in (None, "", [])}
        if stats:
            lines.append(f"  stats: {json.dumps(stats, ensure_ascii=False)}")
        lines.append(f"  machine draft: {c.question}")
        for e in c.evidence[:max_ev]:
            snip = f' — "{e["snippet"][:220]}"' if e.get("snippet") else ""
            lines.append(f"  - ({e.get('year')}, cited {e.get('cites')}) {e.get('title', '')[:160]}{snip}")
    return "\n".join(lines)


def build_question_prompt(brief: Brief, profile: dict, custom: str, cands: list[GapCandidate],
                          corpus_line: str, n: int) -> str:
    parts = [
        "RESEARCH BRIEF", brief_block(brief), "",
        "RESEARCHER PROFILE", profile_block(profile), "",
    ]
    if custom:
        parts += ["RESEARCHER'S OWN INSTRUCTIONS (follow unless they break the rules)", custom, ""]
    parts += [
        "CORPUS", corpus_line, "",
        "GAP SIGNALS (machine-detected from titles/abstracts; each has an id)", candidates_block(cands), "",
        "TASK",
        f"Write the {n} strongest research questions. Rank them best first. Return JSON exactly in this shape:",
        QUESTION_SCHEMA,
    ]
    return "\n".join(parts)


async def write_questions(llm: LLMClient, brief: Brief, profile: dict, custom: str,
                          cands: list[GapCandidate], corpus_line: str, n: int) -> list[dict]:
    prompt = build_question_prompt(brief, profile, custom, cands, corpus_line, n)
    raw = await llm.complete(SYSTEM_RULES, prompt, max_tokens=6000)
    data = extract_json(raw)
    items = data.get("questions") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not items:
        raise ValueError("LLM returned no parseable questions")
    valid_ids = {c.gid for c in cands}
    out = []
    for q in items:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        ids = [i for i in (q.get("signal_ids") or []) if i in valid_ids]
        if not ids:
            continue  # hallucination guard: must rest on a real signal
        q["signal_ids"] = ids
        q["check_terms"] = [t for t in (q.get("check_terms") or []) if isinstance(t, str) and t.strip()][:4]
        q["source"] = llm.name
        out.append(q)
    return out[:n]


def stats_text(stats: dict) -> str:
    parts = []
    for k, v in stats.items():
        if v in (None, "", []) or k in ("level", "subtype"):
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        parts.append(f"{k.replace('_', ' ')} {v}")
    return "; ".join(parts)


def template_questions(brief: Brief, cands: list[GapCandidate], n: int) -> list[dict]:
    """No-LLM fallback: one question per top gap signal."""
    main_term = (brief.term_groups()[0][0] if brief.term_groups() else brief.focus)
    out = []
    for c in cands[:n]:
        check = list(dict.fromkeys([t for t in c.terms if t] + [main_term]))[:4]
        out.append({
            "title": c.title,
            "question": c.question,
            "signal_ids": [c.gid],
            "why_gap": KIND_LABELS.get(c.kind, c.kind) + ": " + stats_text(c.stats),
            "novelty_angle": "",
            "approach": c.approach,
            "data_and_resources": "",
            "feasibility": "",
            "risks": "Machine-generated from statistics; check synonyms and recent preprints.",
            "check_terms": check,
            "source": "template",
        })
    return out
