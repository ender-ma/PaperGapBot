"""The research brief and the specificity gate.

The bot refuses to start screening until a brief is specific enough. The gate has
two layers:

* static checks on the brief itself (field, niche, focus, key-term groups ...)
* a corpus-size probe against the primary paper source (too broad / too narrow)

An optional LLM reviewer adds advice but never overrides the hard checks.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

from .mining.text import STOPWORDS

# Words that on their own describe a whole field rather than a niche.
GENERIC_WORDS = {
    "ai", "artificial", "intelligence", "machine", "learning", "deep", "data", "science",
    "health", "healthcare", "medicine", "medical", "disease", "diseases", "cancer",
    "biology", "biological", "chemistry", "physics", "engineering", "technology",
    "climate", "change", "environment", "environmental", "education", "economics",
    "economy", "society", "social", "policy", "management", "business", "research",
    "analysis", "system", "systems", "model", "models", "method", "methods", "study",
    "application", "applications", "impact", "effect", "effects", "development",
    "review", "overview", "general", "various", "different", "new", "novel", "modern",
    "world", "global", "human", "humans", "people", "public", "network", "networks",
    "algorithm", "algorithms", "computer", "computing", "software", "internet",
    "agriculture", "food", "energy", "water", "materials", "material", "genetics",
    "genomics", "microbiology", "psychology", "mental", "nutrition", "infection",
    "infections", "treatment", "therapy", "drug", "drugs", "patients", "children",
}

BRIEF_FIELDS = [
    "field", "niche", "focus", "key_terms", "context", "methods", "exclude",
    "years", "sources", "max_papers",
]

FIELD_HELP = {
    "field": "Broad discipline, e.g. Microbiology",
    "niche": "Sub-field you work in, e.g. Antimicrobial resistance in food animals",
    "focus": "The specific topic, e.g. plasmid-mediated colistin resistance (mcr genes) in poultry",
    "key_terms": "Concept groups that define the corpus. Commas = AND, | = synonyms. "
                 "e.g. colistin|polymyxin, mcr|mcr-1, poultry|chicken|broiler",
    "context": "Settings to test for coverage (optional). e.g. Nigeria|Nigerian, West Africa|sub-Saharan Africa",
    "methods": "Methods you could apply (optional). e.g. whole genome sequencing|WGS, metagenomics",
    "exclude": "Terms to exclude (optional). e.g. veterinary vaccine, retracted",
    "years": "Year range, e.g. 2012-2026",
    "sources": "Comma list from: openalex, europepmc, arxiv",
    "max_papers": "How many papers to screen, e.g. 50000",
}


def split_groups(raw: str) -> list[list[str]]:
    """'a|b, c' -> [['a','b'], ['c']]. Commas/semicolons/newlines separate groups."""
    groups: list[list[str]] = []
    for chunk in re.split(r"[,;\n]+", raw or ""):
        terms = [t.strip().strip('"').strip("'").strip() for t in chunk.split("|")]
        terms = [t for t in terms if t]
        if terms:
            groups.append(terms)
    return groups


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]*", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass
class Brief:
    field: str = ""
    niche: str = ""
    focus: str = ""
    key_terms: str = ""
    context: str = ""
    methods: str = ""
    exclude: str = ""
    year_from: int | None = None
    year_to: int | None = None
    sources: list[str] = dc_field(default_factory=lambda: ["openalex"])
    max_papers: int = 20_000

    # ---- parsing helpers -------------------------------------------------
    def term_groups(self) -> list[list[str]]:
        return split_groups(self.key_terms)

    def context_groups(self) -> list[list[str]]:
        return split_groups(self.context)

    def method_groups(self) -> list[list[str]]:
        return split_groups(self.methods)

    def exclude_terms(self) -> list[str]:
        return [t for g in split_groups(self.exclude) for t in g]

    def all_query_terms(self) -> list[str]:
        return [t for g in self.term_groups() for t in g]

    def label(self) -> str:
        return self.focus or self.niche or self.field or "(empty brief)"

    # ---- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Brief":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def set_field(self, name: str, value: str) -> str | None:
        """Set a field from user text. Returns an error message or None."""
        name = name.lower().strip().replace("-", "_")
        value = (value or "").strip()
        if name in {"field", "niche", "focus", "key_terms", "context", "methods", "exclude"}:
            setattr(self, name, value)
            return None
        if name in {"keyterms", "terms", "keys"}:
            self.key_terms = value
            return None
        if name in {"years", "year"}:
            m = re.match(r"^\s*(\d{4})\s*[-–to ]+\s*(\d{4})\s*$", value)
            if not m:
                return "Use a range like 2012-2026"
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            self.year_from, self.year_to = a, b
            return None
        if name == "sources":
            allowed = {"openalex", "europepmc", "arxiv"}
            items = [s.strip().lower() for s in re.split(r"[,\s]+", value) if s.strip()]
            bad = [s for s in items if s not in allowed]
            if bad or not items:
                return "Sources must be from: openalex, europepmc, arxiv"
            self.sources = list(dict.fromkeys(items))
            return None
        if name in {"max_papers", "max", "papers"}:
            digits = re.sub(r"[^\d]", "", value)
            if not digits:
                return "Give a number, e.g. 50000"
            self.max_papers = max(50, int(digits))
            return None
        return f"Unknown field '{name}'. Fields: {', '.join(BRIEF_FIELDS)}"

    def summary_lines(self) -> list[tuple[str, str]]:
        years = (
            f"{self.year_from}-{self.year_to}" if self.year_from and self.year_to else "all years"
        )
        return [
            ("Field", self.field or "—"),
            ("Niche", self.niche or "—"),
            ("Focus", self.focus or "—"),
            ("Key terms", self.key_terms or "—"),
            ("Context", self.context or "—"),
            ("Methods", self.methods or "—"),
            ("Exclude", self.exclude or "—"),
            ("Years", years),
            ("Sources", ", ".join(self.sources)),
            ("Max papers", f"{self.max_papers:,}"),
        ]


# ---------------------------------------------------------------------------
# Specificity gate
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    weight: int
    hard: bool


@dataclass
class GateReport:
    fingerprint: str
    checks: list[Check] = dc_field(default_factory=list)
    corpus_count: int | None = None
    corpus_source: str | None = None
    narrowing_options: list[str] = dc_field(default_factory=list)
    llm_advice: str | None = None
    llm_score: int | None = None
    min_score: int = 70

    @property
    def score(self) -> int:
        total = sum(c.weight for c in self.checks) or 1
        got = sum(c.weight for c in self.checks if c.ok)
        return round(100 * got / total)

    @property
    def hard_failures(self) -> list[Check]:
        return [c for c in self.checks if c.hard and not c.ok]

    @property
    def probed(self) -> bool:
        return self.corpus_count is not None

    @property
    def passed(self) -> bool:
        return self.probed and not self.hard_failures and self.score >= self.min_score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = self.score
        d["passed"] = self.passed
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "GateReport | None":
        if not d:
            return None
        checks = [Check(**c) for c in d.get("checks", [])]
        return cls(
            fingerprint=d.get("fingerprint", ""),
            checks=checks,
            corpus_count=d.get("corpus_count"),
            corpus_source=d.get("corpus_source"),
            narrowing_options=d.get("narrowing_options", []),
            llm_advice=d.get("llm_advice"),
            llm_score=d.get("llm_score"),
            min_score=d.get("min_score", 70),
        )


def static_checks(brief: Brief) -> list[Check]:
    checks: list[Check] = []
    field_words = set(content_words(brief.field))
    niche_words = content_words(brief.niche)
    focus_words = content_words(brief.focus)
    groups = brief.term_groups()

    checks.append(Check(
        "Field given", bool(field_words), "Say which discipline this is." if not field_words else brief.field,
        weight=10, hard=True,
    ))

    niche_new = [w for w in niche_words if w not in field_words]
    niche_ok = len(niche_words) >= 2 and len(niche_new) >= 1
    checks.append(Check(
        "Niche narrows the field",
        niche_ok,
        "Niche must be a sub-area of the field in at least 2 words, not a restatement of it."
        if not niche_ok else brief.niche,
        weight=15, hard=True,
    ))

    focus_ok = len(focus_words) >= 3
    checks.append(Check(
        "Focus is specific (3+ content words)",
        focus_ok,
        f"Focus has {len(focus_words)} content word(s); describe the exact topic."
        if not focus_ok else brief.focus,
        weight=15, hard=True,
    ))

    groups_ok = len(groups) >= 2
    checks.append(Check(
        "At least 2 key-term groups (intersection)",
        groups_ok,
        "One concept alone describes a field. Add a second group, e.g. 'colistin|polymyxin, poultry|chicken'."
        if not groups_ok else f"{len(groups)} groups: " + " AND ".join("(" + " OR ".join(g) + ")" for g in groups),
        weight=15, hard=True,
    ))

    generic_share = (
        sum(1 for w in focus_words if w in GENERIC_WORDS) / len(focus_words) if focus_words else 1.0
    )
    checks.append(Check(
        "Focus is not mostly generic words",
        generic_share < 0.5,
        f"{round(generic_share * 100)}% of focus words are generic (e.g. 'AI', 'health', 'cancer')."
        if generic_share >= 0.5 else "OK",
        weight=10, hard=False,
    ))

    specific_terms = set(focus_words) | {t.lower() for g in groups for t in g}
    specific_terms -= GENERIC_WORDS
    checks.append(Check(
        "Enough specific terms (5+)",
        len(specific_terms) >= 5,
        f"{len(specific_terms)} specific terms across focus and key terms.",
        weight=10, hard=False,
    ))

    has_dims = bool(brief.context_groups() or brief.method_groups())
    checks.append(Check(
        "Context or methods given",
        has_dims,
        "Add a context (place, population, organism) or methods so the bot can test coverage gaps."
        if not has_dims else "OK",
        weight=10, hard=False,
    ))

    span_ok = True
    detail = "All years"
    if brief.year_from and brief.year_to:
        span = brief.year_to - brief.year_from + 1
        span_ok = 3 <= span <= 30
        detail = f"{span} year window" + ("" if span_ok else " (use 3-30 years)")
    checks.append(Check("Sensible year window", span_ok, detail, weight=5, hard=False))
    return checks


def corpus_check(count: int, min_corpus: int, max_corpus: int, source: str) -> Check:
    if count < min_corpus:
        return Check(
            "Corpus size in range", False,
            f"Only {count:,} papers on {source}. Too narrow to find gaps statistically "
            f"(need ≥ {min_corpus:,}). Add synonyms or drop a key-term group.",
            weight=10, hard=True,
        )
    if count > max_corpus:
        return Check(
            "Corpus size in range", False,
            f"{count:,} papers on {source}. Too broad for a niche (max {max_corpus:,}). "
            "Add a key-term group or tighten the focus.",
            weight=10, hard=True,
        )
    return Check("Corpus size in range", True, f"{count:,} matching papers on {source}.", weight=10, hard=True)
