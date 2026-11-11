"""Gap-mining engine.

Works on titles + abstracts (+ metadata keywords) and scales to a few hundred
thousand papers on a laptop/VPS:

pass 1  build a vocabulary of candidate terms from a sample (<=15k docs)
pass 2  stream every paper once: term index, design cues, contexts, methods,
        author-stated gap / contradiction sentences
then    sparse matrix algebra (scipy) for co-occurrence, coverage and trends,
        eight detectors, and a ranking step that keeps the list diverse.

Every gap is a *signal* backed by counts and example papers, not a verdict.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import numpy as np
from scipy import sparse

from ..brief import Brief
from .cues import (
    AUTO_METHODS, CONTRA_RX, GAP_RX, REVIEW_TITLE, STRONG_CLINICAL, STRONG_INVIVO,
    WEAK_OBSERVATIONAL, WEAK_PRECLINICAL,
)
from .text import (
    candidate_ngrams, compile_group_regex, fold, is_content, normalize_phrase, sentences,
    shorten, tokenize,
)

log = logging.getLogger(__name__)

KIND_LABELS = {
    "EXPLICIT": "Authors flag it as unknown",
    "CONTRADICTION": "Unresolved conflicting findings",
    "BRIDGE": "Unexplored connection",
    "CONTEXT": "Under-studied setting",
    "METHOD": "Method not yet applied",
    "EMERGING": "Rising but thin",
    "DORMANT": "Highly cited, gone quiet",
    "EVIDENCE": "Weak or unsynthesised evidence",
}
DEFAULT_KIND_WEIGHTS = {
    "EXPLICIT": 1.0, "BRIDGE": 1.0, "CONTEXT": 1.1, "METHOD": 0.95, "EVIDENCE": 0.95,
    "CONTRADICTION": 0.9, "EMERGING": 0.9, "DORMANT": 0.8,
}

SAMPLE_MAX = 15_000
PRUNE_EVERY = 5_000
V_MULTI = 2_500
V_UNI = 900
MAX_STATEMENTS_PER_DOC = 3

ProgressFn = Callable[[str, float], None]


@dataclass
class GapCandidate:
    kind: str
    title: str
    terms: list[str]
    score: float
    question: str
    approach: str
    evidence: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    gid: str = ""
    final: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GapCandidate":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MiningResult:
    n_docs: int
    n_vocab: int
    year_min: int | None
    year_max: int | None
    sources: dict
    candidates: list[GapCandidate]
    notes: list[str]
    stats: dict


# ---------------------------------------------------------------------------
# Pass 1: vocabulary
# ---------------------------------------------------------------------------

def _doc_text(p) -> str:
    return f"{p.title or ''}. {p.abstract or ''}"


def build_vocab(docs_factory: Callable[[], Iterable], n_total: int, query_keys: set[str],
                progress: ProgressFn | None = None) -> tuple[list[str], int]:
    stride = max(1, math.ceil(n_total / SAMPLE_MAX))
    df: Counter = Counter()
    n_s = 0
    for i, p in enumerate(docs_factory()):
        if i % stride:
            continue
        keys = {k for k, _ in candidate_ngrams(_doc_text(p))}
        for kw in p.keywords or []:
            nk = normalize_phrase(kw)
            if nk:
                keys.add(nk)
        df.update(keys)
        n_s += 1
        if n_s % PRUNE_EVERY == 0:
            for k in [k for k, c in df.items() if c == 1]:
                del df[k]
        if progress and n_s % 500 == 0:
            progress("Building vocabulary", min(1.0, n_s * stride / max(n_total, 1)))
    if n_s == 0:
        return [], 0
    min_df = max(3, math.ceil(0.0008 * n_s))
    max_df = 0.30 * n_s
    multi, uni = [], []
    for k, c in df.items():
        if c < min_df or c > max_df or k in query_keys or k.replace(" ", "").isdigit():
            continue
        (multi if " " in k else uni).append((c, k))
    multi.sort(reverse=True)
    uni.sort(reverse=True)
    sel = {k: c for c, k in multi[:V_MULTI] + uni[:V_UNI]}
    # Drop sub-phrases that almost always appear inside a longer phrase.
    for k in sorted(sel, key=lambda x: -len(x.split())):
        if k not in sel:
            continue
        toks = k.split()
        n = len(toks)
        for size in range(1, n):
            for start in range(0, n - size + 1):
                sub = " ".join(toks[start:start + size])
                if sub in sel and sel[k] >= 0.8 * sel[sub]:
                    del sel[sub]
    vocab = sorted(sel, key=lambda x: -sel[x])
    return vocab, n_s


# ---------------------------------------------------------------------------
# Pass 2: scan
# ---------------------------------------------------------------------------

class Corpus:
    """Everything the detectors need, as arrays."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.surfaces: dict[int, str] = {}
        self.docs: list[tuple] = []  # (pid, title, year, cites, url, source)
        self.X: sparse.csr_matrix | None = None
        self.Xc: sparse.csc_matrix | None = None
        self.df: np.ndarray | None = None
        self.years: np.ndarray | None = None
        self.cites: np.ndarray | None = None
        self.flags: dict[str, np.ndarray] = {}
        self.ctx_labels: list[str] = []
        self.ctx: sparse.csr_matrix | None = None
        self.meth_labels: list[str] = []
        self.meth_user: list[bool] = []
        self.meth: sparse.csr_matrix | None = None
        self.statements: list[tuple[int, str, str]] = []
        self.source_counts: Counter = Counter()
        self.ok: np.ndarray | None = None  # False for terms that are cue phrases / contexts / methods

    @property
    def N(self) -> int:
        return len(self.docs)

    @property
    def V(self) -> int:
        return len(self.vocab)

    def label(self, j: int) -> str:
        return self.surfaces.get(j) or self.vocab[j]

    def docs_with(self, j: int) -> np.ndarray:
        return self.Xc.indices[self.Xc.indptr[j]: self.Xc.indptr[j + 1]]

    def docs_with_all(self, js: list[int]) -> np.ndarray:
        out = None
        for j in js:
            d = self.docs_with(j)
            out = d if out is None else np.intersect1d(out, d, assume_unique=True)
        return out if out is not None else np.array([], dtype=np.int32)

    def evidence(self, doc_ids, k: int = 3, order: str = "cited", snippets: dict | None = None) -> list[dict]:
        ids = list(doc_ids)
        if not ids:
            return []
        if order == "recent":
            ids.sort(key=lambda d: (self.docs[d][2] or 0, self.docs[d][3]), reverse=True)
        else:
            ids.sort(key=lambda d: (self.docs[d][3], self.docs[d][2] or 0), reverse=True)
        out = []
        for d in ids[:k]:
            pid, title, year, cites, url, source = self.docs[d]
            item = {"pid": pid, "title": title, "year": year, "cites": cites, "url": url, "source": source}
            if snippets and d in snippets:
                item["snippet"] = snippets[d]
            out.append(item)
        return out


def scan(docs_factory: Callable[[], Iterable], vocab: list[str], brief: Brief, n_total: int,
         progress: ProgressFn | None = None) -> Corpus:
    cp = Corpus()
    cp.vocab = vocab
    index = {k: i for i, k in enumerate(vocab)}

    ctx_groups = brief.context_groups()
    cp.ctx_labels = [g[0] for g in ctx_groups]
    ctx_rx = [compile_group_regex(g) for g in ctx_groups]

    user_methods = brief.method_groups()
    user_tokens = {fold(t) for g in user_methods + brief.term_groups() for term in g for t in tokenize(term)}
    meth_specs: list[tuple[str, list[str], bool]] = [(g[0], g, True) for g in user_methods]
    for label, terms in AUTO_METHODS:
        mt = {fold(t) for term in terms for t in tokenize(term)}
        if mt & user_tokens:
            continue  # the user already covers this method (or it defines the corpus)
        meth_specs.append((label, terms, False))
    cp.meth_labels = [m[0] for m in meth_specs]
    cp.meth_user = [m[2] for m in meth_specs]
    meth_rx = [compile_group_regex(m[1]) for m in meth_specs]

    indptr = array("q", [0])
    indices = array("i")
    years = array("i")
    cites = array("i")
    f_review, f_pre, f_vivo, f_clin, f_obs = (array("b") for _ in range(5))
    ctx_rows, ctx_cols, m_rows, m_cols = array("i"), array("i"), array("i"), array("i")

    for d, p in enumerate(docs_factory()):
        text = _doc_text(p)
        ids: set[int] = set()
        for key, surface in candidate_ngrams(text):
            j = index.get(key)
            if j is not None:
                ids.add(j)
                if j not in cp.surfaces:
                    cp.surfaces[j] = surface
        for kw in p.keywords or []:
            j = index.get(normalize_phrase(kw))
            if j is not None:
                ids.add(j)
                cp.surfaces.setdefault(j, kw.strip().lower())
        indices.extend(sorted(ids))
        indptr.append(len(indices))

        years.append(int(p.year or 0))
        cites.append(int(p.cited_by or 0))
        title = p.title or ""
        f_review.append(1 if (p.ptype == "review" or REVIEW_TITLE.search(title)) else 0)
        f_pre.append(1 if WEAK_PRECLINICAL.search(text) else 0)
        f_vivo.append(1 if STRONG_INVIVO.search(text) else 0)
        f_clin.append(1 if STRONG_CLINICAL.search(text) else 0)
        f_obs.append(1 if WEAK_OBSERVATIONAL.search(text) else 0)
        for k, rx in enumerate(ctx_rx):
            if rx is not None and rx.search(text):
                ctx_rows.append(d)
                ctx_cols.append(k)
        for k, rx in enumerate(meth_rx):
            if rx is not None and rx.search(text):
                m_rows.append(d)
                m_cols.append(k)

        found = 0
        for sent in sentences(p.abstract or ""):
            if found >= MAX_STATEMENTS_PER_DOC:
                break
            if GAP_RX.search(sent):
                cp.statements.append((d, "EXPLICIT", shorten(sent, 400)))
                found += 1
            elif CONTRA_RX.search(sent):
                cp.statements.append((d, "CONTRADICTION", shorten(sent, 400)))
                found += 1

        cp.docs.append((p.pid, title, p.year, int(p.cited_by or 0), p.url, p.source))
        cp.source_counts[p.source] += 1
        if progress and d % 2000 == 0:
            progress("Scanning papers", min(1.0, d / max(n_total, 1)))

    N, V = len(cp.docs), len(vocab)
    data = np.ones(len(indices), dtype=np.float32)
    cp.X = sparse.csr_matrix(
        (data, np.frombuffer(indices, dtype=np.int32), np.frombuffer(indptr, dtype=np.int64)),
        shape=(N, V),
    )
    cp.Xc = cp.X.tocsc()
    cp.df = np.asarray(cp.X.sum(axis=0)).ravel()
    cp.years = np.frombuffer(years, dtype=np.int32).copy()
    cp.cites = np.frombuffer(cites, dtype=np.int32).astype(np.float32)
    for name, arr in (("review", f_review), ("pre", f_pre), ("vivo", f_vivo), ("clin", f_clin), ("obs", f_obs)):
        cp.flags[name] = np.frombuffer(arr, dtype=np.int8).astype(np.float32)

    def _mat(rows, cols, k):
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32),
             (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
            shape=(N, k),
        )

    cp.ctx = _mat(ctx_rows, ctx_cols, len(cp.ctx_labels))
    cp.meth = _mat(m_rows, m_cols, len(cp.meth_labels))
    cue_rx = [WEAK_PRECLINICAL, STRONG_INVIVO, STRONG_CLINICAL, WEAK_OBSERVATIONAL, REVIEW_TITLE,
              GAP_RX, CONTRA_RX] + [rx for rx in ctx_rx + meth_rx if rx is not None]
    cp.ok = np.array([not any(rx.search(cp.label(j)) for rx in cue_rx) for j in range(V)], dtype=bool)
    return cp


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _idf(cp: Corpus) -> np.ndarray:
    return np.log((cp.N + 1) / (cp.df + 1))


def _distinct(cp: Corpus, ordered_js, limit: int, thresh: float = 0.7) -> list[int]:
    """Keep terms in order, skipping ones whose papers mostly overlap an earlier pick."""
    picked: list[int] = []
    picked_sets: list[set] = []
    for j in ordered_js:
        j = int(j)
        docs = set(cp.docs_with(j).tolist())
        if any(len(docs & ps) / max(1, len(docs | ps)) > thresh for ps in picked_sets):
            continue
        picked.append(j)
        picked_sets.append(docs)
        if len(picked) >= limit:
            break
    return picked


def detect_statements(cp: Corpus, brief: Brief) -> list[GapCandidate]:
    if not cp.statements or cp.V == 0:
        return []
    index = {k: i for i, k in enumerate(cp.vocab)}
    idf = _idf(cp)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for s_idx, (d, kind, sent) in enumerate(cp.statements):
        ids = {index[k] for k, _ in candidate_ngrams(sent) if k in index and cp.ok[index[k]]}
        if not ids:
            continue
        j = max(ids, key=lambda i: idf[i] * (1.3 if " " in cp.vocab[i] else 1.0))
        groups[(kind, j)].append(s_idx)

    N = cp.N
    min_docs = {"EXPLICIT": max(3, round(N / 4000)), "CONTRADICTION": max(2, round(N / 8000))}
    valid_years = cp.years[cp.years > 0]
    y_lo, y_hi = (int(valid_years.min()), int(valid_years.max())) if valid_years.size else (0, 0)
    focus = brief.focus or brief.niche
    out = []
    for (kind, j), s_list in groups.items():
        docs = sorted({cp.statements[i][0] for i in s_list})
        if len(docs) < min_docs[kind]:
            continue
        yrs = [cp.years[d] for d in docs if cp.years[d] > 0]
        recency = ((np.mean(yrs) - y_lo) / (y_hi - y_lo + 1)) if yrs and y_hi > y_lo else 0.5
        cite = float(np.mean(np.log1p(cp.cites[docs])))
        score = len(docs) * (0.5 + 0.5 * recency) * (1 + 0.1 * cite)
        snippets = {}
        for i in s_list:
            d, _, sent = cp.statements[i]
            snippets.setdefault(d, sent)
        ev = cp.evidence(list(snippets), k=3, order="recent", snippets=snippets)
        t = cp.label(j)
        first = shorten(ev[0].get("snippet", ""), 160) if ev else ""
        if kind == "EXPLICIT":
            title = f"Open unknowns around “{t}”"
            question = (f"What remains unknown about {t} in {focus}? {len(docs)} papers flag it "
                        f"explicitly (e.g. “{first}”).")
            approach = ("Design a study that answers the stated unknown directly; start from the "
                        "limitations and future-work sections of the cited papers.")
        else:
            title = f"Conflicting findings on “{t}”"
            question = (f"Why do studies on {t} in {focus} disagree, and which factors "
                        f"(population, method, setting) explain the differences?")
            approach = ("Meta-analysis with subgroup/moderator analysis, or a head-to-head study "
                        "that controls the suspected moderators.")
        out.append(GapCandidate(
            kind=kind, title=title, terms=[t], score=float(score), question=question,
            approach=approach, evidence=ev,
            stats={"papers_flagging": len(docs), "mean_year": round(float(np.mean(yrs)), 1) if yrs else None},
        ))
    return out


BRIDGE_TERMS = 2_000  # keeps the dense V×V matrices ~16 MB each


def detect_bridges(cp: Corpus, brief: Brief, max_out: int = 25) -> list[GapCandidate]:
    N, V = cp.N, cp.V
    if N < 50 or V < 10:
        return []
    sub = np.sort(np.argsort(-cp.df)[:BRIDGE_TERMS])  # local index -> vocab index
    Xs = cp.Xc[:, sub]
    C = (Xs.T @ Xs).toarray().astype(np.float32)
    d = cp.df[sub].astype(np.float32)
    E = np.outer(d, d) / np.float32(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(E > 0, C / E, 0.0)
    min_co = max(3, math.ceil(0.0002 * N))
    A = (C >= min_co) & (lift >= 2.0)
    np.fill_diagonal(A, False)
    Af = A.astype(np.float32)
    deg = Af.sum(axis=1)
    S = Af @ Af
    min_df = max(5, math.ceil(0.001 * N))
    ok_df = (d >= min_df) & cp.ok[sub]
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(deg[:, None] * deg[None, :] > 0, S / np.sqrt(np.outer(deg, deg)), 0.0)
        ratio = np.where(E > 0, C / E, 1.0)
    mask = (
        ok_df[:, None] & ok_df[None, :] & (E >= 3) & (ratio <= 0.25) & (S >= 3)
        & (cos >= 0.15) & (cos <= 0.6)
    )
    mask = np.triu(mask, k=1)
    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return []
    scores = cos[ii, jj] * np.log1p(E[ii, jj]) * (1 - ratio[ii, jj])
    order = np.argsort(-scores)[:400]
    focus = brief.focus or brief.niche
    out: list[GapCandidate] = []
    used: Counter = Counter()
    for o in order:
        i, j = int(ii[o]), int(jj[o])          # local indices
        gi, gj = int(sub[i]), int(sub[j])      # vocab indices
        if used[i] >= 2 or used[j] >= 2:
            continue
        ti, tj = set(cp.vocab[gi].split()), set(cp.vocab[gj].split())
        if ti & tj:
            continue
        shared = np.nonzero(A[i] & A[j])[0]
        shared = sorted(shared, key=lambda b: -min(lift[i, b], lift[j, b]))[:3]
        mids = [cp.label(int(sub[b])) for b in shared]
        a_lab, c_lab = cp.label(gi), cp.label(gj)
        ev = []
        if shared:
            b0 = int(sub[shared[0]])
            ev += cp.evidence(cp.docs_with_all([gi, b0]), k=1)
            ev += cp.evidence(cp.docs_with_all([gj, b0]), k=1)
        together = cp.docs_with_all([gi, gj])
        ev += cp.evidence(together, k=1)
        obs, exp = int(C[i, j]), float(E[i, j])
        mids_txt = ", ".join(mids[:2]) if mids else "the same topics"
        out.append(GapCandidate(
            kind="BRIDGE",
            title=f"“{a_lab}” × “{c_lab}”: rarely studied together",
            terms=[a_lab, c_lab],
            score=float(scores[o]),
            question=(f"Is {a_lab} linked to {c_lab} in {focus}? Both connect to {mids_txt}, "
                      f"yet only {obs} papers study them together (≈{exp:.0f} expected by chance)."),
            approach=("Test the link directly: a targeted experiment or survey, or re-analysis of "
                      "existing datasets that measure both."),
            evidence=ev,
            stats={"together": obs, "expected": round(exp, 1), "shared_neighbours": mids,
                   "df_a": int(d[i]), "df_b": int(d[j])},
        ))
        used[i] += 1
        used[j] += 1
        if len(out) >= max_out:
            break
    return out


def _attribute_deficits(cp: Corpus, brief: Brief, kind: str, labels: list[str], M: sparse.csr_matrix,
                        user_flags: list[bool]) -> list[GapCandidate]:
    N = cp.N
    if N == 0 or not labels:
        return []
    focus = brief.focus or brief.niche
    n_k = np.asarray(M.sum(axis=0)).ravel()
    O = (cp.Xc.T @ M).toarray() if cp.V else np.zeros((0, len(labels)))
    d = cp.df.astype(np.float32)
    Mc = M.tocsc()
    min_df = max(8, math.ceil(0.002 * N))
    out: list[GapCandidate] = []
    for k, lab in enumerate(labels):
        user = user_flags[k]
        nk = int(n_k[k])
        share = nk / N
        if not user and (share < 0.01 or share > 0.6 or nk < 10):
            continue
        if user and share < 0.02:
            k_docs = Mc.indices[Mc.indptr[k]: Mc.indptr[k + 1]]
            if kind == "CONTEXT":
                title = f"Niche barely studied in {lab}"
                question = (f"What is the status of {focus} in {lab}? Only {nk} of {N:,} screened "
                            f"papers mention it.")
                approach = (f"Baseline characterisation / prevalence study in {lab}, then compare "
                            "with the global literature.")
            else:
                title = f"{lab} barely applied in this niche"
                question = (f"What would {lab} reveal about {focus}? Only {nk} of {N:,} screened "
                            f"papers use it.")
                approach = f"Pilot {lab} on a well-studied sub-question first, ideally with public data."
            out.append(GapCandidate(
                kind=kind, title=title, terms=[lab], score=1e6 * (1 - share),
                question=question, approach=approach,
                evidence=cp.evidence(k_docs, k=3),
                stats={"papers_mentioning": nk, "share": round(share, 4), "level": "niche"},
            ))
        if nk == 0 or cp.V == 0:
            continue
        exp = d * nk / N
        obs = O[:, k]
        mask = (d >= min_df) & (exp >= 3) & (obs <= 0.25 * exp) & cp.ok
        if not mask.any():
            continue
        z = np.where(mask, (exp - obs) / np.sqrt(np.maximum(exp, 1e-9)), 0.0)
        order = [j for j in np.argsort(-(z * np.log1p(d)))[:40] if z[j] > 0]
        top = _distinct(cp, order, 4)
        k_docs = Mc.indices[Mc.indptr[k]: Mc.indptr[k + 1]]
        for j in top:
            t = cp.label(j)
            if kind == "CONTEXT":
                title = f"“{t}” is rarely studied in {lab}"
                question = (f"How does {t} present or operate in {lab}, compared with the settings "
                            f"already studied? ({int(obs[j])} papers vs ≈{exp[j]:.0f} expected)")
                approach = f"Replicate the strongest existing designs on {t} in {lab}."
            else:
                title = f"{lab} rarely applied to “{t}”"
                question = (f"Can {lab} answer open questions about {t} in {focus}? "
                            f"({int(obs[j])} papers combine them vs ≈{exp[j]:.0f} expected)")
                approach = f"Apply {lab} to {t}; pilot on existing or public datasets first."
            ev = cp.evidence(cp.docs_with(j), k=2)
            both = np.intersect1d(cp.docs_with(j), k_docs, assume_unique=True)
            ev += cp.evidence(both, k=1)
            out.append(GapCandidate(
                kind=kind, title=title, terms=[t, lab], score=float(z[j] * np.log1p(d[j])),
                question=question, approach=approach, evidence=ev,
                stats={"observed": int(obs[j]), "expected": round(float(exp[j]), 1),
                       "papers_on_term": int(d[j]), "papers_with_attr": nk, "level": "term"},
            ))
    return out


def detect_contexts(cp: Corpus, brief: Brief) -> list[GapCandidate]:
    return _attribute_deficits(cp, brief, "CONTEXT", cp.ctx_labels, cp.ctx, [True] * len(cp.ctx_labels))


def detect_methods(cp: Corpus, brief: Brief) -> list[GapCandidate]:
    return _attribute_deficits(cp, brief, "METHOD", cp.meth_labels, cp.meth, cp.meth_user)


def _end_year(brief: Brief, years: np.ndarray) -> int | None:
    valid = years[years > 0]
    if valid.size == 0:
        return None
    today = _dt.date.today()
    end = today.year if today.month >= 10 else today.year - 1
    if brief.year_to:
        end = min(end, brief.year_to)
    return int(min(end, valid.max()))


def detect_trends(cp: Corpus, brief: Brief) -> tuple[list[GapCandidate], str | None]:
    end = _end_year(brief, cp.years)
    if end is None or cp.V == 0:
        return [], "No publication years available; trend detectors skipped."
    valid = cp.years[cp.years > 0]
    if end - int(valid.min()) < 5:
        return [], "Corpus spans under 6 years; trend detectors skipped."
    y = cp.years
    recent = ((y >= end - 2) & (y <= end)).astype(np.float32)
    prev = ((y >= end - 5) & (y <= end - 3)).astype(np.float32)
    old = ((y > 0) & (y < end - 5)).astype(np.float32)
    XT = cp.Xc.T
    r, p, o = XT @ recent, XT @ prev, XT @ old
    cite_sum = XT @ cp.cites
    d = cp.df.astype(np.float32)
    N = cp.N
    R, P = recent.sum(), prev.sum()
    n_y = float((y > 0).sum())
    focus = brief.focus or brief.niche
    out: list[GapCandidate] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = ((r + 1) / (p + 1)) / ((R + 1) / (P + 1))
    em_mask = (r >= max(5, math.ceil(0.0003 * N))) & (rel >= 2.0) & (d / N <= 0.05) & cp.ok
    em_idx = np.nonzero(em_mask)[0]
    em_scores = np.log2(rel[em_idx]) * np.log1p(r[em_idx])
    em_pos = {int(em_idx[o]): o for o in range(len(em_idx))}
    em_order = [int(em_idx[o]) for o in np.argsort(-em_scores)[:60]]
    for j in _distinct(cp, em_order, 12):
        o_i = em_pos[j]
        t = cp.label(j)
        out.append(GapCandidate(
            kind="EMERGING", title=f"Rising but thin: “{t}”", terms=[t], score=float(em_scores[o_i]),
            question=(f"What foundational questions about {t} in {focus} are still open? It grew "
                      f"{rel[j]:.1f}× faster than the field ({int(p[j])}→{int(r[j])} papers in "
                      f"{end - 5}-{end - 3} vs {end - 2}-{end})."),
            approach="Scoping review of the recent papers, then a pilot on the most-cited open question.",
            evidence=cp.evidence(cp.docs_with(j), k=3, order="recent"),
            stats={"recent": int(r[j]), "previous": int(p[j]), "relative_growth": round(float(rel[j]), 2)},
        ))

    mean_cite_all = float(cp.cites.mean()) if N else 0.0
    if mean_cite_all > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_cite = np.where(d > 0, cite_sum / d, 0.0)
            ratio = mean_cite / mean_cite_all
        exp_r = d * (R / max(n_y, 1))
        dm_mask = ((o >= max(10, math.ceil(0.0005 * N))) & (ratio >= 1.5) & (exp_r >= 4)
                   & (r <= 0.35 * exp_r) & cp.ok)
        dm_idx = np.nonzero(dm_mask)[0]
        dm_scores = np.log1p(ratio[dm_idx]) * (exp_r[dm_idx] - r[dm_idx]) / np.sqrt(exp_r[dm_idx])
        dm_pos = {int(dm_idx[o]): o for o in range(len(dm_idx))}
        dm_order = [int(dm_idx[o]) for o in np.argsort(-dm_scores)[:60]]
        for j in _distinct(cp, dm_order, 8):
            o_i = dm_pos[j]
            t = cp.label(j)
            out.append(GapCandidate(
                kind="DORMANT", title=f"Highly cited but gone quiet: “{t}”", terms=[t],
                score=float(dm_scores[o_i]),
                question=(f"Can newer methods or data revisit {t} in {focus}? Its papers are cited "
                          f"{ratio[j]:.1f}× the field average, but only {int(r[j])} appeared in "
                          f"{end - 2}-{end} (≈{exp_r[j]:.0f} expected)."),
                approach="Re-test the classic papers' claims with current methods, data or settings.",
                evidence=cp.evidence(cp.docs_with(j), k=3, order="cited"),
                stats={"recent": int(r[j]), "expected_recent": round(float(exp_r[j]), 1),
                       "citation_ratio": round(float(ratio[j]), 2)},
            ))
    return out, None


def detect_evidence(cp: Corpus, brief: Brief) -> list[GapCandidate]:
    if cp.V == 0:
        return []
    XT = cp.Xc.T
    d = cp.df.astype(np.float32)
    N = cp.N
    rev = XT @ cp.flags["review"]
    pre = XT @ cp.flags["pre"]
    strong_any = XT @ np.maximum(cp.flags["vivo"], cp.flags["clin"])
    obs = XT @ cp.flags["obs"]
    clin = XT @ cp.flags["clin"]
    focus = brief.focus or brief.niche
    out: list[GapCandidate] = []

    primary = d - rev
    syn_idx = np.nonzero((primary >= max(20, math.ceil(0.002 * N))) & (rev == 0) & cp.ok)[0]
    syn_scores = np.log1p(primary[syn_idx])
    syn_pos = {int(syn_idx[o]): o for o in range(len(syn_idx))}
    syn_order = [int(syn_idx[o]) for o in np.argsort(-syn_scores)[:60]]
    for j in _distinct(cp, syn_order, 10):
        o_i = syn_pos[j]
        t = cp.label(j)
        out.append(GapCandidate(
            kind="EVIDENCE", title=f"No review synthesises “{t}”", terms=[t], score=float(syn_scores[o_i]),
            question=(f"What does the combined evidence from {int(primary[j])} primary studies say "
                      f"about {t} in {focus}?"),
            approach="Systematic review / meta-analysis (PRISMA); register the protocol on PROSPERO or OSF.",
            evidence=cp.evidence(cp.docs_with(j), k=3),
            stats={"subtype": "synthesis", "primary_studies": int(primary[j]), "reviews": 0},
        ))

    min_n = max(10, math.ceil(0.002 * N))
    val_idx = np.nonzero((pre >= min_n) & (strong_any <= 0.1 * pre) & cp.ok)[0]
    val_scores = np.log1p(pre[val_idx]) * (1 - strong_any[val_idx] / np.maximum(pre[val_idx], 1))
    val_pos = {int(val_idx[o]): o for o in range(len(val_idx))}
    val_order = [int(val_idx[o]) for o in np.argsort(-val_scores)[:60]]
    for j in _distinct(cp, val_order, 8):
        o_i = val_pos[j]
        t = cp.label(j)
        out.append(GapCandidate(
            kind="EVIDENCE", title=f"“{t}”: mostly in silico / in vitro", terms=[t],
            score=float(val_scores[o_i]),
            question=(f"Do in silico / in vitro findings on {t} hold up in vivo or in clinical/field "
                      f"settings? ({int(pre[j])} preclinical papers vs {int(strong_any[j])} with in vivo "
                      f"or clinical data)"),
            approach="Validation study in an animal model, clinical samples or field data.",
            evidence=cp.evidence(cp.docs_with(j), k=3),
            stats={"subtype": "validation", "preclinical": int(pre[j]), "in_vivo_or_clinical": int(strong_any[j])},
        ))

    des_idx = np.nonzero((obs >= min_n) & (clin <= 0.1 * obs) & cp.ok)[0]
    des_scores = np.log1p(obs[des_idx]) * (1 - clin[des_idx] / np.maximum(obs[des_idx], 1))
    des_pos = {int(des_idx[o]): o for o in range(len(des_idx))}
    des_order = [int(des_idx[o]) for o in np.argsort(-des_scores)[:60]]
    for j in _distinct(cp, des_order, 8):
        o_i = des_pos[j]
        t = cp.label(j)
        out.append(GapCandidate(
            kind="EVIDENCE", title=f"“{t}”: few prospective studies", terms=[t],
            score=float(des_scores[o_i]),
            question=(f"Would a prospective or longitudinal design change what we know about {t}? "
                      f"({int(obs[j])} cross-sectional/retrospective papers vs {int(clin[j])} prospective)"),
            approach="Prospective cohort or longitudinal follow-up with pre-registered outcomes.",
            evidence=cp.evidence(cp.docs_with(j), k=3),
            stats={"subtype": "design", "weak_design": int(obs[j]), "strong_design": int(clin[j])},
        ))
    return out


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_candidates(cands: list[GapCandidate], brief: Brief, prefs: dict | None = None,
                    top_k: int = 24, max_per_kind: int = 5) -> list[GapCandidate]:
    prefs = prefs or {}
    weights = dict(DEFAULT_KIND_WEIGHTS)
    for k in prefs.get("focus_kinds", []) or []:
        if k in weights:
            weights[k] *= 1.3
    for k in prefs.get("avoid_kinds", []) or []:
        if k in weights:
            weights[k] *= 0.5
    focus_toks = {fold(t) for t in tokenize(f"{brief.focus} {brief.niche}") if is_content(t)}
    by_kind: dict[str, list[GapCandidate]] = defaultdict(list)
    for c in cands:
        by_kind[c.kind].append(c)
    for kind, lst in by_kind.items():
        lst.sort(key=lambda c: -c.score)
        n = len(lst)
        for r, c in enumerate(lst):
            norm = 1 - r / max(n, 1)
            toks = {fold(t) for term in c.terms for t in tokenize(term)}
            rel = min(1.0, len(toks & focus_toks) / 2)
            c.final = round((0.5 + 0.5 * norm) * weights.get(kind, 1.0) * (1 + 0.25 * rel), 4)
    chosen: list[GapCandidate] = []
    per_kind: Counter = Counter()
    term_use: Counter = Counter()
    for c in sorted(cands, key=lambda c: -c.final):
        if per_kind[c.kind] >= max_per_kind:
            continue
        if any(term_use[t.lower()] >= 2 for t in c.terms):
            continue
        chosen.append(c)
        per_kind[c.kind] += 1
        for t in c.terms:
            term_use[t.lower()] += 1
        if len(chosen) >= top_k:
            break
    for i, c in enumerate(chosen, 1):
        c.gid = f"G{i}"
    return chosen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def mine(docs_factory: Callable[[], Iterable], n_total: int, brief: Brief, prefs: dict | None = None,
         progress: ProgressFn | None = None, top_k: int = 24) -> MiningResult:
    query_keys = {normalize_phrase(t) for t in brief.all_query_terms()}
    query_keys.discard("")
    vocab, n_sample = build_vocab(docs_factory, n_total, query_keys, progress)
    cp = scan(docs_factory, vocab, brief, n_total, progress)
    notes: list[str] = []
    if cp.N == 0:
        return MiningResult(0, 0, None, None, {}, [], ["No papers to analyse."], {})
    if progress:
        progress("Detecting gaps", 0.1)
    cands: list[GapCandidate] = []
    cands += detect_statements(cp, brief)
    if progress:
        progress("Detecting gaps", 0.3)
    cands += detect_bridges(cp, brief)
    if progress:
        progress("Detecting gaps", 0.6)
    cands += detect_contexts(cp, brief)
    cands += detect_methods(cp, brief)
    trend, note = detect_trends(cp, brief)
    cands += trend
    if note:
        notes.append(note)
    cands += detect_evidence(cp, brief)
    if progress:
        progress("Ranking", 0.9)
    if cp.N < 300:
        notes.append(f"Small corpus ({cp.N} papers): statistical signals are weak; treat as leads only.")
    ranked = rank_candidates(cands, brief, prefs, top_k=top_k)
    valid = cp.years[cp.years > 0]
    ctx_counts = {lab: int(n) for lab, n in zip(cp.ctx_labels, np.asarray(cp.ctx.sum(axis=0)).ravel())}
    meth_counts = {lab: int(n) for lab, n in zip(cp.meth_labels, np.asarray(cp.meth.sum(axis=0)).ravel())}
    stats = {
        "vocab_sample_docs": n_sample,
        "statements_found": len(cp.statements),
        "candidates_total": len(cands),
        "candidates_by_kind": dict(Counter(c.kind for c in cands)),
        "context_counts": ctx_counts,
        "method_counts": meth_counts,
        "reviews": int(cp.flags["review"].sum()),
    }
    return MiningResult(
        n_docs=cp.N, n_vocab=cp.V,
        year_min=int(valid.min()) if valid.size else None,
        year_max=int(valid.max()) if valid.size else None,
        sources=dict(cp.source_counts), candidates=ranked, notes=notes, stats=stats,
    )
