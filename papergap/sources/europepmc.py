"""Europe PMC REST API (life sciences + biomedicine, PubMed superset). Free, no key.

Limits published by Europe PMC: 10 requests/s or 500/min, pageSize max 1000,
deep paging via cursorMark only.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from typing import AsyncIterator, Callable

import httpx

from ..brief import Brief
from .base import Paper, PaperSource, get_json, make_pid, new_client

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return html.unescape(_TAG_RE.sub(" ", text)).replace("  ", " ").strip()


def _term(t: str) -> str:
    t = t.replace('"', "").strip()
    return f'(TITLE:"{t}" OR ABSTRACT:"{t}")'


def build_query(brief: Brief, with_filters: bool = True) -> str:
    parts = ["(" + " OR ".join(_term(t) for t in g) + ")" for g in brief.term_groups()]
    q = " AND ".join(parts)
    ex = brief.exclude_terms()
    if ex and q:
        q += " NOT (" + " OR ".join(_term(t) for t in ex) + ")"
    if with_filters and q:
        q = f"({q}) AND HAS_ABSTRACT:Y"
        if brief.year_from and brief.year_to:
            q += f" AND PUB_YEAR:[{brief.year_from} TO {brief.year_to}]"
    return q


def to_paper(r: dict) -> Paper:
    title = _clean(r.get("title"))
    doi = (r.get("doi") or "").strip() or None
    kws = list((r.get("keywordList") or {}).get("keyword") or [])
    for mh in (r.get("meshHeadingList") or {}).get("meshHeading") or []:
        name = mh.get("descriptorName")
        if name:
            kws.append(name)
    ptypes = [p.lower() for p in ((r.get("pubTypeList") or {}).get("pubType") or [])]
    ptype = "review" if any("review" in p or "meta-analysis" in p for p in ptypes) else (
        "preprint" if r.get("source") == "PPR" else "article"
    )
    year = r.get("pubYear")
    try:
        year = int(year) if year else None
    except ValueError:
        year = None
    src, rid = r.get("source") or "MED", r.get("id") or ""
    venue = ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
    return Paper(
        pid=make_pid(doi, title, f"epmc:{src}:{rid}"),
        source="europepmc",
        title=title,
        abstract=_clean(r.get("abstractText")),
        year=year,
        cited_by=int(r.get("citedByCount") or 0),
        doi=doi,
        url=f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{src}/{rid}",
        keywords=kws,
        ptype=ptype,
        venue=venue,
    )


class EuropePMCSource(PaperSource):
    name = "europepmc"

    def __init__(self, email: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.email = email
        self.client = new_client(timeout=120.0, transport=transport)

    def _params(self, query: str, page_size: int, result_type: str = "core", cursor: str = "*") -> dict:
        p = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": page_size,
            "cursorMark": cursor,
        }
        if self.email:
            p["email"] = self.email
        return p

    async def count(self, brief: Brief) -> int:
        data = await get_json(self.client, BASE, self._params(build_query(brief), 1, "idlist"))
        return int(data.get("hitCount") or 0)

    async def sample_keywords(self, brief: Brief, n: int = 100) -> list[tuple[str, int]]:
        data = await get_json(self.client, BASE, self._params(build_query(brief), min(n, 1000)))
        c: Counter = Counter()
        for r in (data.get("resultList") or {}).get("result") or []:
            for k in (r.get("keywordList") or {}).get("keyword") or []:
                c[k.strip()] += 1
            for mh in (r.get("meshHeadingList") or {}).get("meshHeading") or []:
                if mh.get("descriptorName"):
                    c[mh["descriptorName"]] += 1
        return c.most_common(25)

    async def harvest(
        self, brief: Brief, max_papers: int, should_stop: Callable[[], bool] | None = None
    ) -> AsyncIterator[list[Paper]]:
        query = build_query(brief)
        cursor = "*"
        fetched = 0
        page_size = 1000 if max_papers >= 1000 else max(25, max_papers)
        while fetched < max_papers:
            if should_stop and should_stop():
                return
            data = await get_json(self.client, BASE, self._params(query, page_size, "core", cursor))
            results = (data.get("resultList") or {}).get("result") or []
            if not results:
                return
            batch = [to_paper(r) for r in results if r.get("abstractText")][: max_papers - fetched]
            fetched += len(batch)
            if batch:
                yield batch
            nxt = data.get("nextCursorMark")
            if not nxt or nxt == cursor:
                return
            cursor = nxt

    async def quick_search(self, terms: list[str], year_from: int | None = None, n: int = 3) -> tuple[int, list[Paper]]:
        q = " AND ".join(_term(t) for t in terms if t.strip())
        if year_from:
            q = f"({q}) AND PUB_YEAR:[{year_from} TO 3000]"
        data = await get_json(self.client, BASE, self._params(q, n, "lite"))
        papers = [to_paper(r) for r in (data.get("resultList") or {}).get("result") or []]
        return int(data.get("hitCount") or 0), papers

    async def aclose(self) -> None:
        await self.client.aclose()
