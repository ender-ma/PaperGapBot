"""arXiv API (physics, CS, maths, quantitative biology ...). Free, no key.

arXiv asks for one request every ~3 s and caps deep result sets (~30k), so this
source is best for preprint-heavy fields. No citation counts are available.
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import AsyncIterator, Callable

import httpx

from ..brief import Brief
from .base import Paper, PaperSource, SourceError, get_json, make_pid, new_client

BASE = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
    "ax": "http://arxiv.org/schemas/atom",
}
PAGE = 500
MAX_TOTAL = 30_000
DELAY_S = 3.1


def _term(t: str) -> str:
    t = t.replace('"', "").strip()
    return f'(ti:"{t}" OR abs:"{t}")'


def build_query(brief: Brief) -> str:
    parts = ["(" + " OR ".join(_term(t) for t in g) + ")" for g in brief.term_groups()]
    q = " AND ".join(parts)
    ex = brief.exclude_terms()
    if ex and q:
        q += " ANDNOT (" + " OR ".join(_term(t) for t in ex) + ")"
    if brief.year_from and brief.year_to and q:
        q = f"({q}) AND submittedDate:[{brief.year_from}01010000 TO {brief.year_to}12312359]"
    return q


def parse_feed(xml_text: str) -> tuple[int, list[Paper]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError(f"Bad arXiv response: {exc}") from exc
    total_el = root.find("os:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    papers: list[Paper] = []
    for e in root.findall("a:entry", NS):
        def txt(tag: str) -> str:
            el = e.find(tag, NS)
            return re.sub(r"\s+", " ", el.text).strip() if el is not None and el.text else ""

        abs_url = txt("a:id")
        title = txt("a:title")
        if not abs_url or title.lower() == "error":
            continue
        published = txt("a:published")
        year = int(published[:4]) if published[:4].isdigit() else None
        doi = txt("ax:doi") or None
        cats = [c.get("term") for c in e.findall("a:category", NS) if c.get("term")]
        aid = abs_url.rsplit("/abs/", 1)[-1]
        papers.append(Paper(
            pid=make_pid(doi, title, "arxiv:" + re.sub(r"v\d+$", "", aid)),
            source="arxiv",
            title=title,
            abstract=txt("a:summary"),
            year=year,
            cited_by=0,
            doi=doi,
            url=abs_url,
            keywords=cats,
            ptype="preprint",
            venue="arXiv",
        ))
    return total, papers


class ArxivSource(PaperSource):
    name = "arxiv"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, delay_s: float = DELAY_S):
        self.client = new_client(timeout=90.0, transport=transport)
        self.delay_s = delay_s
        self._last = 0.0

    async def _fetch(self, query: str, start: int, n: int) -> tuple[int, list[Paper]]:
        loop = asyncio.get_running_loop()
        wait = self._last + self.delay_s - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        resp = await get_json(
            self.client, BASE,
            {"search_query": query, "start": start, "max_results": n, "sortBy": "relevance"},
            expect_json=False,
        )
        self._last = loop.time()
        return parse_feed(resp.text)

    async def count(self, brief: Brief) -> int:
        total, _ = await self._fetch(build_query(brief), 0, 1)
        return total

    async def sample_keywords(self, brief: Brief, n: int = 100) -> list[tuple[str, int]]:
        _, papers = await self._fetch(build_query(brief), 0, min(n, 200))
        c: Counter = Counter(k for p in papers for k in p.keywords)
        return c.most_common(15)

    async def harvest(
        self, brief: Brief, max_papers: int, should_stop: Callable[[], bool] | None = None
    ) -> AsyncIterator[list[Paper]]:
        query = build_query(brief)
        limit = min(max_papers, MAX_TOTAL)
        start = 0
        while start < limit:
            if should_stop and should_stop():
                return
            total, papers = await self._fetch(query, start, min(PAGE, limit - start))
            if not papers:
                return
            yield papers
            start += len(papers)
            if start >= total:
                return

    async def quick_search(self, terms: list[str], year_from: int | None = None, n: int = 3) -> tuple[int, list[Paper]]:
        q = " AND ".join(_term(t) for t in terms if t.strip())
        return await self._fetch(q, 0, n)

    async def aclose(self) -> None:
        await self.client.aclose()
