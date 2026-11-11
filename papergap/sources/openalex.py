"""OpenAlex works API (all disciplines, ~300M+ works).

Since Feb 2026 OpenAlex requires a free API key and bills usage against a daily
allowance ($1/day on a free key). A `search=` page of 100 works costs about
$0.001, so ~100k papers per day fit in the free allowance. Each response reports
`meta.cost_usd`, which this client sums to stop before the configured budget.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import AsyncIterator, Callable

import httpx

from ..brief import Brief
from ..mining.text import reconstruct_abstract
from .base import BudgetExhausted, Paper, PaperSource, SourceError, get_json, make_pid, new_client

log = logging.getLogger(__name__)

BASE = "https://api.openalex.org"
SELECT = (
    "id,doi,title,display_name,publication_year,cited_by_count,abstract_inverted_index,"
    "keywords,primary_topic,type,primary_location"
)
EST_COST = {"search": 0.001, "filter": 0.0001}


def _quote(term: str) -> str:
    term = term.replace('"', "").strip()
    return f'"{term}"'


def build_query(brief: Brief) -> str:
    parts = ["(" + " OR ".join(_quote(t) for t in g) + ")" for g in brief.term_groups()]
    q = " AND ".join(parts)
    ex = brief.exclude_terms()
    if ex and q:
        q += " NOT (" + " OR ".join(_quote(t) for t in ex) + ")"
    return q


def and_query(terms: list[str]) -> str:
    return " AND ".join(_quote(t) for t in terms if t.strip())


def _strip_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I) or None


def to_paper(w: dict) -> Paper:
    title = w.get("title") or w.get("display_name") or ""
    doi = _strip_doi(w.get("doi"))
    kws = [k.get("display_name") for k in (w.get("keywords") or []) if k and k.get("display_name")]
    pt = w.get("primary_topic") or {}
    if pt.get("display_name"):
        kws.append(pt["display_name"])
    loc = w.get("primary_location") or {}
    venue = ((loc.get("source") or {}).get("display_name")) if loc else None
    oid = w.get("id") or ""
    return Paper(
        pid=make_pid(doi, title, "oa:" + oid.rsplit("/", 1)[-1]),
        source="openalex",
        title=title,
        abstract=reconstruct_abstract(w.get("abstract_inverted_index")),
        year=w.get("publication_year"),
        cited_by=int(w.get("cited_by_count") or 0),
        doi=doi,
        url=f"https://doi.org/{doi}" if doi else (oid or None),
        keywords=kws,
        ptype=w.get("type"),
        venue=venue,
    )


class OpenAlexSource(PaperSource):
    name = "openalex"

    def __init__(
        self,
        api_key: str | None,
        email: str | None = None,
        budget_usd: float = 0.9,
        mode: str = "search",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.email = email
        self.budget_usd = budget_usd
        self.mode = mode if mode in EST_COST else "search"
        self.spent_usd = 0.0
        self.client = new_client(transport=transport)

    # ------------------------------------------------------------------
    def _base_params(self) -> dict:
        p: dict = {}
        if self.api_key:
            p["api_key"] = self.api_key
        if self.email:
            p["mailto"] = self.email
        return p

    def _query_params(self, brief: Brief, query: str | None = None) -> dict:
        q = query if query is not None else build_query(brief)
        filters = ["has_abstract:true"]
        if brief.year_from and brief.year_to:
            filters.append(f"publication_year:{brief.year_from}-{brief.year_to}")
        params = self._base_params()
        if self.mode == "filter":
            filters.insert(0, f"title_and_abstract.search:{q}")
        else:
            params["search"] = q
        params["filter"] = ",".join(filters)
        return params

    def _charge(self, data: dict) -> None:
        cost = (data.get("meta") or {}).get("cost_usd")
        self.spent_usd += float(cost) if isinstance(cost, (int, float)) else EST_COST[self.mode]

    async def _get(self, path: str, params: dict) -> dict:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExhausted(
                f"OpenAlex budget reached (${self.spent_usd:.3f} of ${self.budget_usd:.2f})."
            )
        data = await get_json(self.client, BASE + path, params)
        if isinstance(data, dict):
            self._charge(data)
            return data
        raise SourceError("Unexpected response from OpenAlex")

    # ------------------------------------------------------------------
    async def remaining_budget(self) -> float | None:
        """Ask OpenAlex how much of today's allowance is left (best effort)."""
        if not self.api_key:
            return None
        try:
            data = await get_json(self.client, BASE + "/rate-limit", {"api_key": self.api_key}, retries=2)
            rl = data.get("rate_limit") or {}
            left = rl.get("daily_remaining_usd")
            prepaid = rl.get("prepaid_remaining_usd") or 0
            if left is None:
                return None
            return float(left) + float(prepaid)
        except Exception as exc:  # noqa: BLE001
            log.info("Could not read OpenAlex rate limit: %s", exc)
            return None

    async def count(self, brief: Brief) -> int:
        params = self._query_params(brief)
        params.update({"per_page": 1, "select": "id"})
        data = await self._get("/works", params)
        return int((data.get("meta") or {}).get("count") or 0)

    async def sample_keywords(self, brief: Brief, n: int = 100) -> list[tuple[str, int]]:
        params = self._query_params(brief)
        params.update({"per_page": min(100, n), "select": "keywords,primary_topic"})
        data = await self._get("/works", params)
        c: Counter = Counter()
        for w in data.get("results") or []:
            for k in w.get("keywords") or []:
                if k and k.get("display_name"):
                    c[k["display_name"]] += 1
        return c.most_common(25)

    async def harvest(
        self, brief: Brief, max_papers: int, should_stop: Callable[[], bool] | None = None
    ) -> AsyncIterator[list[Paper]]:
        cursor: str | None = "*"
        fetched = 0
        base = self._query_params(brief)
        base.update({"per_page": 100, "select": SELECT})
        while cursor and fetched < max_papers:
            if should_stop and should_stop():
                return
            params = dict(base, cursor=cursor)
            data = await self._get("/works", params)
            results = data.get("results") or []
            if not results:
                return
            batch = [to_paper(w) for w in results][: max_papers - fetched]
            fetched += len(batch)
            yield batch
            cursor = (data.get("meta") or {}).get("next_cursor")

    async def quick_search(self, terms: list[str], year_from: int | None = None, n: int = 3) -> tuple[int, list[Paper]]:
        params = self._base_params()
        params["search"] = and_query(terms)
        if year_from:
            params["filter"] = f"publication_year:>{year_from - 1}"
        params.update({"per_page": n, "select": "id,doi,title,publication_year,cited_by_count"})
        data = await self._get("/works", params)
        papers = [to_paper(w) for w in data.get("results") or []]
        return int((data.get("meta") or {}).get("count") or 0), papers

    async def aclose(self) -> None:
        await self.client.aclose()
