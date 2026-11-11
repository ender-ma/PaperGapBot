"""Common types for paper sources."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import httpx

from ..brief import Brief

log = logging.getLogger(__name__)

USER_AGENT = "PaperGapBot/1.0 (research gap finder; Telegram bot)"


@dataclass
class Paper:
    pid: str                      # stable id used for de-duplication
    source: str
    title: str
    abstract: str
    year: int | None = None
    cited_by: int = 0
    doi: str | None = None
    url: str | None = None
    keywords: list[str] = field(default_factory=list)
    ptype: str | None = None      # article, review, preprint ...
    venue: str | None = None


def make_pid(doi: str | None, title: str, fallback: str) -> str:
    if doi:
        d = doi.lower().strip()
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
        return "doi:" + d
    norm = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    if norm:
        return "t:" + hashlib.sha1(norm.encode()).hexdigest()[:16]
    return fallback


class SourceError(RuntimeError):
    pass


class BudgetExhausted(SourceError):
    pass


class PaperSource:
    """Interface every source implements."""

    name = "base"

    async def count(self, brief: Brief) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    async def sample_keywords(self, brief: Brief, n: int = 100) -> list[tuple[str, int]]:
        """Frequent keywords in a sample of matching papers (used to suggest narrowing)."""
        return []

    def harvest(
        self, brief: Brief, max_papers: int, should_stop: Callable[[], bool] | None = None
    ) -> AsyncIterator[list[Paper]]:  # pragma: no cover - interface
        raise NotImplementedError

    async def quick_search(self, terms: list[str], year_from: int | None = None, n: int = 3) -> tuple[int, list[Paper]]:
        """AND-search used by the novelty check. Returns (count, top papers)."""
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    retries: int = 5,
    expect_json: bool = True,
) -> httpx.Response | dict:
    """GET with exponential backoff on 429/5xx and network errors."""
    delay = 1.5
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                log.warning("HTTP %s from %s, retrying in %.1fs", resp.status_code, url, wait)
                if resp.status_code == 429 and attempt == retries - 1:
                    raise BudgetExhausted(f"Rate limit or budget reached at {url}")
                await asyncio.sleep(wait + random.random())
                delay *= 2
                continue
            if resp.status_code >= 400:
                raise SourceError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
            return resp.json() if expect_json else resp
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            log.warning("Network error %s on %s (attempt %d)", exc, url, attempt + 1)
            await asyncio.sleep(delay + random.random())
            delay *= 2
    raise SourceError(f"Failed after {retries} attempts: {url} ({last_exc})")


def new_client(timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        transport=transport,
    )
