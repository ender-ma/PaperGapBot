from __future__ import annotations

import json

from papergap.llm import LLMClient
from papergap.sources.base import PaperSource


class FakeSource(PaperSource):
    name = "fake"

    def __init__(self, papers=None, count: int = 5000, keywords=None):
        self.papers = papers or []
        self._count = count
        self._keywords = keywords or [("biosecurity", 12), ("layer hens", 7), ("colistin", 30)]
        self.searches: list[list[str]] = []

    async def count(self, brief):
        return self._count

    async def sample_keywords(self, brief, n=100):
        return self._keywords

    async def harvest(self, brief, max_papers, should_stop=None):
        for i in range(0, min(len(self.papers), max_papers), 500):
            if should_stop and should_stop():
                return
            yield self.papers[i: min(i + 500, max_papers)]

    async def quick_search(self, terms, year_from=None, n=3):
        self.searches.append(terms)
        return 2, self.papers[:n]


class FakeLLM(LLMClient):
    name = "fake-llm"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system, user, max_tokens=4000):
        self.calls.append((system, user))
        if "methodologist" in system:
            return json.dumps({"score": 8, "verdict": "Specific niche.", "missing": [], "sharper_focus": "x"})
        return "Here you go:\n```json\n" + json.dumps({"questions": [
            {"title": "Biosecurity and mcr-1 in Nigerian farms", "question": "Does biosecurity reduce mcr-1 carriage on Nigerian poultry farms?",
             "signal_ids": ["G1", "G99"], "approach": "Cross-farm survey with WGS", "feasibility": "medium",
             "check_terms": ["biosecurity", "mcr-1", "Nigeria"]},
            {"title": "Invented", "question": "This cites nothing real", "signal_ids": ["G404"],
             "check_terms": ["a", "b"]},
        ]}) + "\n```"
