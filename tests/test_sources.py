import asyncio
import json

import httpx
import pytest

from papergap.demo import demo_brief
from papergap.mining.text import reconstruct_abstract
from papergap.sources.arxiv import ArxivSource, parse_feed
from papergap.sources.base import BudgetExhausted
from papergap.sources.europepmc import EuropePMCSource
from papergap.sources.europepmc import build_query as epmc_query
from papergap.sources.openalex import OpenAlexSource, build_query


def test_reconstruct_abstract():
    inv = {"Colistin": [0], "resistance": [1], "remains": [2], "unclear.": [3]}
    assert reconstruct_abstract(inv) == "Colistin resistance remains unclear."


def test_query_builders():
    b = demo_brief()
    b.exclude = "vaccine"
    assert build_query(b) == '("colistin" OR "polymyxin") AND ("poultry" OR "chicken" OR "broiler") NOT ("vaccine")'
    q = epmc_query(b)
    assert 'TITLE:"colistin"' in q and "HAS_ABSTRACT:Y" in q and "PUB_YEAR:[2008 TO 2026]" in q


def _work(i):
    return {
        "id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/10.1/{i}", "title": f"Paper {i}",
        "publication_year": 2020, "cited_by_count": i, "type": "article",
        "abstract_inverted_index": {"Some": [0], "text.": [1]},
        "keywords": [{"display_name": "mcr-1"}], "primary_topic": {"display_name": "AMR"},
    }


def test_openalex_cursor_paging_and_cost():
    seen = []

    def handler(request: httpx.Request):
        p = dict(request.url.params)
        seen.append(p)
        assert p["api_key"] == "KEY"
        assert "search" in p and "has_abstract:true" in p["filter"]
        page = 1 if p["cursor"] == "*" else 2
        return httpx.Response(200, json={
            "meta": {"count": 150, "next_cursor": "c2" if page == 1 else None, "cost_usd": 0.001},
            "results": [_work(i) for i in range(100 if page == 1 else 50)],
        })

    src = OpenAlexSource("KEY", transport=httpx.MockTransport(handler))

    async def go():
        out = []
        async for batch in src.harvest(demo_brief(), 1000):
            out += batch
        await src.aclose()
        return out

    papers = asyncio.run(go())
    assert len(papers) == 150 and len(seen) == 2
    assert papers[0].doi == "10.1/0" and papers[0].pid == "doi:10.1/0"
    assert papers[0].abstract == "Some text." and "AMR" in papers[0].keywords
    assert abs(src.spent_usd - 0.002) < 1e-9


def test_openalex_budget_stops():
    def handler(request):
        return httpx.Response(200, json={"meta": {"count": 10**6, "next_cursor": "x", "cost_usd": 0.5},
                                         "results": [_work(1)]})

    src = OpenAlexSource("KEY", budget_usd=0.9, transport=httpx.MockTransport(handler))

    async def go():
        n = 0
        async for batch in src.harvest(demo_brief(), 10**6):
            n += len(batch)
        return n

    with pytest.raises(BudgetExhausted):
        asyncio.run(go())


def test_europepmc_parse_and_paging():
    def handler(request):
        p = dict(request.url.params)
        cursor = p["cursorMark"]
        results = [{
            "id": "1", "source": "MED", "doi": "10.2/x", "title": "A <i>tagged</i> title",
            "abstractText": "<h4>Background</h4>Little is known about X.", "pubYear": "2021",
            "citedByCount": 4, "keywordList": {"keyword": ["poultry"]},
            "meshHeadingList": {"meshHeading": [{"descriptorName": "Chickens"}]},
            "pubTypeList": {"pubType": ["Review"]},
        }] if cursor == "*" else []
        return httpx.Response(200, json={"hitCount": 1, "nextCursorMark": "n1",
                                         "resultList": {"result": results}})

    src = EuropePMCSource(transport=httpx.MockTransport(handler))

    async def go():
        out = []
        async for b in src.harvest(demo_brief(), 100):
            out += b
        return out

    papers = asyncio.run(go())
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "A tagged title" and "Little is known" in p.abstract
    assert p.ptype == "review" and "Chickens" in p.keywords and p.year == 2021


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Graph neural networks for
      antimicrobial peptides</title>
    <summary>We study X. The mechanism remains unclear.</summary>
    <arxiv:doi>10.3/abc</arxiv:doi>
    <category term="q-bio.BM"/>
  </entry>
</feed>"""


def test_arxiv_parse():
    total, papers = parse_feed(ATOM)
    assert total == 1
    p = papers[0]
    assert p.title == "Graph neural networks for antimicrobial peptides"
    assert p.year == 2024 and p.doi == "10.3/abc" and p.keywords == ["q-bio.BM"]


def test_arxiv_harvest_with_mock():
    src = ArxivSource(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=ATOM)), delay_s=0)

    async def go():
        out = []
        async for b in src.harvest(demo_brief(), 10):
            out += b
        return out

    assert len(asyncio.run(go())) == 1
