import asyncio

from papergap.brief import Brief, corpus_check, split_groups, static_checks
from papergap.config import Settings
from papergap.demo import demo_brief
from papergap.gate import evaluate_gate

from .fakes import FakeLLM, FakeSource


def settings(**kw):
    s = Settings(openalex_api_key="k", min_corpus=100, max_corpus=500_000, gate_min_score=70)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_split_groups():
    assert split_groups("colistin|polymyxin, poultry | chicken;broiler") == [
        ["colistin", "polymyxin"], ["poultry", "chicken"], ["broiler"]]
    assert split_groups("") == []


def test_vague_brief_fails_hard_checks():
    b = Brief(field="Medicine", niche="Medicine", focus="AI in healthcare", key_terms="artificial intelligence")
    failed = {c.name for c in static_checks(b) if c.hard and not c.ok}
    assert "Niche narrows the field" in failed
    assert "Focus is specific (3+ content words)" in failed
    assert "At least 2 key-term groups (intersection)" in failed


def test_good_brief_passes_static_checks():
    checks = static_checks(demo_brief())
    assert all(c.ok for c in checks if c.hard)


def test_corpus_bounds():
    assert not corpus_check(10, 100, 1000, "x").ok
    assert not corpus_check(5000, 100, 1000, "x").ok
    assert corpus_check(500, 100, 1000, "x").ok


def test_gate_passes_with_probe():
    src = FakeSource(count=40_000)
    rep = asyncio.run(evaluate_gate(demo_brief(), settings(), FakeLLM(), lambda n, s: src))
    assert rep.passed, rep.to_dict()
    assert rep.corpus_count == 40_000
    assert rep.llm_score == 8


def test_gate_too_broad_offers_narrowing():
    src = FakeSource(count=2_000_000)
    rep = asyncio.run(evaluate_gate(demo_brief(), settings(), None, lambda n, s: src))
    assert not rep.passed
    assert "biosecurity" in rep.narrowing_options
    assert "colistin" not in rep.narrowing_options  # already a key term


def test_gate_does_not_probe_when_static_fails():
    called = []

    def factory(n, s):
        called.append(n)
        return FakeSource()

    rep = asyncio.run(evaluate_gate(Brief(field="AI"), settings(), None, factory))
    assert not rep.passed and not called


def test_openalex_without_key_is_blocked():
    b = demo_brief()
    rep = asyncio.run(evaluate_gate(b, settings(openalex_api_key=None), None, lambda n, s: FakeSource()))
    assert not rep.passed
    assert any("API key" in c.name for c in rep.hard_failures)


def test_fingerprint_changes_with_edit():
    b = demo_brief()
    fp = b.fingerprint()
    assert b.set_field("years", "2015-2025") is None
    assert b.fingerprint() != fp
    assert b.set_field("sources", "openalex, europepmc") is None and b.sources == ["openalex", "europepmc"]
    assert b.set_field("sources", "google") is not None
    assert b.set_field("max_papers", "50,000") is None and b.max_papers == 50000
