import asyncio
import os

from papergap.config import Settings
from papergap.demo import demo_brief, synthetic_corpus
from papergap.llm import extract_json
from papergap.pipeline import JobProgress, run_job
from papergap.report import split_message
from papergap.storage import Store

from .fakes import FakeLLM, FakeSource


def make_settings(tmp_path):
    return Settings(openalex_api_key="k", db_path=str(tmp_path / "t.db"), reports_dir=str(tmp_path / "rep"),
                    questions_per_run=5)


def test_full_job_with_llm(tmp_path):
    s = make_settings(tmp_path)
    store = Store(s.db_path)
    papers = synthetic_corpus(2000)
    src = FakeSource(papers)
    brief = demo_brief()
    job_id = store.create_job(1, 1, brief)
    prog = JobProgress()
    llm = FakeLLM()
    out = asyncio.run(run_job(job_id, brief, {"stage": "MSc"}, "Prefer field studies", s, store, prog, llm,
                              lambda n, st: src))
    # LLM output: the question citing an unknown signal id is dropped, the valid one keeps only G1
    assert len(out["questions"]) == 1
    q = out["questions"][0]
    assert q["signal_ids"] == ["G1"] and q["novelty"]["count"] == 2
    assert src.searches == [["biosecurity", "mcr-1", "Nigeria"]]
    # prompt carried profile + custom instructions + signals
    prompt = llm.calls[-1][1]
    assert "MSc" in prompt and "Prefer field studies" in prompt and "[G1]" in prompt
    # files + DB
    assert os.path.exists(out["html"]) and os.path.exists(out["csv"])
    job = store.get_job(job_id)
    assert job["status"] == "done"
    gaps, qs = store.get_results(job_id)
    assert len(gaps) > 5 and qs
    assert store.count_papers(job_id) == 0  # abstracts deleted after the run


def test_full_job_without_llm_uses_templates(tmp_path):
    s = make_settings(tmp_path)
    store = Store(s.db_path)
    src = FakeSource(synthetic_corpus(1500))
    job_id = store.create_job(1, 1, demo_brief())
    out = asyncio.run(run_job(job_id, demo_brief(), {}, "", s, store, JobProgress(), None, lambda n, st: src))
    assert len(out["questions"]) == 5
    assert all(q["source"] == "template" for q in out["questions"])
    assert all("novelty" in q for q in out["questions"])


def test_dedupe_across_sources(tmp_path):
    s = make_settings(tmp_path)
    s.questions_per_run = 2
    store = Store(s.db_path)
    papers = synthetic_corpus(800)
    brief = demo_brief()
    brief.sources = ["openalex", "europepmc"]
    brief.max_papers = 5000
    job_id = store.create_job(1, 1, brief)
    out = asyncio.run(run_job(job_id, brief, {}, "", s, store, JobProgress(), None,
                              lambda n, st: FakeSource(papers)))
    assert out["stats"]["n_docs"] == 800  # same pids from both sources counted once


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('blah ```json\n{"a": 2}\n``` blah') == {"a": 2}
    assert extract_json('Sure! {"a": 3} hope that helps') == {"a": 3}
    assert extract_json("nope") is None


def test_split_message_keeps_limit():
    text = "\n\n".join("<b>x</b> " + "y" * 500 for _ in range(30))
    parts = split_message(text, 3900)
    assert all(len(p) <= 3900 for p in parts) and len(parts) > 3
