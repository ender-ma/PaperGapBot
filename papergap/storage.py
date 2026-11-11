"""SQLite storage: users, briefs, jobs, harvested papers and results."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Iterable, Iterator

from .brief import Brief, GateReport
from .sources.base import Paper

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    profile_json TEXT DEFAULT '{}',
    prompt_extra TEXT DEFAULT '',
    brief_json TEXT DEFAULT '{}',
    gate_json TEXT DEFAULT '',
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    chat_id INTEGER,
    status TEXT,
    brief_json TEXT,
    created_at REAL,
    finished_at REAL,
    stats_json TEXT DEFAULT '{}',
    error TEXT,
    report_path TEXT,
    csv_path TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    job_id INTEGER,
    pid TEXT,
    source TEXT,
    title TEXT,
    abstract TEXT,
    year INTEGER,
    cited_by INTEGER,
    doi TEXT,
    url TEXT,
    keywords_json TEXT,
    ptype TEXT,
    PRIMARY KEY (job_id, pid)
);
CREATE TABLE IF NOT EXISTS results (
    job_id INTEGER PRIMARY KEY,
    gaps_json TEXT,
    questions_json TEXT
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---- users ------------------------------------------------------------
    def _ensure_user(self, user_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO users (user_id, updated_at) VALUES (?, ?)", (user_id, time.time())
        )

    def get_user(self, user_id: int) -> dict:
        with self._lock:
            self._ensure_user(user_id)
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row)

    def get_brief(self, user_id: int, default_sources: Iterable[str] = ("openalex",),
                  default_max: int = 20_000) -> Brief:
        u = self.get_user(user_id)
        data = json.loads(u["brief_json"] or "{}")
        if not data:
            return Brief(sources=list(default_sources), max_papers=default_max)
        return Brief.from_dict(data)

    def save_brief(self, user_id: int, brief: Brief) -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.conn.execute(
                "UPDATE users SET brief_json=?, updated_at=? WHERE user_id=?",
                (json.dumps(brief.to_dict()), time.time(), user_id),
            )
            self.conn.commit()

    def get_gate(self, user_id: int) -> GateReport | None:
        u = self.get_user(user_id)
        return GateReport.from_dict(json.loads(u["gate_json"])) if u["gate_json"] else None

    def save_gate(self, user_id: int, gate: GateReport | None) -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.conn.execute(
                "UPDATE users SET gate_json=? WHERE user_id=?",
                (json.dumps(gate.to_dict()) if gate else "", user_id),
            )
            self.conn.commit()

    def get_profile(self, user_id: int) -> dict:
        return json.loads(self.get_user(user_id)["profile_json"] or "{}")

    def save_profile(self, user_id: int, profile: dict) -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.conn.execute("UPDATE users SET profile_json=? WHERE user_id=?", (json.dumps(profile), user_id))
            self.conn.commit()

    def get_prompt_extra(self, user_id: int) -> str:
        return self.get_user(user_id)["prompt_extra"] or ""

    def save_prompt_extra(self, user_id: int, text: str) -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.conn.execute("UPDATE users SET prompt_extra=? WHERE user_id=?", (text, user_id))
            self.conn.commit()

    # ---- jobs ---------------------------------------------------------------
    def create_job(self, user_id: int, chat_id: int, brief: Brief) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO jobs (user_id, chat_id, status, brief_json, created_at) VALUES (?,?,?,?,?)",
                (user_id, chat_id, "running", json.dumps(brief.to_dict()), time.time()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields) -> None:
        if not fields:
            return
        if "stats" in fields:
            fields["stats_json"] = json.dumps(fields.pop("stats"))
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self.conn.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", (*fields.values(), job_id))
            self.conn.commit()

    def get_job(self, job_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def last_job(self, user_id: int, status: str | None = None) -> dict | None:
        q = "SELECT * FROM jobs WHERE user_id=?"
        args: list = [user_id]
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY job_id DESC LIMIT 1"
        with self._lock:
            row = self.conn.execute(q, args).fetchone()
        return dict(row) if row else None

    def list_jobs(self, user_id: int, limit: int = 10) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE user_id=? ORDER BY job_id DESC LIMIT ?", (user_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_interrupted_jobs(self) -> int:
        """Jobs left 'running' by a crash/restart are marked as interrupted."""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE jobs SET status='interrupted', error='Bot restarted during the run' WHERE status='running'"
            )
            self.conn.commit()
            return cur.rowcount

    # ---- papers -------------------------------------------------------------
    def add_papers(self, job_id: int, papers: list[Paper]) -> int:
        rows = [
            (job_id, p.pid, p.source, p.title, p.abstract, p.year, p.cited_by, p.doi, p.url,
             json.dumps(p.keywords), p.ptype)
            for p in papers if (p.title or p.abstract)
        ]
        with self._lock:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO papers (job_id, pid, source, title, abstract, year, cited_by, doi, url,"
                " keywords_json, ptype) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()
            return self.conn.total_changes - before

    def count_papers(self, job_id: int) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM papers WHERE job_id=?", (job_id,)).fetchone()[0])

    def iter_papers(self, job_id: int) -> Iterator[Paper]:
        """Stream papers with a dedicated connection (safe to call from a worker thread)."""
        conn = sqlite3.connect(self.path)
        try:
            cur = conn.execute(
                "SELECT pid, source, title, abstract, year, cited_by, doi, url, keywords_json, ptype "
                "FROM papers WHERE job_id=? ORDER BY rowid",
                (job_id,),
            )
            for r in cur:
                yield Paper(
                    pid=r[0], source=r[1], title=r[2] or "", abstract=r[3] or "", year=r[4],
                    cited_by=r[5] or 0, doi=r[6], url=r[7], keywords=json.loads(r[8] or "[]"), ptype=r[9],
                )
        finally:
            conn.close()

    def delete_papers(self, job_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM papers WHERE job_id=?", (job_id,))
            self.conn.commit()

    # ---- results --------------------------------------------------------------
    def save_results(self, job_id: int, gaps: list[dict], questions: list[dict]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO results (job_id, gaps_json, questions_json) VALUES (?,?,?)",
                (job_id, json.dumps(gaps), json.dumps(questions)),
            )
            self.conn.commit()

    def get_results(self, job_id: int) -> tuple[list[dict], list[dict]] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM results WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        return json.loads(row["gaps_json"]), json.loads(row["questions_json"])
