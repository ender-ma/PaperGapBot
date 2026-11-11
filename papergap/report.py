"""Report rendering: Telegram messages (HTML parse mode), an HTML file and a CSV."""
from __future__ import annotations

import csv
import datetime as _dt
import html
import os

from .brief import Brief, GateReport
from .mining.engine import KIND_LABELS

TG_LIMIT = 3900
KIND_EMOJI = {
    "EXPLICIT": "❓", "CONTRADICTION": "⚖️", "BRIDGE": "🔗", "CONTEXT": "🌍", "METHOD": "🧪",
    "EMERGING": "📈", "DORMANT": "💤", "EVIDENCE": "📚",
}


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=False)


def esc_attr(text) -> str:
    return html.escape(str(text or ""), quote=True)


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split on blank lines so HTML tags are never cut in half."""
    chunks, cur = [], ""
    for block in text.split("\n\n"):
        piece = block if not cur else "\n\n" + block
        if len(cur) + len(piece) > limit and cur:
            chunks.append(cur)
            cur = block
        else:
            cur += piece
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:  # a single huge block: hard split on newlines
        while len(c) > limit:
            cut = c.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            out.append(c[:cut])
            c = c[cut:].lstrip("\n")
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Telegram text
# ---------------------------------------------------------------------------

def brief_message(brief: Brief, gate: GateReport | None = None, show_gate: bool = True) -> str:
    lines = ["<b>📋 Your research brief</b>"]
    for k, v in brief.summary_lines():
        lines.append(f"<b>{esc(k)}:</b> {esc(v)}")
    if not show_gate:
        return "\n".join(lines)
    if gate and gate.fingerprint == brief.fingerprint():
        state = "✅ passed" if gate.passed else "❌ not passed"
        lines.append(f"\nSpecificity gate: {state} (score {gate.score}/100)")
    else:
        lines.append("\nSpecificity gate: not checked yet. Send /check")
    return "\n".join(lines)


def gate_message(gate: GateReport) -> str:
    head = "✅ <b>Brief is specific enough. Ready to screen.</b>" if gate.passed else \
        "⛔ <b>Not starting yet: the brief needs to be more specific.</b>"
    lines = [head, f"Specificity score: <b>{gate.score}/100</b> (pass mark {gate.min_score}, all hard checks must pass)", ""]
    for c in gate.checks:
        mark = "✅" if c.ok else ("❌" if c.hard else "⚠️")
        tag = " <i>(required)</i>" if c.hard and not c.ok else ""
        lines.append(f"{mark} <b>{esc(c.name)}</b>{tag}\n   {esc(c.detail)}")
    if gate.narrowing_options:
        lines.append("\n<b>Ways to narrow</b> (tap a button below to add one as a key-term group):")
        lines.append(", ".join(esc(o) for o in gate.narrowing_options))
    if gate.llm_advice:
        lines.append(f"\n🤖 <b>Reviewer ({gate.llm_score}/10):</b> {esc(gate.llm_advice)}")
    if not gate.passed:
        lines.append("\nFix it with /edit &lt;field&gt; &lt;value&gt;, e.g. <code>/edit focus ...</code>, then /check")
    return "\n".join(lines)


def _novelty_badge(q: dict) -> str:
    nv = q.get("novelty") or {}
    if not nv:
        return ""
    count = nv.get("count")
    label = nv.get("label", "")
    icon = "🟢" if count == 0 else "🟡" if (count is not None and count <= 30) else "🔴" if count else "⚪"
    return f"{icon} {esc(label)}" + (f" ({count:,} matches)" if count is not None else "")


def question_block(i: int, q: dict, gaps_by_id: dict) -> str:
    sig = []
    for gid in q.get("signal_ids") or []:
        g = gaps_by_id.get(gid)
        if g:
            sig.append(f"{KIND_EMOJI.get(g['kind'], '•')}{gid}")
    lines = [f"<b>{i}. {esc(q.get('title') or 'Question')}</b>", esc(q.get("question"))]
    meta = []
    nb = _novelty_badge(q)
    if nb:
        meta.append(nb)
    if q.get("feasibility"):
        meta.append(f"feasibility: {esc(q['feasibility'])}")
    if meta:
        lines.append(" · ".join(meta))
    if q.get("approach"):
        lines.append(f"<i>Approach:</i> {esc(q['approach'])}")
    if sig:
        lines.append(f"<i>Signals:</i> {' '.join(sig)}")
    # one evidence link
    for gid in q.get("signal_ids") or []:
        g = gaps_by_id.get(gid)
        ev = (g or {}).get("evidence") or []
        if ev and ev[0].get("url"):
            e = ev[0]
            lines.append(f'<a href="{esc_attr(e["url"])}">{esc((e.get("title") or "paper")[:90])}</a> ({esc(e.get("year"))})')
            break
    return "\n".join(lines)


def results_message(job: dict, stats: dict, questions: list[dict], gaps: list[dict], start: int = 0,
                    count: int = 5) -> list[str]:
    gaps_by_id = {g["gid"]: g for g in gaps}
    parts = []
    if start == 0:
        srcs = ", ".join(f"{k} {v:,}" for k, v in (stats.get("sources") or {}).items())
        yrs = f"{stats.get('year_min')}–{stats.get('year_max')}" if stats.get("year_min") else ""
        parts.append(
            f"<b>✅ Screening done</b> (job #{job['job_id']})\n"
            f"{stats.get('n_docs', 0):,} papers screened · {esc(srcs)} · {yrs}\n"
            f"{stats.get('candidates_total', len(gaps))} gap signals found, top {len(gaps)} kept · "
            f"questions by {esc(stats.get('question_writer', ''))}"
        )
    for i, q in enumerate(questions[start: start + count], start + 1):
        parts.append(question_block(i, q, gaps_by_id))
    rest = len(questions) - (start + count)
    if rest > 0:
        parts.append(f"➕ {rest} more: /more")
    else:
        parts.append("Full report with every gap signal and its evidence: /report · Gap list: /gaps")
    return split_message("\n\n".join(parts))


def gaps_message(gaps: list[dict]) -> list[str]:
    parts = ["<b>Gap signals</b> (machine-detected, ranked)"]
    for g in gaps:
        stats = g.get("stats") or {}
        bits = []
        for key in ("papers_flagging", "together", "expected", "observed", "recent", "primary_studies",
                    "preclinical", "weak_design", "papers_mentioning"):
            if key in stats:
                bits.append(f"{key.replace('_', ' ')} {stats[key]}")
        parts.append(
            f"{KIND_EMOJI.get(g['kind'], '•')} <b>{g['gid']}</b> {esc(g['title'])}\n"
            f"<i>{esc(KIND_LABELS.get(g['kind'], g['kind']))}</i>" + (f" · {esc(', '.join(bits))}" if bits else "")
        )
    return split_message("\n\n".join(parts))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#fafaf7;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e6e4dd;--acc:#2f5d50;--g:#2e7d32;--y:#b7791f;--r:#b23b3b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mute:#a3a29b;--line:#33332f;--acc:#7fbfa9}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 60px}h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.2rem;margin:2em 0 .6em;border-bottom:1px solid var(--line);padding-bottom:.3em}
.meta{color:var(--mute);font-size:.92rem}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}
.card h3{margin:0 0 .3em;font-size:1.05rem}.q{font-size:1.02rem;margin:.2em 0 .6em}.row{margin:.25em 0}.lbl{color:var(--mute);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;margin-right:.4em}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.8rem;border:1px solid var(--line);margin:0 4px 4px 0}
.n0{color:var(--g);border-color:var(--g)}.n1{color:var(--y);border-color:var(--y)}.n2{color:var(--r);border-color:var(--r)}
a{color:var(--acc)}ul{padding-left:1.1em;margin:.3em 0}li{margin:.25em 0}.snip{color:var(--mute);font-style:italic}
details{margin:.4em 0}summary{cursor:pointer;color:var(--acc)}table{border-collapse:collapse;width:100%;font-size:.92rem}td{border-top:1px solid var(--line);padding:4px 6px;vertical-align:top}
.kind{font-size:.8rem;color:var(--acc);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
"""


def _novelty_class(q: dict) -> str:
    c = (q.get("novelty") or {}).get("count")
    if c is None:
        return ""
    return "n0" if c == 0 else "n1" if c <= 30 else "n2"


def render_html(brief: Brief, job: dict, stats: dict, questions: list[dict], gaps: list[dict],
                notes: list[str]) -> str:
    gaps_by_id = {g["gid"]: g for g in gaps}
    now = _dt.datetime.now().strftime("%d %b %Y %H:%M")
    srcs = ", ".join(f"{esc(k)} ({v:,})" for k, v in (stats.get("sources") or {}).items())
    out = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Research gaps: {esc(brief.label())}</title><style>{CSS}</style></head><body><main>",
        f"<div class='kind'>Research gap report · job #{esc(job.get('job_id'))}</div>",
        f"<h1>{esc(brief.focus or brief.niche)}</h1>",
        f"<p class='meta'>{stats.get('n_docs', 0):,} papers screened from {srcs} · "
        f"{esc(stats.get('year_min'))}–{esc(stats.get('year_max'))} · generated {now} · "
        f"questions by {esc(stats.get('question_writer', ''))}</p>",
        "<details><summary>Research brief</summary><table>",
    ]
    for k, v in brief.summary_lines():
        out.append(f"<tr><td><b>{esc(k)}</b></td><td>{esc(v)}</td></tr>")
    out.append("</table></details>")

    out.append("<h2>Proposed research questions</h2>")
    for i, q in enumerate(questions, 1):
        nv = q.get("novelty") or {}
        out.append("<div class='card'>")
        out.append(f"<h3>{i}. {esc(q.get('title'))}</h3><p class='q'>{esc(q.get('question'))}</p>")
        badges = []
        if nv:
            cnt = nv.get("count")
            badges.append(f"<span class='badge {_novelty_class(q)}'>{esc(nv.get('label'))}"
                          + (f" · {cnt:,} matches" if cnt is not None else "") + "</span>")
        if q.get("feasibility"):
            badges.append(f"<span class='badge'>feasibility: {esc(q['feasibility'])}</span>")
        for gid in q.get("signal_ids") or []:
            g = gaps_by_id.get(gid)
            if g:
                badges.append(f"<span class='badge'>{KIND_EMOJI.get(g['kind'], '')} {esc(gid)} "
                              f"{esc(KIND_LABELS.get(g['kind'], ''))}</span>")
        out.append("<div>" + "".join(badges) + "</div>")
        for key, label in (("why_gap", "Why it is a gap"), ("novelty_angle", "What is new"),
                           ("approach", "Approach"), ("data_and_resources", "Needs"), ("risks", "Risks")):
            if q.get(key):
                out.append(f"<div class='row'><span class='lbl'>{label}</span>{esc(q[key])}</div>")
        if nv.get("nearest"):
            out.append(f"<details><summary>Closest existing papers for: {esc(' + '.join(nv.get('query') or []))}</summary><ul>")
            for p in nv["nearest"]:
                out.append(f"<li><a href='{esc_attr(p.get('url'))}'>{esc(p.get('title'))}</a> ({esc(p.get('year'))})</li>")
            out.append("</ul></details>")
        out.append("</div>")

    out.append("<h2>Gap signals and evidence</h2>")
    for g in gaps:
        out.append("<div class='card'>")
        out.append(f"<div class='kind'>{KIND_EMOJI.get(g['kind'], '')} {esc(g['gid'])} · "
                   f"{esc(KIND_LABELS.get(g['kind'], g['kind']))}</div>")
        out.append(f"<h3>{esc(g['title'])}</h3><p>{esc(g['question'])}</p>")
        stats_g = {k: v for k, v in (g.get("stats") or {}).items() if v not in (None, "", [])}
        if stats_g:
            out.append("<div class='meta'>" + esc(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats_g.items())) + "</div>")
        if g.get("evidence"):
            out.append("<ul>")
            for e in g["evidence"]:
                snip = f"<br><span class='snip'>“{esc(e['snippet'])}”</span>" if e.get("snippet") else ""
                out.append(f"<li><a href='{esc_attr(e.get('url'))}'>{esc(e.get('title'))}</a> "
                           f"({esc(e.get('year'))}, cited {esc(e.get('cites'))}){snip}</li>")
            out.append("</ul>")
        out.append("</div>")

    out.append("<h2>How to read this</h2><div class='card'><ul>")
    out += [
        "<li><b>Gap signals are leads, not proof.</b> They come from titles, abstracts and keywords, so check full texts, synonyms and recent preprints before committing.</li>",
        "<li><b>Unexplored connection</b>: two topics that share many neighbouring topics but are almost never studied together.</li>",
        "<li><b>Under-studied setting / method</b>: a topic that is common overall but rare in your chosen context or with your chosen method, compared with what chance predicts.</li>",
        "<li><b>Rising but thin / gone quiet</b>: growth or decline relative to the whole niche.</li>",
        "<li><b>Weak or unsynthesised evidence</b>: many primary studies but no review, or mostly in silico / in vitro / cross-sectional work.</li>",
        "<li><b>Novelty badge</b>: how many indexed papers match the question's key terms together. Zero matches means none found with those words, not that none exist.</li>",
    ]
    for n in notes:
        out.append(f"<li>{esc(n)}</li>")
    out.append("</ul></div></main></body></html>")
    return "\n".join(out)


def write_files(reports_dir: str, job_id: int, brief: Brief, job: dict, stats: dict,
                questions: list[dict], gaps: list[dict], notes: list[str]) -> tuple[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "-" for ch in (brief.focus or brief.niche or "brief").lower())[:40].strip("-")
    html_path = os.path.join(reports_dir, f"gaps_{job_id}_{slug}.html")
    csv_path = os.path.join(reports_dir, f"gaps_{job_id}_{slug}.csv")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(brief, job, stats, questions, gaps, notes))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "id", "kind", "title", "question", "approach", "novelty", "novelty_matches",
                    "signals_or_terms", "evidence_1", "evidence_1_url", "evidence_2", "evidence_2_url"])
        for i, q in enumerate(questions, 1):
            nv = q.get("novelty") or {}
            w.writerow(["question", f"Q{i}", "", q.get("title"), q.get("question"), q.get("approach"),
                        nv.get("label", ""), nv.get("count", ""), " ".join(q.get("signal_ids") or []),
                        "", "", "", ""])
        for g in gaps:
            ev = g.get("evidence") or []
            e1 = ev[0] if len(ev) > 0 else {}
            e2 = ev[1] if len(ev) > 1 else {}
            w.writerow(["gap", g["gid"], g["kind"], g["title"], g["question"], g["approach"], "", "",
                        "; ".join(g.get("terms") or []), e1.get("title", ""), e1.get("url", ""),
                        e2.get("title", ""), e2.get("url", "")])
    return html_path, csv_path
