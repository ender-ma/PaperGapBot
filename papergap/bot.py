"""Telegram interface."""
from __future__ import annotations

import asyncio
import datetime as _dt
import functools
import html
import logging
import os
import time
from typing import Callable

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application, ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes,
    ConversationHandler, MessageHandler, filters,
)

from .brief import BRIEF_FIELDS, FIELD_HELP, Brief
from .config import Settings
from .gate import evaluate_gate
from .llm import SYSTEM_RULES, LLMClient, make_llm
from .mining.engine import KIND_LABELS
from .pipeline import JobCancelled, JobProgress, run_job
from .report import brief_message, esc, gaps_message, gate_message, results_message, split_message
from .sources import PaperSource, make_source
from .storage import Store

log = logging.getLogger(__name__)

WIZ = 1
STEPS: list[tuple[str, str, bool]] = [
    ("field", "1/8 <b>Field</b>: your broad discipline.\n<i>e.g. Microbiology</i>", True),
    ("niche", "2/8 <b>Niche</b>: the sub-field you work in, narrower than the field.\n"
              "<i>e.g. Antimicrobial resistance in food animals</i>", True),
    ("focus", "3/8 <b>Focus</b>: the exact topic, in at least 3 meaningful words.\n"
              "<i>e.g. plasmid-mediated colistin resistance (mcr genes) in poultry</i>", True),
    ("key_terms", "4/8 <b>Key-term groups</b>: these decide which papers get screened.\n"
                  "Commas separate groups that must ALL appear; <code>|</code> separates synonyms.\n"
                  "<i>e.g. colistin|polymyxin, poultry|chicken|broiler</i>", True),
    ("context", "5/8 <b>Context</b> (optional): places, populations or organisms to test coverage for.\n"
                "<i>e.g. Nigeria|Nigerian, West Africa|sub-Saharan Africa</i>\n/skip to leave empty", False),
    ("methods", "6/8 <b>Methods</b> (optional): methods you could apply.\n"
                "<i>e.g. whole genome sequencing|WGS, metagenomics</i>\n/skip to leave empty", False),
    ("exclude", "7/8 <b>Exclude</b> (optional): terms to leave out.\n<i>e.g. vaccine, human clinical</i>\n"
                "/skip to leave empty", False),
    ("years", "8/8 <b>Years</b>: range to screen, e.g. <code>2012-2026</code>.\n/skip = last 15 years", False),
]
PROFILE_KEYS = {
    "stage": "career stage, e.g. MSc student, early-career researcher",
    "skills": "what you can do, e.g. bioinformatics, wet lab, field sampling",
    "resources": "lab, data, budget, e.g. no sequencer; access to poultry farms in Ogun State",
    "goals": "e.g. MSc thesis topic, grant proposal, first-author paper",
    "avoid": "things you don't want, e.g. animal experiments",
    "kinds": "gap types to favour: " + ", ".join(KIND_LABELS),
    "avoid_kinds": "gap types to down-rank",
}
HELP = """<b>Paper Gap Bot</b> screens up to hundreds of thousands of papers in your niche, finds gaps, and turns them into research questions.

<b>How it works</b>
1. /new: build a research brief (field → niche → focus → key terms …)
2. /check: the <b>specificity gate</b>. The bot will not start until your brief defines a real niche and the matching literature is neither too broad nor too thin.
3. /run: harvest papers, detect gaps, write questions, check novelty.

<b>Brief</b>
/brief: show it · /edit &lt;field&gt; &lt;value&gt;: change one field
/check: run the gate · /run: start screening

<b>Results</b>
/status · /cancel · /results · /more · /gaps · /report · /history

<b>Personalise</b>
/profile: your stage, skills, resources (shapes the questions)
/prompt: your own instructions for the question writer
/settings: sources, limits and AI setup

<b>Gap types</b>
❓ authors say it's unknown · ⚖️ conflicting findings · 🔗 unexplored connection · 🌍 under-studied setting · 🧪 method not yet applied · 📈 rising but thin · 💤 highly cited, gone quiet · 📚 no review / weak evidence"""


def _years_default() -> tuple[int, int]:
    y = _dt.date.today().year
    return y - 14, y


class PaperGapBot:
    def __init__(self, settings: Settings, store: Store | None = None, llm: LLMClient | None = None,
                 source_factory: Callable[[str, Settings], PaperSource] = make_source):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.llm = llm if llm is not None else make_llm(settings)
        self.source_factory = source_factory
        self.running: dict[int, tuple[int, JobProgress]] = {}  # user_id -> (job_id, progress)
        self.job_slots = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "2")))

    # ------------------------------------------------------------------ utils
    def allowed(self, update: Update) -> bool:
        ids = self.settings.allowed_user_ids
        return not ids or (update.effective_user and update.effective_user.id in ids)

    @staticmethod
    def guard(fn):
        @functools.wraps(fn)
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *a, **kw):
            if not self.allowed(update):
                uid = update.effective_user.id if update.effective_user else "?"
                if update.effective_message:
                    await update.effective_message.reply_text(
                        f"This bot is private. Your Telegram user id is {uid}; ask the owner to add it to ALLOWED_USER_IDS."
                    )
                return ConversationHandler.END
            return await fn(self, update, context, *a, **kw)
        return wrapper

    async def say(self, update: Update, text: str, markup=None, preview: bool = False) -> None:
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            await update.effective_message.reply_text(
                chunk, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=not preview),
                reply_markup=markup if i == len(chunks) - 1 else None,
            )

    def brief_of(self, user_id: int) -> Brief:
        return self.store.get_brief(user_id, self.settings.default_sources, self.settings.default_max_papers)

    # ------------------------------------------------------------------ basics
    @guard
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        name = esc(update.effective_user.first_name or "there")
        await self.say(update, f"Hi {name} 👋\n\n" + HELP + "\n\nStart with /new")

    @guard
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.say(update, HELP)

    async def whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(f"Your Telegram user id: {update.effective_user.id}")

    @guard
    async def settings_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.settings
        llm = f"{self.llm.name} ({getattr(self.llm, 'model', '')})" if self.llm else "none: template questions"
        brief = self.brief_of(update.effective_user.id)
        await self.say(update, "\n".join([
            "<b>⚙️ Settings</b>",
            f"Question writer: {esc(llm)}",
            f"OpenAlex key: {'✅ set' if s.openalex_api_key else '❌ missing (get one free at openalex.org/settings/api)'}",
            f"OpenAlex daily budget cap: ${s.openalex_daily_budget_usd:.2f} (~{int(s.openalex_daily_budget_usd * 100_000):,} papers)",
            f"Your sources: {esc(', '.join(brief.sources))} (change: <code>/edit sources openalex,europepmc</code>)",
            f"Your max papers: {brief.max_papers:,} (hard cap {s.hard_max_papers:,}; change: <code>/edit max_papers 50000</code>)",
            f"Corpus must be {s.min_corpus:,}–{s.max_corpus:,} papers to pass the gate; pass mark {s.gate_min_score}/100",
        ]))

    # ------------------------------------------------------------------ wizard
    @guard
    async def new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        old = self.brief_of(update.effective_user.id)
        context.user_data["wiz"] = Brief(sources=old.sources, max_papers=old.max_papers).to_dict()
        context.user_data["wiz_step"] = 0
        await self.say(update, "Let's build your research brief. Send /cancel to stop.\n\n" + STEPS[0][1])
        return WIZ

    async def _wiz_next(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        step = context.user_data["wiz_step"] + 1
        context.user_data["wiz_step"] = step
        if step < len(STEPS):
            await self.say(update, STEPS[step][1])
            return WIZ
        brief = Brief.from_dict(context.user_data.pop("wiz"))
        if not brief.year_from:
            brief.year_from, brief.year_to = _years_default()
        uid = update.effective_user.id
        self.store.save_brief(uid, brief)
        self.store.save_gate(uid, None)
        await self.say(update, brief_message(brief, show_gate=False))
        await self._do_check(update, context)
        return ConversationHandler.END

    @guard
    async def wiz_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        step = context.user_data.get("wiz_step", 0)
        name = STEPS[step][0]
        brief = Brief.from_dict(context.user_data.get("wiz"))
        err = brief.set_field(name, update.effective_message.text or "")
        if err:
            await self.say(update, f"⚠️ {esc(err)}\n\n" + STEPS[step][1])
            return WIZ
        context.user_data["wiz"] = brief.to_dict()
        return await self._wiz_next(update, context)

    @guard
    async def wiz_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        step = context.user_data.get("wiz_step", 0)
        if STEPS[step][2]:
            await self.say(update, "This step is required.\n\n" + STEPS[step][1])
            return WIZ
        return await self._wiz_next(update, context)

    async def wiz_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.pop("wiz", None)
        await update.effective_message.reply_text("Brief wizard cancelled. Your previous brief is unchanged.")
        return ConversationHandler.END

    # ------------------------------------------------------------------ brief
    @guard
    async def brief_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        await self.say(update, brief_message(self.brief_of(uid), self.store.get_gate(uid)))

    @guard
    async def edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        if len(context.args) < 2:
            lines = ["<b>Edit one field:</b> <code>/edit &lt;field&gt; &lt;value&gt;</code>", ""]
            lines += [f"<b>{f}</b>: {esc(FIELD_HELP[f])}" for f in BRIEF_FIELDS]
            await self.say(update, "\n".join(lines))
            return
        name = context.args[0]
        value = (update.effective_message.text or "").split(None, 2)[2]
        brief = self.brief_of(uid)
        err = brief.set_field(name, value)
        if err:
            await self.say(update, f"⚠️ {esc(err)}")
            return
        if brief.max_papers > self.settings.hard_max_papers:
            brief.max_papers = self.settings.hard_max_papers
        self.store.save_brief(uid, brief)
        await self.say(update, f"✏️ Updated <b>{esc(name)}</b>. The brief changed, so run /check again before /run.")

    # ------------------------------------------------------------------ gate
    @guard
    async def check(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._do_check(update, context)

    async def _do_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        brief = self.brief_of(uid)
        await update.effective_chat.send_action(ChatAction.TYPING)
        note = await update.effective_message.reply_text("🔍 Checking how specific your brief is…")
        gate = await evaluate_gate(brief, self.settings, self.llm, self.source_factory)
        self.store.save_gate(uid, gate)
        context.user_data["narrow_opts"] = gate.narrowing_options
        rows = []
        if gate.passed:
            rows.append([InlineKeyboardButton(f"▶️ Start screening ({min(brief.max_papers, gate.corpus_count or 0):,} papers)",
                                              callback_data="run")])
        for i, opt in enumerate(gate.narrowing_options[:6]):
            rows.append([InlineKeyboardButton(f"➕ {opt[:40]}", callback_data=f"narrow:{i}")])
        try:
            await note.delete()
        except BadRequest:
            pass
        await self.say(update, gate_message(gate), InlineKeyboardMarkup(rows) if rows else None)
        return gate

    # ------------------------------------------------------------------ run
    @guard
    async def run(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_run(update, context)

    async def _start_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        if uid in self.running:
            job_id, prog = self.running[uid]
            await self.say(update, f"Job #{job_id} is still running:\n{esc(prog.line())}\n\n/cancel to stop it.")
            return
        brief = self.brief_of(uid)
        gate = self.store.get_gate(uid)
        just_checked = False
        if gate is None or gate.fingerprint != brief.fingerprint():
            await self.say(update, "Your brief hasn't been checked since it last changed. Checking now…")
            gate = await self._do_check(update, context)
            just_checked = True
        if not gate.passed:
            if not just_checked:
                await self.say(update, "⛔ The specificity gate hasn't passed, so screening won't start. "
                                       "Edit the brief (/edit) and /check again.")
            return
        chat_id = update.effective_chat.id
        job_id = self.store.create_job(uid, chat_id, brief)
        prog = JobProgress(target=min(brief.max_papers, self.settings.hard_max_papers))
        self.running[uid] = (job_id, prog)
        await self.say(update, (
            f"🚀 <b>Job #{job_id} started.</b> Screening up to {prog.target:,} papers from "
            f"{esc(', '.join(brief.sources))}. Large runs take a while (≈1–3 min per 10k papers to fetch).\n"
            "I'll keep one progress message updated. /status any time, /cancel to stop."
        ))
        context.application.create_task(self._job(context, uid, chat_id, job_id, brief, prog), update=update)

    async def _job(self, context: ContextTypes.DEFAULT_TYPE, uid: int, chat_id: int, job_id: int,
                   brief: Brief, prog: JobProgress) -> None:
        bot = context.bot
        status = await bot.send_message(chat_id, "⏳ Queued…")
        stop = asyncio.Event()

        async def ticker():
            last = ""
            while not stop.is_set():
                text = "⏳ " + prog.line()
                if text != last:
                    try:
                        await status.edit_text(text)
                        last = text
                    except BadRequest:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        log.debug("status edit failed: %s", exc)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass

        tick = asyncio.create_task(ticker())
        try:
            async with self.job_slots:
                result = await run_job(
                    job_id, brief, self.store.get_profile(uid), self.store.get_prompt_extra(uid),
                    self.settings, self.store, prog, self.llm, self.source_factory,
                )
            stop.set()
            await tick
            try:
                await status.edit_text(f"✅ Job #{job_id} finished in {int((time.time() - prog.started) // 60)} min.")
            except BadRequest:
                pass
            job = self.store.get_job(job_id)
            context.user_data["results_job"] = job_id
            context.user_data["results_start"] = 0
            msgs = results_message(job, result["stats"], result["questions"], result["gaps"], 0, 5)
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("➕ More", callback_data="more"),
                                        InlineKeyboardButton("📄 Full report", callback_data="report")]])
            for i, m in enumerate(msgs):
                await bot.send_message(chat_id, m, parse_mode=ParseMode.HTML,
                                       link_preview_options=LinkPreviewOptions(is_disabled=True),
                                       reply_markup=kb if i == len(msgs) - 1 else None)
            if result["notes"]:
                await bot.send_message(chat_id, "ℹ️ Notes:\n" + "\n".join(f"• {n}" for n in result["notes"])[:3800])
            await self._send_files(bot, chat_id, job)
        except JobCancelled:
            self.store.update_job(job_id, status="cancelled", finished_at=time.time())
            await bot.send_message(chat_id, f"🛑 Job #{job_id} cancelled.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s failed", job_id)
            self.store.update_job(job_id, status="failed", finished_at=time.time(), error=str(exc)[:500])
            await bot.send_message(chat_id, f"❌ Job #{job_id} failed: {exc}"[:3500])
        finally:
            stop.set()
            if not tick.done():
                tick.cancel()
            self.running.pop(uid, None)
            if not self.settings.keep_papers:
                self.store.delete_papers(job_id)

    async def _send_files(self, bot, chat_id: int, job: dict) -> None:
        for key, caption in (("report_path", "Full report (open in your browser)"),
                             ("csv_path", "Spreadsheet of questions and gap signals")):
            path = job.get(key)
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    await bot.send_document(chat_id, document=f, filename=os.path.basename(path), caption=caption)

    @guard
    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        if uid in self.running:
            job_id, prog = self.running[uid]
            await self.say(update, f"⏳ Job #{job_id}\n{esc(prog.line())}")
            return
        job = self.store.last_job(uid)
        if not job:
            await self.say(update, "No jobs yet. Build a brief with /new.")
            return
        extra = f"\nError: {esc(job['error'])}" if job.get("error") else ""
        await self.say(update, f"Last job #{job['job_id']}: <b>{esc(job['status'])}</b>{extra}")

    @guard
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        if uid in self.running:
            self.running[uid][1].cancelled = True
            await self.say(update, "Stopping after the current step…")
        else:
            await self.say(update, "Nothing is running.")

    # ------------------------------------------------------------------ results
    def _results_for(self, uid: int, context, arg: str | None = None):
        job = None
        if arg and arg.isdigit():
            job = self.store.get_job(int(arg))
            if job and job["user_id"] != uid:
                job = None
        if job is None:
            jid = context.user_data.get("results_job")
            job = self.store.get_job(jid) if jid else None
        if job is None or job.get("status") != "done":
            job = self.store.last_job(uid, status="done")
        if not job:
            return None, None
        res = self.store.get_results(job["job_id"])
        return job, res

    @guard
    async def results(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        import json

        uid = update.effective_user.id
        job, res = self._results_for(uid, context, context.args[0] if context.args else None)
        if not res:
            await self.say(update, "No finished jobs yet.")
            return
        gaps, questions = res
        context.user_data["results_job"] = job["job_id"]
        context.user_data["results_start"] = 0
        stats = json.loads(job.get("stats_json") or "{}")
        for m in results_message(job, stats, questions, gaps, 0, 5):
            await self.say(update, m)

    async def _more(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        import json

        uid = update.effective_user.id
        job, res = self._results_for(uid, context)
        if not res:
            await self.say(update, "No finished jobs yet.")
            return
        gaps, questions = res
        start = context.user_data.get("results_start", 0) + 5
        if start >= len(questions):
            await self.say(update, "That's all the questions. /gaps lists every gap signal; /report sends the full report.")
            return
        context.user_data["results_start"] = start
        stats = json.loads(job.get("stats_json") or "{}")
        for m in results_message(job, stats, questions, gaps, start, 5):
            await self.say(update, m)

    @guard
    async def more(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._more(update, context)

    @guard
    async def gaps(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        job, res = self._results_for(update.effective_user.id, context, context.args[0] if context.args else None)
        if not res:
            await self.say(update, "No finished jobs yet.")
            return
        for m in gaps_message(res[0]):
            await self.say(update, m)

    @guard
    async def report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        job, _ = self._results_for(update.effective_user.id, context, context.args[0] if context.args else None)
        if not job:
            await self.say(update, "No finished jobs yet.")
            return
        await self._send_files(context.bot, update.effective_chat.id, job)

    @guard
    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        import json

        jobs = self.store.list_jobs(update.effective_user.id, 10)
        if not jobs:
            await self.say(update, "No jobs yet.")
            return
        lines = ["<b>Recent jobs</b>"]
        for j in jobs:
            b = Brief.from_dict(json.loads(j["brief_json"] or "{}"))
            when = _dt.datetime.fromtimestamp(j["created_at"]).strftime("%d %b %H:%M")
            lines.append(f"#{j['job_id']} · {esc(j['status'])} · {when}\n   {esc(b.label()[:80])}")
        lines.append("\nOpen one: /results &lt;job id&gt;")
        await self.say(update, "\n".join(lines))

    # ------------------------------------------------------------------ personalisation
    @guard
    async def profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        prof = self.store.get_profile(uid)
        if context.args and context.args[0].lower() == "clear":
            self.store.save_profile(uid, {})
            await self.say(update, "Profile cleared.")
            return
        if len(context.args) >= 2:
            key = context.args[0].lower()
            if key not in PROFILE_KEYS:
                await self.say(update, "Unknown key. Use one of: " + ", ".join(PROFILE_KEYS))
                return
            prof[key] = (update.effective_message.text or "").split(None, 2)[2].strip()
            self.store.save_profile(uid, prof)
            await self.say(update, f"👤 Saved <b>{esc(key)}</b>.")
            return
        lines = ["<b>👤 Researcher profile</b> (shapes which questions are proposed)"]
        for k, hint in PROFILE_KEYS.items():
            lines.append(f"<b>{k}</b>: {esc(prof.get(k, '—'))}\n   <i>{esc(hint)}</i>")
        lines.append("\nSet one: <code>/profile skills bioinformatics, R, Python</code> · <code>/profile clear</code>")
        await self.say(update, "\n".join(lines))

    @guard
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        if context.args and context.args[0].lower() == "reset":
            self.store.save_prompt_extra(uid, "")
            await self.say(update, "Custom instructions cleared.")
            return
        if context.args and context.args[0].lower() == "rules":
            await self.say(update, "<b>Built-in rules for the question writer</b>\n<pre>" + esc(SYSTEM_RULES) + "</pre>")
            return
        if context.args:
            text = (update.effective_message.text or "").split(None, 1)[1].strip()
            self.store.save_prompt_extra(uid, text[:3000])
            await self.say(update, "📝 Saved. These instructions go to the question writer on every run.")
            return
        current = self.store.get_prompt_extra(uid)
        await self.say(update, "\n".join([
            "<b>📝 Your instructions for the question writer</b>",
            esc(current) if current else "<i>(none)</i>",
            "",
            "Set: <code>/prompt Prefer questions answerable with public genomic data within 12 months; "
            "write for an MSc thesis.</code>",
            "/prompt rules: see the built-in rules · /prompt reset: clear",
        ]))

    # ------------------------------------------------------------------ buttons
    @guard
    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        await q.answer()
        data = q.data or ""
        uid = update.effective_user.id
        if data == "run":
            await self._start_run(update, context)
        elif data.startswith("narrow:"):
            opts = context.user_data.get("narrow_opts") or []
            i = int(data.split(":", 1)[1])
            if i >= len(opts):
                await self.say(update, "That option expired; run /check again.")
                return
            brief = self.brief_of(uid)
            brief.key_terms = (brief.key_terms + ", " + opts[i]).strip(", ")
            self.store.save_brief(uid, brief)
            await self.say(update, f"➕ Added key-term group <b>{esc(opts[i])}</b>. Re-checking…")
            await self._do_check(update, context)
        elif data == "more":
            await self._more(update, context)
        elif data == "report":
            job, _ = self._results_for(uid, context)
            if job:
                await self._send_files(context.bot, update.effective_chat.id, job)

    # ------------------------------------------------------------------ errors
    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.exception("Handler error", exc_info=context.error)
        if isinstance(update, Update) and update.effective_chat:
            try:
                await context.bot.send_message(update.effective_chat.id,
                                               f"⚠️ Something went wrong: {html.escape(str(context.error))[:300]}")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ wiring
    def build(self) -> Application:
        if not self.settings.telegram_token:
            raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")

        async def post_init(app: Application) -> None:
            n = self.store.mark_interrupted_jobs()
            if n:
                log.info("Marked %d interrupted job(s)", n)
            await app.bot.set_my_commands([
                BotCommand("new", "Build a research brief"),
                BotCommand("brief", "Show your brief"),
                BotCommand("edit", "Change one field of the brief"),
                BotCommand("check", "Run the specificity gate"),
                BotCommand("run", "Start screening"),
                BotCommand("status", "Progress of the current job"),
                BotCommand("cancel", "Stop the current job"),
                BotCommand("results", "Show research questions"),
                BotCommand("more", "Next questions"),
                BotCommand("gaps", "List gap signals"),
                BotCommand("report", "Send the full report"),
                BotCommand("history", "Past jobs"),
                BotCommand("profile", "Your researcher profile"),
                BotCommand("prompt", "Your instructions for the AI"),
                BotCommand("settings", "Sources, limits, AI setup"),
                BotCommand("help", "How it works"),
            ])

        app = ApplicationBuilder().token(self.settings.telegram_token).post_init(post_init).build()
        wizard = ConversationHandler(
            entry_points=[CommandHandler("new", self.new)],
            states={WIZ: [
                CommandHandler("skip", self.wiz_skip),
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.wiz_text),
            ]},
            fallbacks=[CommandHandler("cancel", self.wiz_cancel)],
            name="brief_wizard",
        )
        app.add_handler(wizard)
        for cmd, fn in [
            ("start", self.start), ("help", self.help), ("whoami", self.whoami),
            ("settings", self.settings_cmd), ("brief", self.brief_cmd), ("edit", self.edit),
            ("check", self.check), ("run", self.run), ("status", self.status_cmd),
            ("cancel", self.cancel), ("results", self.results), ("more", self.more), ("gaps", self.gaps),
            ("report", self.report), ("history", self.history), ("profile", self.profile),
            ("prompt", self.prompt),
        ]:
            app.add_handler(CommandHandler(cmd, fn))
        app.add_handler(CallbackQueryHandler(self.button))
        app.add_error_handler(self.on_error)
        return app


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    bot = PaperGapBot(settings)
    log.info("Question writer: %s", bot.llm.name if bot.llm else "templates (no LLM)")
    app = bot.build()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
