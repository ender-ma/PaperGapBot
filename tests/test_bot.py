"""Drive the Telegram handlers with mocked updates (no network)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from telegram.ext import ConversationHandler

from papergap.bot import STEPS, WIZ, PaperGapBot
from papergap.config import Settings
from papergap.demo import synthetic_corpus
from papergap.storage import Store

from .fakes import FakeSource


class Chat:
    def __init__(self):
        self.sent: list[str] = []
        self.markups: list = []


def make_env(tmp_path, count=3000, allowed=None):
    s = Settings(telegram_token="123:ABC", openalex_api_key="k", db_path=str(tmp_path / "b.db"),
                 reports_dir=str(tmp_path / "r"), questions_per_run=4)
    if allowed:
        s.allowed_user_ids = allowed
    src = FakeSource(synthetic_corpus(1500), count=count)
    bot = PaperGapBot(s, Store(s.db_path), llm=None, source_factory=lambda n, st: src)
    bot.llm = None
    chat = Chat()
    tasks: list = []

    async def reply_text(text, **kw):
        chat.sent.append(text)
        chat.markups.append(kw.get("reply_markup"))
        return SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())

    async def send_message(chat_id, text, **kw):
        chat.sent.append(text)
        return SimpleNamespace(edit_text=AsyncMock())

    tg_bot = SimpleNamespace(send_message=send_message, send_document=AsyncMock())
    user_data: dict = {}

    def update(text="", args=None, data=None, uid=7):
        msg = SimpleNamespace(text=text, reply_text=reply_text)
        cq = SimpleNamespace(data=data, answer=AsyncMock()) if data else None
        u = SimpleNamespace(effective_user=SimpleNamespace(id=uid, first_name="Ade"), effective_message=msg,
                            effective_chat=SimpleNamespace(id=uid, send_action=AsyncMock()), callback_query=cq)
        ctx = SimpleNamespace(user_data=user_data, args=args or [], bot=tg_bot,
                              application=SimpleNamespace(create_task=lambda coro, update=None: tasks.append(
                                  asyncio.ensure_future(coro))))
        return u, ctx

    return bot, chat, update, tasks, tg_bot


ANSWERS = [
    "Microbiology",
    "Antimicrobial resistance in food animals",
    "plasmid-mediated colistin resistance (mcr genes) in poultry production",
    "colistin|polymyxin, poultry|chicken|broiler",
    "Nigeria|Nigerian, Kenya",
    "whole genome sequencing|WGS, metagenomics",
    None,          # /skip exclude
    "2008-2026",
]


def test_wizard_gate_run_flow(tmp_path):
    bot, chat, update, tasks, tg_bot = make_env(tmp_path)

    async def go():
        u, c = update("/new")
        assert await bot.new(u, c) == WIZ
        state = WIZ
        for ans in ANSWERS:
            u, c = update(ans or "/skip")
            state = await (bot.wiz_skip(u, c) if ans is None else bot.wiz_text(u, c))
        assert state == ConversationHandler.END
        assert any("Ready to screen" in m for m in chat.sent), chat.sent[-1]
        run_buttons = [b for mk in chat.markups if mk for row in mk.inline_keyboard for b in row if b.callback_data == "run"]
        assert run_buttons
        u, c = update(data="run")
        await bot.button(u, c)
        assert tasks
        await asyncio.gather(*tasks)

    asyncio.run(go())
    joined = "\n".join(chat.sent)
    assert "Job #1 started" in joined
    assert "Screening done" in joined, joined[-2000:]
    assert tg_bot.send_document.await_count == 2  # HTML + CSV
    assert bot.store.last_job(7)["status"] == "done"


def test_required_step_cannot_be_skipped(tmp_path):
    bot, chat, update, *_ = make_env(tmp_path)

    async def go():
        u, c = update("/new")
        await bot.new(u, c)
        u, c = update("/skip")
        return await bot.wiz_skip(u, c)

    assert asyncio.run(go()) == WIZ
    assert "required" in chat.sent[-1]


def test_run_refused_for_vague_brief(tmp_path):
    bot, chat, update, tasks, _ = make_env(tmp_path)

    async def go():
        u, c = update("/edit field AI", args=["field", "AI"])
        await bot.edit(u, c)
        u, c = update("/edit focus AI in health", args=["focus", "AI", "in", "health"])
        await bot.edit(u, c)
        u, c = update("/run")
        await bot.run(u, c)

    asyncio.run(go())
    assert not tasks
    assert any("Not starting yet" in m for m in chat.sent)


def test_too_broad_then_narrow_button(tmp_path):
    bot, chat, update, tasks, _ = make_env(tmp_path, count=2_000_000)

    async def go():
        u, c = update("/new")
        await bot.new(u, c)
        for ans in ANSWERS:
            u, c = update(ans or "/skip")
            await (bot.wiz_skip(u, c) if ans is None else bot.wiz_text(u, c))
        assert any("Too broad" in m for m in chat.sent)
        u, c = update(data="narrow:0")
        await bot.button(u, c)

    asyncio.run(go())
    assert "biosecurity" in bot.store.get_brief(7).key_terms


def test_private_bot_blocks_strangers(tmp_path):
    bot, chat, update, *_ = make_env(tmp_path, allowed={1})

    async def go():
        u, c = update("/new", uid=99)
        return await bot.new(u, c)

    assert asyncio.run(go()) == ConversationHandler.END
    assert "private" in chat.sent[-1]


def test_profile_and_prompt(tmp_path):
    bot, chat, update, *_ = make_env(tmp_path)

    async def go():
        u, c = update("/profile skills bioinformatics, R", args=["skills", "bioinformatics,", "R"])
        await bot.profile(u, c)
        u, c = update("/prompt Prefer public data", args=["Prefer", "public", "data"])
        await bot.prompt(u, c)

    asyncio.run(go())
    assert bot.store.get_profile(7)["skills"] == "bioinformatics, R"
    assert bot.store.get_prompt_extra(7) == "Prefer public data"


def test_application_builds(tmp_path):
    bot, *_ = make_env(tmp_path)
    app = bot.build()
    assert len(app.handlers[0]) >= 15
