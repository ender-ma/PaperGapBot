# Paper Gap Bot (Telegram)

A Telegram bot that screens tens to hundreds of thousands of papers in **your** niche, finds holes in the
literature, and turns them into novel research questions with evidence and a novelty check.

It **will not start** until your brief is specific: a field, a narrower niche, a focused topic, at least two
intersecting key-term groups, and a matching literature that is neither too broad nor too thin.

```
/new → brief wizard → /check (specificity gate) → /run → questions + full report (HTML + CSV)
```

---

## What it finds

| Signal | What it means | How it is detected |
|---|---|---|
| ❓ Authors flag it as unknown | Papers say "remains unclear", "little is known", "further studies are needed" … about the same thing | Cue-phrase mining in abstracts, grouped by topic |
| ⚖️ Conflicting findings | Repeated "conflicting / inconsistent results" on a topic | Cue phrases |
| 🔗 Unexplored connection | Two topics share many neighbours but are almost never studied together | Co-occurrence matrix, observed vs expected (Swanson-style A–B–C) |
| 🌍 Under-studied setting | A common topic is rare in your context (country, population, organism) | Observed vs expected coverage |
| 🧪 Method not yet applied | A method you name (or a common one) is rarely used on a frequent topic | Observed vs expected coverage |
| 📈 Rising but thin | A topic growing much faster than the niche, still small | Year windows |
| 💤 Highly cited, gone quiet | Topic with high citations but little recent work | Year windows + citations |
| 📚 Weak or unsynthesised evidence | Many primary studies but no review/meta-analysis; mostly in silico / in vitro; mostly cross-sectional | Study-design cues |

An LLM (Claude by default) then writes the best signals up as specific research questions, fitted to your
profile, and must cite the signal ids it used (questions that cite nothing real are dropped). Each question
is then searched again to show how much prior work matches it (🟢 none found · 🟡 some · 🔴 crowded).

Gaps are **leads, not proof**: they come from titles, abstracts and keywords. Always read the cited papers.

---

## Setup (about 15 minutes)

### 1. Get the keys

| Key | Where | Cost |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | In Telegram, message **@BotFather** → `/newbot` → copy the token | Free |
| `OPENALEX_API_KEY` | Make an account at openalex.org → openalex.org/settings/api | Free key = $1/day of usage, about **100,000 papers/day** |
| `ANTHROPIC_API_KEY` (optional) | console.anthropic.com | Roughly a few US cents per run with `claude-sonnet-5-5` |

No AI key? The bot still works and writes template questions from the gap signals.
Biomedical / life-science niche? Add `europepmc` as a source. It is free and needs no key.

### 2. Install (Linux, macOS, or Windows with WSL/Ubuntu)

```bash
# in the folder you unzipped
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env            # paste your keys, save with Ctrl+O, Enter, Ctrl+X
```

### 3. Try the offline demo (no keys needed)

```bash
python -m papergap.demo        # writes reports/demo_report.html from a fake corpus with planted gaps
python -m pytest -q            # 35 tests, all offline
```

### 4. Start the bot

```bash
python -m papergap
```

Open your bot in Telegram and send `/start`. Send `/whoami` to see your user id. To keep the bot private,
put that id in `ALLOWED_USER_IDS` in `.env` and restart.

The bot only runs while this command is running. To keep it on 24/7, run it on a small VPS
(1 GB RAM is enough for runs of up to about 100k papers; use 2 GB for 300k):

```bash
docker build -t paper-gap-bot .
docker run -d --restart unless-stopped --env-file .env -v $PWD/data:/app/data --name paper-gap-bot paper-gap-bot
```

---

## Using it

1. **`/new`**: answer 8 short questions:
   - **Field**: e.g. *Microbiology*
   - **Niche**: e.g. *Antimicrobial resistance in food animals*
   - **Focus**: e.g. *plasmid-mediated colistin resistance (mcr genes) in poultry*
   - **Key-term groups**: they decide which papers are screened. Commas mean AND, `|` means OR:
     `colistin|polymyxin, poultry|chicken|broiler`
   - **Context** (optional): `Nigeria|Nigerian, West Africa|sub-Saharan Africa`
   - **Methods** (optional): `whole genome sequencing|WGS, metagenomics`
   - **Exclude** (optional), **Years** (e.g. `2012-2026`)
2. The **specificity gate** runs automatically. It checks the brief and asks the paper source how many
   papers match. If the brief is too broad, it offers keywords from a sample of the matching papers.
   Tap one to add it as a key-term group.
3. Tap **▶️ Start screening** (or `/run`). One progress message updates as it runs.
4. You get the top questions in chat, then an **HTML report** (opens in any browser) and a **CSV**.

### Personalise

- `/profile stage MSc student` · `/profile skills bioinformatics, Python` ·
  `/profile resources no sequencer; access to farms in Ogun State` · `/profile kinds BRIDGE, CONTEXT`
- `/prompt Prefer questions answerable with public data within 12 months.` Your own instructions for the
  question writer (`/prompt rules` shows the built-in rules).
- `/edit sources openalex,europepmc` · `/edit max_papers 100000` · `/edit years 2015-2026`

### All commands

`/new /brief /edit /check /run /status /cancel /results /more /gaps /report /history /profile /prompt /settings /help /whoami`

---

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | (required) | From @BotFather |
| `ALLOWED_USER_IDS` | empty = anyone | Comma-separated Telegram ids |
| `OPENALEX_API_KEY` | (none) | Required if OpenAlex is a source |
| `OPENALEX_DAILY_BUDGET_USD` | `0.90` | Stops before the free $1/day allowance runs out |
| `DEFAULT_SOURCES` | `openalex` | `openalex`, `europepmc`, `arxiv` |
| `DEFAULT_MAX_PAPERS` / `HARD_MAX_PAPERS` | `20000` / `300000` | Per run |
| `MIN_CORPUS` / `MAX_CORPUS` | `100` / `500000` | Gate: matching-literature size must fall in this range |
| `GATE_MIN_SCORE` | `70` | Specificity score needed (all required checks must also pass) |
| `LLM_PROVIDER` | `auto` | `auto`, `anthropic`, `openai`, `none` |
| `LLM_MODEL` | `claude-sonnet-5-5` | `claude-haiku-5-5` is cheaper |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | (none) | Any OpenAI-compatible endpoint (OpenRouter, Groq, local Ollama …) |
| `QUESTIONS_PER_RUN` | `8` | |
| `KEEP_PAPERS` | `false` | Keep harvested abstracts in the database after a run |
| `MAX_CONCURRENT_JOBS` | `2` | Across all users |

---

## Sources and limits

- **OpenAlex**: all disciplines, about 300M+ works. Since Feb 2026 it needs a free API key. Each page of 100
  search results costs about $0.001 against the $1/day free allowance, so about 100k papers/day. The bot
  reads the remaining allowance before each run and stops early rather than overspend.
- **Europe PMC**: life sciences and biomedicine (PubMed superset). Free, 1,000 records per request.
- **arXiv**: preprints in physics, CS, maths and quantitative biology. Slow (one request every 3 s, about 30k max),
  and has no citation counts.

Speed: mining about 100k papers takes around a minute on a small server. Fetching is the slow part.

## Project layout

```
papergap/
  bot.py            Telegram handlers (wizard, gate, jobs, results, profile, prompt)
  brief.py          Research brief + specificity checks
  gate.py           Gate runner (static checks + corpus probe + optional AI review)
  sources/          openalex.py, europepmc.py, arxiv.py
  mining/           text.py (phrases), cues.py (regexes), engine.py (8 detectors + ranking)
  llm.py            Claude / OpenAI-compatible / template question writer
  novelty.py        Re-searches each question's key terms
  pipeline.py       harvest → mine → questions → novelty → report
  report.py         Telegram messages, HTML report, CSV
  storage.py        SQLite
  demo.py           Synthetic corpus with planted gaps
tests/              35 offline tests (mocked APIs, simulated Telegram chat)
```