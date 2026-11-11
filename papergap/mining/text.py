"""Light-weight text utilities (no heavy NLP dependencies)."""
from __future__ import annotations

import re

STOPWORDS = set("""
a about above after again against all almost also although always am among an and any are aren't as at be
because been before being below between both but by can cannot could couldn't did didn't do does doesn't doing
don't down during each either else et al etc few for from further had hadn't has hasn't have haven't having he her
here hers herself him himself his how however i if in into is isn't it its itself just let me more most much must
my myself neither no nor not now of off often on once only or other others otherwise our ours ourselves out over
own per rather same several she should shouldn't since so some such than that the their theirs them themselves
then there therefore these they this those though through thus to too toward towards under until up upon us very
via was wasn't we were weren't what when where whereas whether which while who whom whose why will with within
without would wouldn't yet you your yours yourself yourselves vs versus i.e e.g ie eg one two three four five six
seven eight nine ten first second third also may might shall
""".split())

# Academic boilerplate: never a term on its own, never at the edge of an n-gram.
BOILERPLATE = set("""
study studies studied result results method methods conclusion conclusions background objective objectives aim
aims aimed purpose finding findings paper papers article articles review reviews data analysis analyses analyzed
analysed approach approaches use used uses using usage based show shows showed shown found find present presents
presented propose proposed new novel significant significantly high higher highest low lower lowest increase
increased increases increasing decrease decreased decreasing different various well total respectively compared
comparison including include includes included associated association related relationship potential important
importance level levels role factor factors group groups year years time times rate rates number numbers
evaluate evaluated evaluation investigate investigated investigation examine examined assess assessed assessment
determine determined report reported reports suggest suggests suggested indicate indicates indicated observed
demonstrate demonstrated revealed provide provides provided obtained performed conducted current recent recently
present-day overall main major key general specific particular common commonly mainly likely possible possibly
effect effects affect affected among total value values case cases type types form forms within whereas thus
author authors work works literature evidence research researchers field understanding knowledge lack limited
need needed needs further future remains remain unclear unknown little known gap gaps however although therefore
background methods results conclusions copyright elsevier rights reserved springer wiley ltd inc published
journal abstract introduction discussion purpose setting participants highlight highlights highlighted
growing concern concerns frequently frequent characterised characterized collected screened detected identified
conflicting contradictory inconsistent controversial debated discrepancy discrepancies mixed equivocal divergent
unexplored poorly understood elusive warranted warrant warrants required necessary investigated explored
widely largely often usually generally typically particularly especially additionally furthermore moreover
notably importantly interestingly finally overall still already previously currently novel great greater
good better best large larger small smaller several many various numerous whole entire due able
silico vitro vivo systematic scoping narrative contribution contributions
""".split())

# Allowed as the LAST word of a phrase ("antibiotic use", "risk factors"), never alone or first.
END_OK = set("""
use usage data design level levels rate rates factor factors type types analysis analyses effect effects
model models method methods approach group groups time number value values role evaluation assessment
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_NO_PLURAL_FOLD = {
    "species", "series", "analysis", "diagnosis", "virus", "status", "bias", "sepsis", "mellitus",
    "nucleus", "focus", "corpus", "genus", "stimulus", "basis", "crisis", "thesis", "hypothesis",
    "synthesis", "process", "class", "mass", "stress", "loss", "access", "success", "gas", "bus",
    "news", "physics", "economics", "genomics", "proteomics", "metabolomics", "metagenomics",
    "transcriptomics", "informatics", "bioinformatics", "immunoinformatics", "statistics", "ethics",
    "mathematics", "dynamics", "kinetics", "genetics", "epigenetics", "diabetes", "herpes", "rabies",
    "scabies", "aids", "covid", "sars", "mers", "this", "its", "is", "was", "has", "gas", "lens",
    "osis", "itis", "us", "plus", "less", "cross", "glass", "grass", "chaos", "canvas", "atlas",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def fold(token: str) -> str:
    """Very light plural folding used only for internal matching."""
    if token in _NO_PLURAL_FOLD or len(token) <= 3 or "-" in token[-3:]:
        return token
    if token.endswith(("osis", "itis", "ss", "us", "is", "ics")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("'s"):
        return token[:-1]
    return token


def is_content(token: str) -> bool:
    if token in STOPWORDS:
        return False
    if token.isdigit():
        return False
    if len(token) < 2:
        return False
    if re.fullmatch(r"[\d\-']+", token):
        return False
    return True


def content_runs(tokens: list[str]) -> list[list[str]]:
    """Split a token list into runs of consecutive content tokens."""
    runs: list[list[str]] = []
    cur: list[str] = []
    for t in tokens:
        if is_content(t):
            cur.append(t)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


_ABBREV_RE = re.compile(r"\b(?:e\.g|i\.e|et al|vs|cf|approx|ca)\.", re.I)
_GENUS_RE = re.compile(r"\b([A-Za-z])\.\s?([a-z]{3,})\b")  # "E. coli" -> "E-coli"
_CLAUSE_RE = re.compile(r"[.;:!?()\[\]{},\u2014\u2013/]+|\s-\s")


def clauses(text: str) -> list[str]:
    text = _ABBREV_RE.sub(" ", text or "")
    text = _GENUS_RE.sub(r"\1-\2", text)
    return [c for c in _CLAUSE_RE.split(text) if c.strip()]


def candidate_ngrams(text: str, max_n: int = 3) -> list[tuple[str, str]]:
    """Return (key, surface) pairs for 1..max_n-grams inside content runs.

    key: folded tokens joined by space (used for matching)
    surface: original lower-case tokens joined by space (used for display)
    Phrases never cross punctuation. Unigrams and first words may not be
    boilerplate; last words may be boilerplate only if listed in END_OK.
    """
    out: list[tuple[str, str]] = []
    for clause in clauses(text):
        for run in content_runs(tokenize(clause)):
            folded = [fold(t) for t in run]
            n_run = len(run)
            for i in range(n_run):
                first = run[i]
                if first in BOILERPLATE:
                    continue
                for n in range(1, max_n + 1):
                    j = i + n
                    if j > n_run:
                        break
                    last = run[j - 1]
                    if n == 1:
                        if len(first) < 3 and not first.isalpha():
                            continue
                    elif last in BOILERPLATE and last not in END_OK:
                        continue
                    out.append((" ".join(folded[i:j]), " ".join(run[i:j])))
    return out


def normalize_phrase(phrase: str) -> str:
    """Normalize a metadata keyword / user term to the same key space as n-grams."""
    toks = [fold(t) for t in tokenize(phrase) if is_content(t)]
    return " ".join(toks)


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def compile_group_regex(terms: list[str]) -> re.Pattern | None:
    """Word-boundary regex for a group of synonyms (case-insensitive)."""
    parts = []
    for t in terms:
        t = t.strip().lower()
        if not t:
            continue
        esc = re.escape(t).replace(r"\ ", r"[\s\-]+")
        parts.append(esc)
    if not parts:
        return None
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])", re.I)


def shorten(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"
