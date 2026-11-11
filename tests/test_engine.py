from papergap.demo import demo_brief, synthetic_corpus
from papergap.mining.cues import CONTRA_RX, GAP_RX
from papergap.mining.engine import mine
from papergap.mining.text import candidate_ngrams

DOCS = synthetic_corpus(3000)
RESULT = mine(lambda: iter(DOCS), len(DOCS), demo_brief())
BY_KIND = {}
for c in RESULT.candidates:
    BY_KIND.setdefault(c.kind, []).append(c)


def titles(kind):
    return " | ".join(c.title for c in BY_KIND.get(kind, []))


def test_cue_regexes():
    assert GAP_RX.search("The role of X remains largely unknown.")
    assert GAP_RX.search("Further studies are needed to confirm this.")
    assert GAP_RX.search("Little is known about Y in Africa.")
    assert GAP_RX.search("This has not yet been investigated.")
    assert not GAP_RX.search("We measured X in 200 samples.")
    assert CONTRA_RX.search("Previous studies report conflicting results.")


def test_ngrams_respect_punctuation_and_boilerplate():
    keys = [k for k, _ in candidate_ngrams("Resistance is a growing concern. Samples of E. coli; antibiotic use")]
    assert "concern sample" not in keys and "growing concern" not in keys
    assert "e-coli" in keys and "antibiotic use" in keys


def test_planted_bridge_found():
    assert "wastewater" in titles("BRIDGE") and "fitness cost" in titles("BRIDGE")


def test_planted_context_gaps_found():
    t = titles("CONTEXT")
    assert "Kenya" in t                      # niche never studied there
    assert "biosecurity practices" in t      # topic never studied in Nigeria


def test_planted_method_gaps_found():
    t = titles("METHOD")
    assert "metagenomics" in t
    assert "slaughterhouse" in t


def test_statements_trends_evidence():
    assert "colistin ban" in titles("CONTRADICTION")
    assert BY_KIND.get("EXPLICIT")
    assert "phage therapy" in titles("EMERGING")
    assert "growth promoters" in titles("DORMANT")
    ev = titles("EVIDENCE")
    assert "antibiotic use" in ev          # no review
    assert "in silico" in ev               # validation gap


def test_ranking_diverse_and_labelled():
    gids = [c.gid for c in RESULT.candidates]
    assert gids == [f"G{i}" for i in range(1, len(gids) + 1)]
    kinds = [c.kind for c in RESULT.candidates]
    assert max(kinds.count(k) for k in set(kinds)) <= 5
    # every signal carries example papers, except "never studied here" signals (nothing to cite)
    assert all(c.evidence for c in RESULT.candidates if c.stats.get("papers_mentioning", 1) > 0)
