"""Regular expressions for gap statements, contradictions and study designs."""
from __future__ import annotations

import re

GAP_RX = re.compile(
    r"""\b(?:
      remains?\s+(?:largely\s+|poorly\s+|still\s+|currently\s+|completely\s+)?
          (?:unknown|unclear|unexplored|uncharacteri[sz]ed|elusive|unresolved|undetermined|
             to\s+be\s+(?:determined|elucidated|established|investigated|explored|clarified))
    | (?:is|are)\s+(?:still\s+)?(?:poorly|not\s+well|incompletely|not\s+fully|insufficiently|rarely)\s+
          (?:understood|characteri[sz]ed|defined|studied|known|documented|explored|investigated)
    | little\s+is\s+known | less\s+is\s+known | not\s+much\s+is\s+known | scarcely\s+(?:studied|known)
    | (?:limited|scarce|sparse|insufficient)\s+(?:data|evidence|information|research|studies|knowledge)
    | lack(?:s|ing)?\s+of\s+(?:data|evidence|studies|research|information|knowledge)
    | paucity\s+of | dearth\s+of
    | (?:no|few)\s+(?:previous\s+|prior\s+|published\s+)?studies\s+(?:have|has)
    | (?:has|have)\s+(?:not\s+(?:yet\s+)?|never\s+)been\s+
          (?:investigated|explored|studied|examined|reported|addressed|evaluated|assessed|
             characteri[sz]ed|described|elucidated)
    | (?:further|future|more|additional)\s+(?:research|studies|work|investigations?|trials?)\s+
          (?:is\s+|are\s+|will\s+be\s+)?(?:needed|required|warranted|necessary)
    | (?:knowledge|research)\s+gaps? | gaps?\s+in\s+(?:the\s+)?(?:knowledge|literature|understanding|evidence)
    | under-?(?:studied|researched|explored|investigated|reported)
    | yet\s+to\s+be\s+(?:determined|elucidated|established|investigated|explored|clarified|understood)
    | warrants?\s+further\s+(?:investigation|study|research)
    | open\s+questions? | unanswered\s+questions?
    )\b""",
    re.I | re.X,
)

CONTRA_RX = re.compile(
    r"""\b(?:
      (?:conflicting|contradictory|inconsistent|mixed|equivocal|divergent)\s+
          (?:results|findings|evidence|data|reports|outcomes)
    | (?:remains?|is|are)\s+(?:controversial|debated|contested)
    | controversy | discrepanc(?:y|ies)
    )\b""",
    re.I | re.X,
)

WEAK_PRECLINICAL = re.compile(
    r"\b(?:in\s+silico|in\s+vitro|molecular\s+docking|docking\s+(?:study|analysis|simulations?)|"
    r"molecular\s+dynamics|immunoinformatic\w*|reverse\s+vaccinology|"
    r"computational(?:ly)?\s+(?:predict\w*|analys\w*|model\w*|screen\w*)|"
    r"bioinformatic\w*\s+(?:analys\w*|predict\w*|approach\w*)|cell\s+lines?|cell\s+culture)\b",
    re.I,
)
STRONG_INVIVO = re.compile(
    r"\b(?:in\s+vivo|animal\s+models?|mice|murine|mouse\s+model|rats|zebrafish|"
    r"non-?human\s+primates?|challenge\s+(?:study|experiment))\b",
    re.I,
)
STRONG_CLINICAL = re.compile(
    r"\b(?:clinical\s+trials?|randomi[sz]ed|prospective|longitudinal|cohort\s+study|"
    r"follow-?up\s+study|multi-?cent(?:er|re)|field\s+trials?|case-control)\b",
    re.I,
)
WEAK_OBSERVATIONAL = re.compile(
    r"\b(?:cross-?sectional|retrospective|case\s+reports?|case\s+series|single-?cent(?:er|re)|"
    r"pilot\s+study|small\s+sample|convenience\s+sampl\w*|questionnaire-based|survey-based)\b",
    re.I,
)
REVIEW_TITLE = re.compile(
    r"\b(?:systematic\s+review|meta-?analys[ie]s|scoping\s+review|umbrella\s+review|"
    r"narrative\s+review|literature\s+review|a\s+review|review\s+of|an\s+overview|state\s+of\s+the\s+art)\b",
    re.I,
)

# Methods that are checked automatically even when the user lists none.
AUTO_METHODS: list[tuple[str, list[str]]] = [
    ("machine learning", ["machine learning", "deep learning", "neural network", "random forest"]),
    ("randomised controlled trial", ["randomized controlled trial", "randomised controlled trial", "RCT"]),
    ("qualitative methods", ["qualitative", "semi-structured interview", "focus group"]),
    ("whole-genome sequencing", ["whole genome sequencing", "whole-genome sequencing", "WGS"]),
    ("metagenomics", ["metagenomic", "metagenomics", "shotgun sequencing", "16S rRNA"]),
    ("single-cell sequencing", ["single-cell", "single cell RNA", "scRNA-seq"]),
    ("CRISPR", ["CRISPR"]),
    ("molecular docking / dynamics", ["molecular docking", "molecular dynamics"]),
    ("mathematical modelling", ["mathematical model", "agent-based model", "compartmental model"]),
    ("economic evaluation", ["cost-effectiveness", "economic evaluation"]),
    ("genome-wide association", ["GWAS", "genome-wide association"]),
    ("Mendelian randomisation", ["mendelian randomization", "mendelian randomisation"]),
    ("proteomics / metabolomics", ["proteomics", "proteomic", "metabolomics", "metabolomic"]),
    ("transcriptomics", ["RNA-seq", "transcriptomic", "transcriptome"]),
    ("geospatial analysis", ["geospatial", "spatial analysis", "remote sensing", "GIS"]),
    ("large language models", ["large language model", "LLM", "LLMs"]),
]
