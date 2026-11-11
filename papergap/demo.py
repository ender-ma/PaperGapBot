"""Offline demo: a synthetic corpus with planted gaps.

    python -m papergap.demo            # writes reports/demo_report.html

The corpus is fake (no network needed). It exists to test the engine and to
show what a report looks like.
"""
from __future__ import annotations

import random

from .brief import Brief
from .sources.base import Paper

TOPICS = [
    "broiler farms", "retail chicken meat", "escherichia coli isolates", "antimicrobial susceptibility testing",
    "minimum inhibitory concentration", "integron", "farm workers", "slaughterhouse", "environmental samples",
    "one health surveillance", "carbapenem resistance", "horizontal gene transfer", "veterinary prescriptions",
    "phylogenetic analysis", "salmonella enterica", "klebsiella pneumoniae", "disk diffusion", "pcr screening",
    "biosecurity practices", "antibiotic use in feed", "egg production", "colistin ban", "layer hens",
    "manure", "drinking water", "live bird markets",
]
BRIDGE_A, BRIDGE_C = "wastewater", "fitness cost"
BRIDGE_B = ["plasmid stability", "conjugation assay", "incx4 plasmid"]
A_SPEC = ["hospital effluent", "sewage treatment", "river sediment"]
C_SPEC = ["growth curve", "competition assay", "biofilm formation"]

TEMPLATES = [
    "We investigated {a} and {b} in {c}.",
    "Samples were analysed for {a}, with attention to {b}.",
    "Results showed that {a} was associated with {b} and {c}.",
    "{A} was common, and {b} was frequently observed.",
    "These findings highlight {a} in relation to {b}.",
]


def demo_brief() -> Brief:
    return Brief(
        field="Microbiology",
        niche="Antimicrobial resistance in food animals",
        focus="plasmid-mediated colistin resistance (mcr genes) in poultry production",
        key_terms="colistin|polymyxin, poultry|chicken|broiler",
        context="Nigeria|Nigerian, Kenya",
        methods="whole genome sequencing|WGS, metagenomics|metagenomic",
        exclude="",
        year_from=2008,
        year_to=2026,
        sources=["openalex"],
        max_papers=5000,
    )


def _sentence(rng: random.Random, phrases: list[str]) -> str:
    t = rng.choice(TEMPLATES)
    a, b, c = (phrases * 3)[:3]
    return t.format(a=a, b=b, c=c, A=a[:1].upper() + a[1:])


def synthetic_corpus(n: int = 3000, seed: int = 7) -> list[Paper]:
    rng = random.Random(seed)
    papers: list[Paper] = []
    years = list(range(2008, 2027))
    weights = [1 + 0.25 * i for i in range(len(years))]  # the field grows
    for i in range(n):
        year = rng.choices(years, weights)[0]
        phrases: list[str] = rng.sample(TOPICS, 4)
        extra_sent: list[str] = []
        cites = int(rng.expovariate(1 / 15))
        is_review = rng.random() < 0.04

        # BRIDGE: A and C share neighbours B but never co-occur
        r = rng.random()
        if r < 0.08:
            phrases = [BRIDGE_A] + rng.sample(BRIDGE_B, 1) + rng.sample(A_SPEC, 2) + phrases[:1]
        elif r < 0.16:
            phrases = [BRIDGE_C] + rng.sample(BRIDGE_B, 1) + rng.sample(C_SPEC, 2) + phrases[:1]
        else:
            phrases = [p for p in phrases if p not in (BRIDGE_A, BRIDGE_C)]

        # EMERGING: phage therapy only in recent years
        if year >= 2023 and rng.random() < 0.10:
            phrases.append("phage therapy")
        # DORMANT: growth promoters early, highly cited, then quiet
        if year <= 2014 and rng.random() < 0.35:
            phrases.append("growth promoters")
            cites += 120
        # VALIDATION gap: in silico inhibitor work, never in vivo
        insilico = rng.random() < 0.03
        if insilico:
            phrases.append("mcr-1 inhibitors")
            extra_sent.append("Candidate compounds were screened by molecular docking in silico.")

        # CONTEXT: Nigeria mentioned in ~7% of papers, never with biosecurity practices
        if rng.random() < 0.07:
            phrases = [p for p in phrases if p != "biosecurity practices"]
            extra_sent.append("Samples were collected from farms in Nigeria.")
        # METHOD: WGS in ~20% of papers, never on slaughterhouse samples
        if rng.random() < 0.2 and "slaughterhouse" not in phrases:
            extra_sent.append("Isolates were characterised by whole genome sequencing.")

        # Reviews never cover antibiotic use in feed
        if is_review and "antibiotic use in feed" in phrases:
            phrases.remove("antibiotic use in feed")

        # EXPLICIT gap statements
        if "egg production" in phrases and rng.random() < 0.25:
            extra_sent.append("The contribution of egg production to mcr-1 dissemination remains unclear.")
        if "colistin ban" in phrases and rng.random() < 0.3:
            extra_sent.append("Studies of the colistin ban report conflicting results.")

        if not insilico and rng.random() < 0.15:
            extra_sent.append("A cross-sectional design was used.")

        body = [_sentence(rng, phrases[k:k + 3]) for k in range(0, len(phrases), 2)]
        abstract = (
            "Colistin resistance in poultry is a growing concern. " + " ".join(body + extra_sent)
        )
        title_bits = phrases[:2]
        title = (
            f"A systematic review of {title_bits[0]} and colistin resistance in poultry"
            if is_review else f"{title_bits[0].capitalize()} and {title_bits[1]} in colistin-resistant poultry isolates"
        )
        papers.append(Paper(
            pid=f"demo:{i}", source="openalex", title=title, abstract=abstract, year=year,
            cited_by=cites, doi=None, url=f"https://example.org/demo/{i}",
            keywords=[], ptype="review" if is_review else "article",
        ))
    return papers


def main() -> None:  # pragma: no cover - convenience entry point
    import asyncio
    import os

    from .config import Settings
    from .pipeline import offline_report

    settings = Settings.from_env()
    path = asyncio.run(offline_report(synthetic_corpus(), demo_brief(), settings, os.path.join(settings.reports_dir, "demo_report.html")))
    print(f"Demo report written to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
