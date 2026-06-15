"""Build the numerical-intensity template stimuli (C2; Sofroniew et al. 2026 §2).

Each template family is a fixed sentence with one ``{x}`` slot; only the
number varies, so token structure is held nearly constant. The number ->
emotional-intensity relation is **semantic**, and for the deliberately
INVERSE families (e.g. a sister's age at death, days until a friend
returns) intensity *decreases* as the raw number grows — the case a
surface/digit account gets backwards. The validator
(``scripts/validate_intensity_semantic.py``) projects the story-derived
vector onto each row's activation and correlates it with ``intensity_rank``
(the semantic order), not the raw ``x``; the inverse families are the
decisive surface-vs-semantics test. See ``plans/numerical-intensity-control.md``.

Deterministic + hand-authored: this script is the frozen source for
``data/public/intensity_templates.jsonl`` (MD5-lock the output in
``configs/stimuli_hashes.yaml``). Emotion-label words are never used.

Run:  uv run python scripts/build_intensity_templates.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
OUT = _repo_root / "data" / "public" / "intensity_templates.jsonl"

# family: (emotion, short_name, template, x-values, direction)
#   direction "increasing" -> intensity rises with x
#   direction "decreasing" -> intensity rises as x falls (INVERSE; the
#                             surface-divergent case)
#   direction "flat"       -> no emotional reading (neutral control)
FAMILIES: list[tuple[str, str, str, list[int], str]] = [
    # ---- admiration ----
    ("admiration", "record_margin",
     "She crossed the finish line {x} minutes ahead of the world record.",
     [0, 1, 2, 5, 10, 20], "increasing"),
    ("admiration", "wallet_returned",
     "He handed back the lost wallet with all ${x} still inside, though his own rent was weeks overdue.",
     [20, 100, 500, 2000, 8000, 40000], "increasing"),
    ("admiration", "self_taught_age",
     "She taught herself to read and write at the age of {x}.",
     [8, 25, 45, 60, 75, 90], "increasing"),
    ("admiration", "team_size",  # INVERSE: fewer hands -> greater feat
     "She rebuilt the entire bridge design overnight with a team of just {x}.",
     [1, 2, 4, 8, 16, 40], "decreasing"),

    # ---- joy ----
    ("joy", "exam_pass",
     "{x} of the 30 students in my class passed the final this morning.",
     [2, 8, 15, 22, 28, 30], "increasing"),
    ("joy", "tickets_sold",
     "We sold {x} of the 500 tickets for the benefit we put together.",
     [30, 120, 250, 380, 470, 500], "increasing"),
    ("joy", "scans_clear",
     "Every one of the {x} scans came back clear this afternoon.",
     [1, 2, 3, 5, 8, 12], "increasing"),
    ("joy", "days_until_return",  # INVERSE: sooner -> greater anticipation
     "My closest friend moves back to town in {x} days.",
     [1, 3, 7, 14, 30, 90], "decreasing"),

    # ---- loathing ----
    ("loathing", "lies_oath",
     "He was caught lying {x} separate times under oath.",
     [1, 2, 5, 10, 25, 60], "increasing"),
    ("loathing", "rent_hikes",
     "The landlord raised the rent for the {x}th time this year while ignoring every repair request.",
     [1, 2, 3, 5, 8, 12], "increasing"),
    ("loathing", "days_chained",
     "He left the dog chained in the yard with no water for {x} days during the heatwave.",
     [1, 2, 3, 5, 8, 14], "increasing"),
    ("loathing", "relief_withheld",  # INVERSE: less passed on -> greater contempt
     "He kept the $50,000 disaster fund and passed just ${x} on to the families who lost everything.",
     [0, 50, 500, 5000, 25000, 50000], "decreasing"),

    # ---- sadness ----
    ("sadness", "age_at_death",  # INVERSE: younger -> greater loss
     "My sister was {x} years old when she passed away.",
     [4, 12, 25, 45, 70, 92], "decreasing"),
    ("sadness", "funeral_attendance",  # INVERSE: fewer came -> lonelier loss
     "Only {x} people came to my father's funeral.",
     [3, 8, 20, 50, 120, 300], "decreasing"),
    ("sadness", "dog_missing_days",
     "Our dog has been gone for {x} days now.",
     [1, 2, 4, 8, 15, 30], "increasing"),
    ("sadness", "years_no_visit",
     "It has been {x} years since any of my children last came by.",
     [0, 1, 2, 5, 10, 20], "increasing"),

    # ---- neutral (no-valence number; expected flat projection) ----
    ("neutral", "bus_minutes",
     "The next bus arrives in {x} minutes.",
     [2, 5, 10, 20, 40, 60], "flat"),
    ("neutral", "flour_cups",
     "The recipe calls for {x} cups of flour.",
     [1, 2, 3, 4, 6, 8], "flat"),
    ("neutral", "office_floor",
     "The office is on the {x}th floor.",
     [1, 3, 7, 12, 20, 30], "flat"),
]


def _ranks(xs: list[int], direction: str) -> list[int]:
    """Semantic-intensity rank per x: 0 = least intense ... n-1 = most.

    Increasing -> rank tracks x ascending; decreasing -> x descending; flat
    -> all 0 (no emotional ordering).
    """
    n = len(xs)
    if direction == "flat":
        return [0] * n
    order = sorted(range(n), key=lambda i: xs[i])  # indices, x ascending
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r if direction == "increasing" else (n - 1 - r)
    return rank


def build() -> list[dict]:
    rows: list[dict] = []
    for emotion, family, template, xs, direction in FAMILIES:
        if len(set(xs)) != len(xs) or len(xs) < 6:
            raise ValueError(f"{family}: need >=6 unique x values")
        if "{x}" not in template:
            raise ValueError(f"{family}: template missing {{x}} slot")
        ranks = _ranks(xs, direction)
        for i, (x, rank) in enumerate(zip(xs, ranks)):
            rows.append({
                "id": f"{emotion}_{family}_{i:02d}",
                "text": template.format(x=x),
                "emotion": emotion,
                "family": family,
                "x": x,
                "intensity_rank": rank,
                "direction": direction,
            })
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    n_emotion = sum(r["emotion"] != "neutral" for r in rows)
    n_inverse = sum(r["direction"] == "decreasing" for r in rows)
    fams = sorted({(r["emotion"], r["family"]) for r in rows})
    print(f"wrote {len(rows)} rows ({n_emotion} emotion, {len(rows) - n_emotion} neutral) "
          f"across {len(fams)} families to {OUT.relative_to(_repo_root)}")
    print(f"inverse (decreasing) rows: {n_inverse}")
    print(f"MD5: {md5}  (record in configs/stimuli_hashes.yaml to freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
