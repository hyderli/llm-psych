"""Build the HELD-OUT confirmation numerical-intensity stimuli (C2 confirmatory test).

This is the *fresh* stimulus set required by the 2026-07-12 per-emotion
layer-selection amendment. The 2026-06-14 set
(``data/public/intensity_templates.jsonl``) was run and inspected before the
selection rule was written, so under that amendment its numbers are
**retrospective/descriptive only**; confirmatory status attaches exclusively
to this set, evaluated at the locked layers only.

Design (identical in kind to the June builder, see
``scripts/build_intensity_templates.py``):

* Each family is a fixed sentence with one ``{x}`` slot, so token structure is
  held nearly constant and only the number varies.
* Every emotion family here is **INVERSE** (``direction="decreasing"``):
  semantic intensity rises as the raw number *falls*. This is the decisive
  surface-vs-semantics test — a vector that merely encodes digit magnitude
  gets these backwards. Increasing families are deliberately omitted: for them
  rho(rank) == rho(x) identically, so they cannot separate meaning from digit.
* ``neutral`` families are flat controls (no emotional reading) and quantify
  the residual number-magnitude confound, which the June run showed to be
  pervasive (neutral |rho| 0.5-0.8 at every layer).
* Emotion-label words are never used.

Differences from the June set, both disclosed in the 2026-07-28 amendment:

1. **n = 12 x-values per family** (June: 6). The Spearman estimate over n=6 is
   the weakest link in the whole C2 suite — one row out of order moves it by
   ~0.14 — and ``plans/emotion-set-expansion-design.md`` already flags this.
   Authoring cost is unchanged; only the x-list is longer.
2. **Coverage: joy, loathing, sadness (+ neutral).** Admiration is excluded:
   it is a vector-quality failure at every swept layer on all three primaries
   under the amendment's clause 2, so it has no locked layer to confirm at.
   It re-enters only via the residualization experiment
   (``plans/residualization-admiration.md``), which must clear this same bar.

Provenance: family templates were LLM-drafted (Claude, 2026-07-28) against the
frozen constraints and then human-audited by the PI before freezing. This
differs from the amendment's "hand-authored" wording and is disclosed in the
2026-07-28 amendment block. No family concept, template, or x-list is reused
from the June set (enforced by ``tests/test_confirmation_stimuli.py``).

Deterministic: this script is the frozen source for
``data/public/intensity_confirmation.jsonl``. MD5-lock the output in
``configs/stimuli_hashes.yaml`` BEFORE any model run.

Run:  uv run python scripts/build_confirmation_intensity.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
OUT = _repo_root / "data" / "public" / "intensity_confirmation.jsonl"

N_PER_FAMILY = 12

# family: (emotion, short_name, template, x-values, direction)
#   "decreasing" -> intensity rises as x falls (INVERSE; surface-divergent)
#   "flat"       -> no emotional reading (neutral control)
FAMILIES: list[tuple[str, str, str, list[int], str]] = [
    # ---- joy (all INVERSE: fewer remaining -> nearer the awaited good) ----
    ("joy", "stops_away",
     "The train she is on is {x} stops from the platform where I am standing.",
     [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 31], "decreasing"),
    ("joy", "signatures_left",
     "The adoption is final once {x} more signatures are collected.",
     [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 23], "decreasing"),
    ("joy", "pages_left",
     "I have {x} pages left before the book I have worked on for six years is finished.",
     [2, 3, 5, 8, 12, 18, 26, 41, 63, 94, 141, 212], "decreasing"),

    # ---- loathing (all INVERSE: less of the owed good passed on -> worse) ----
    ("loathing", "beds_kept_open",
     "Through the January cold snap he kept {x} of the shelter's 200 beds open.",
     [0, 1, 2, 4, 7, 13, 24, 39, 61, 92, 138, 200], "decreasing"),
    ("loathing", "workers_rehired",
     "After the buyout cleared, he rehired {x} of the 300 people he had let go.",
     [0, 1, 3, 6, 11, 19, 34, 54, 86, 131, 197, 300], "decreasing"),
    ("loathing", "hours_notice",
     "He gave the families {x} hours of notice before the building was emptied.",
     [0, 2, 3, 4, 5, 7, 11, 17, 23, 35, 47, 71], "decreasing"),

    # ---- sadness (all INVERSE: less of what remains -> greater loss) ----
    ("sadness", "letters_survived",
     "Of the hundreds of letters my mother wrote, {x} survived the fire.",
     [0, 1, 2, 3, 5, 9, 14, 21, 33, 49, 74, 111], "decreasing"),
    ("sadness", "minutes_allowed",
     "They gave us {x} minutes with him before the door was closed.",
     [2, 3, 5, 7, 11, 17, 24, 36, 51, 73, 97, 131], "decreasing"),
    ("sadness", "names_recalled",
     "By the end, my grandmother could still name {x} of her twelve grandchildren.",
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], "decreasing"),

    # ---- neutral (flat controls; no emotional reading) ----
    ("neutral", "shelf_books",
     "There are {x} books on the shelf by the window.",
     [2, 3, 5, 8, 12, 18, 26, 41, 63, 94, 141, 212], "flat"),
    ("neutral", "parking_level",
     "The car is parked on level {x} of the garage.",
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "flat"),
    ("neutral", "stamps_needed",
     "The envelope needs {x} stamps to post.",
     [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20], "flat"),
]

# Words that would leak the label into the stimulus. Checked case-insensitively
# as substrings, so "sadness"/"sadly" are both caught by "sad".
BANNED_SUBSTRINGS = (
    "admir", "joy", "joyful", "loath", "sad", "happy", "happi", "grief",
    "griev", "angry", "anger", "fear", "afraid", "disgust", "contempt",
    "delight", "sorrow", "mourn", "proud", "pride", "hate", "hatred",
    "love", "loving", "miser", "despair", "elat", "cheer", "glad",
    "distress", "anguish", "revuls", "aversion", "emotion", "feel",
)


# A plural noun within two words after the {x} slot ("{x} stops", "{x} more
# signatures") makes x=1 ungrammatical. The June set did not guard this and
# contains rows like "1 minutes ahead of the world record"; on an INVERSE
# family the offending row sits at the most-intense end, precisely where a
# grammaticality artifact would be read as semantic signal.
_PLURAL_AFTER_X = re.compile(r"\{x\}\s+(?:(\w+)\s+)?(\w+)\b")

# Words ending in "s" that are not plural nouns, so must not trip the check.
_NOT_PLURAL = frozenset({
    "its", "his", "hers", "theirs", "this", "was", "is", "has", "as", "us",
    "yes", "thus", "less", "unless", "always", "perhaps", "across", "plus",
    "versus", "gas", "bus", "class", "glass", "pass", "press", "cross",
    "loss", "miss", "boss", "news", "series", "species",
})


def _plural_after_x(template: str) -> str | None:
    """Return the plural noun within two words after ``{x}``, if any.

    ``"{x} stops from"`` -> ``"stops"``; ``"{x} of its 200 beds"`` -> ``None``
    ("its" is possessive, and "beds" is more than two words out).
    """
    m = _PLURAL_AFTER_X.search(template)
    if not m:
        return None
    for word in m.groups():
        if not word:
            continue
        w = word.lower()
        if w.endswith("s") and w not in _NOT_PLURAL and not w.endswith(("ss", "us", "is")):
            return word
    return None


def _ranks(xs: list[int], direction: str) -> list[int]:
    """Semantic-intensity rank per x: 0 = least intense ... n-1 = most.

    Mirrors ``scripts/build_intensity_templates.py::_ranks`` exactly. It is
    duplicated rather than imported on purpose: the June builder is the frozen
    source of an already-locked stimulus file and must not acquire a new
    importer that could later motivate editing it.
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
    seen: set[tuple[str, str]] = set()
    for emotion, family, template, xs, direction in FAMILIES:
        if (emotion, family) in seen:
            raise ValueError(f"{emotion}/{family}: duplicate family")
        seen.add((emotion, family))
        if len(set(xs)) != len(xs) or len(xs) != N_PER_FAMILY:
            raise ValueError(f"{family}: need exactly {N_PER_FAMILY} unique x values")
        if "{x}" not in template:
            raise ValueError(f"{family}: template missing {{x}} slot")
        if emotion != "neutral" and direction != "decreasing":
            raise ValueError(
                f"{family}: confirmation emotion families must be INVERSE "
                f"(decreasing), got {direction!r}"
            )
        low = template.lower()
        for bad in BANNED_SUBSTRINGS:
            if bad in low:
                raise ValueError(f"{family}: emotion-label word {bad!r} in template")
        if 1 in xs and (plural := _plural_after_x(template)):
            raise ValueError(
                f"{family}: x=1 is ungrammatical before plural {plural!r} "
                f"— drop 1 from the x-list or reword the template"
            )
        ranks = _ranks(xs, direction)
        for i, (x, rank) in enumerate(zip(xs, ranks, strict=True)):
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
    fams = sorted({(r["emotion"], r["family"]) for r in rows})
    n_inverse = sum(r["direction"] == "decreasing" for r in rows)
    per_emotion: dict[str, int] = {}
    for e, _ in fams:
        per_emotion[e] = per_emotion.get(e, 0) + 1
    print(f"wrote {len(rows)} rows across {len(fams)} families to "
          f"{OUT.relative_to(_repo_root)}")
    print("families per emotion: "
          + ", ".join(f"{e} {n}" for e, n in sorted(per_emotion.items())))
    print(f"inverse (decreasing) rows: {n_inverse}")
    print(f"MD5: {md5}  (record in configs/stimuli_hashes.yaml to freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
