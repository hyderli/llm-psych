"""Pre-registration guards for the wheel32 complement-pair stimulus design.

These enforce amendments A1-A4 in ``plans/c2-revalidation-wheel32.md``. They are
model-free and cheap, and they exist so the constraints are checked while the
set is being authored rather than audited after it is built.

The wheel32 set does not exist yet; every test here skips until it does. That is
deliberate — the guards are committed before the stimuli so the record shows the
constraints predate the authoring.

Row schema required beyond the June/July fields (id, text, emotion, family, x,
intensity_rank, direction):

* ``scenario``    -- the world the sentence describes ("shelters", "checkpoint").
                     Both arms of a pair share it. Disjointness across the
                     sweep/confirmation split is checked at this level, not at
                     family level.
* ``pair``        -- pair identifier; the two arms share it, neutrals have None.
* ``denominator`` -- the integer the count is drawn from (twenty throughout).
* ``split``       -- "sweep" or "confirmation".

Run:  uv run pytest tests/test_complement_pairs.py -q
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[1]
WHEEL32_JSONL = _repo_root / "data" / "public" / "intensity_wheel32.jsonl"

N_PER_ARM = 12      # decreasing 1-12, increasing 8-19
N_PER_NEUTRAL = 19  # the union of both arms' numerals

# Lexical negators only. The rule is about NEGATION, not valence: "violations"
# and "missing" are negative in meaning but are not negated predicates, and both
# are compliant.
NEGATORS = {"not", "no", "never", "neither", "nor", "without"}

# Morphological negatives are NOT detected by affix rule. A prefix heuristic
# ("un"/"in"/"im"/"non"/"dis" over some length) fires on "inspector",
# "information", "important", "increasing", "included" and most other innocent
# vocabulary, and suppressing those by length or allowlist produces a rule that
# no longer catches anything. This explicit list covers the forms this design is
# actually at risk of; anything outside it is an audit item, recorded as such in
# A1 rather than pretended to be mechanised.
NEG_MORPH = {
    "unprepared", "unanswered", "undecided", "unresolved", "unfilled",
    "uncovered", "unpaid", "unread", "unsigned", "incomplete", "unavailable",
    "unconfirmed", "unfinished", "unclaimed", "unmet",
}


@pytest.fixture(scope="module")
def rows():
    if not WHEEL32_JSONL.exists():
        pytest.skip("wheel32 stimulus set not built yet")
    return [json.loads(ln) for ln in WHEEL32_JSONL.read_text().splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def families(rows):
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        out[(r["emotion"], r["family"])].append(r)
    return out


# ---------------------------------------------------------------------------
# A1 -- pairing replaces the inverse-only invariant
# ---------------------------------------------------------------------------

def test_every_emotion_family_has_a_complement(rows):
    """Increasing families are admitted only as complements; the pair is the unit."""
    by_pair: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["emotion"] == "neutral":
            continue
        assert r.get("pair"), f"{r['family']}: emotion family has no pair id"
        by_pair[r["pair"]].add(r["direction"])
    assert by_pair, "no emotion pairs found"
    for pair, dirs in by_pair.items():
        assert dirs == {"decreasing", "increasing"}, (
            f"pair {pair}: expected one arm of each direction, got {sorted(dirs)}"
        )


def test_paired_arms_share_scenario_and_denominator(rows):
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pair"):
            by_pair[r["pair"]].append(r)
    for pair, rs in by_pair.items():
        assert len({r["scenario"] for r in rs}) == 1, f"pair {pair}: scenario differs across arms"
        assert len({r["denominator"] for r in rs}) == 1, f"pair {pair}: denominator differs across arms"
        assert len({r["family"] for r in rs}) == 2, f"pair {pair}: expected exactly two families"


def test_paired_arms_match_polarity(families):
    """A complement carried by a negator flips for reasons unrelated to emotion."""
    offenders = []
    for (emotion, family), rs in families.items():
        if emotion == "neutral":
            continue
        text = rs[0]["text"].lower()
        for w in re.findall(r"[a-z]+", text):
            if w in NEGATORS:
                offenders.append((family, w))
            elif w in NEG_MORPH:
                offenders.append((family, f"{w} (morphological negative)"))
    assert not offenders, (
        "negated or morphologically negative predicates in an arm; both halves "
        f"must be lexically positive: {offenders[:6]}"
    )


def test_polarity_guard_does_not_fire_on_valence():
    """Regression cover: the rule is about negation, not negative valence, and
    an affix heuristic flags ordinary vocabulary. Both must stay true."""
    ok = [
        "Of the twenty entries the inspector has ruled on, he recorded 3 of them as violations.",
        "Of the twenty documents the officer may ask for, she is missing 3 of them.",
    ]
    for text in ok:
        for w in re.findall(r"[a-z]+", text.lower()):
            assert w not in NEGATORS and w not in NEG_MORPH, f"false positive on {w!r}"
    bad = "She has 3 of the twenty questions unanswered."
    assert any(w in NEG_MORPH for w in re.findall(r"[a-z]+", bad.lower()))


# ---------------------------------------------------------------------------
# A2 -- grids closed under the complement map
# ---------------------------------------------------------------------------

def test_complementary_grids(rows):
    """Arm B's numerals are the denominator minus arm A's, so both sample the
    same underlying k. A shared grid would make the arms sample different
    regions of the curve and put attenuation asymmetry into the artifact half."""
    by_pair: dict[str, dict[str, set[int]]] = defaultdict(dict)
    denom: dict[str, int] = {}
    for r in rows:
        if not r.get("pair"):
            continue
        by_pair[r["pair"]].setdefault(r["direction"], set()).add(r["x"])
        denom[r["pair"]] = r["denominator"]
    for pair, arms in by_pair.items():
        dec, inc = arms["decreasing"], arms["increasing"]
        assert inc == {denom[pair] - x for x in dec}, (
            f"pair {pair}: grids not closed under the complement map; "
            f"decreasing={sorted(dec)} increasing={sorted(inc)}"
        )


def test_grid_sizes(families):
    for (emotion, family), rs in families.items():
        xs = {r["x"] for r in rs}
        want = N_PER_NEUTRAL if emotion == "neutral" else N_PER_ARM
        assert len(xs) == len(rs) == want, (
            f"{family}: {len(rs)} rows / {len(xs)} unique x, expected {want}"
        )


def test_no_zero_and_denominator_above_max(rows):
    """k=0 is a categorical state, not the bottom of a continuum; and a count
    equal to the denominator is total coverage, a regime change."""
    for r in rows:
        assert r["x"] >= 1, f"{r['family']}: x=0 excluded by A2"
        assert r["x"] < r["denominator"], (
            f"{r['family']}: x={r['x']} is not below denominator {r['denominator']}"
        )


# ---------------------------------------------------------------------------
# A4 -- the split is at scenario level and the pair is atomic
# ---------------------------------------------------------------------------

def test_pairs_do_not_straddle_the_split(rows):
    by_pair: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("pair"):
            by_pair[r["pair"]].add(r["split"])
    for pair, splits in by_pair.items():
        assert len(splits) == 1, (
            f"pair {pair} straddles the sweep/confirmation split: a layer locked "
            f"on one arm is locked on the other arm's data"
        )


def test_scenarios_disjoint_across_split(rows):
    sweep = {r["scenario"] for r in rows if r["split"] == "sweep"}
    confirm = {r["scenario"] for r in rows if r["split"] == "confirmation"}
    shared = sweep & confirm
    assert not shared, (
        f"scenarios appear on both sides of the split: {sorted(shared)}. "
        "Family-level disjointness is not sufficient — the July guards check "
        "family name, sentence and skeleton only."
    )


def test_every_scenario_has_a_neutral(rows):
    """The ratio diagnostic is per-scenario; a scenario without its own neutral
    cannot separate magnitude contamination from attenuation asymmetry."""
    emo = {r["scenario"] for r in rows if r["emotion"] != "neutral"}
    neu = {r["scenario"] for r in rows if r["emotion"] == "neutral"}
    missing = emo - neu
    assert not missing, f"scenarios with no matched neutral: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Carried over from the June/July constraints
# ---------------------------------------------------------------------------

def test_template_holds_structure_constant(families):
    for (emotion, family), rs in families.items():
        skeletons = {"".join(c for c in r["text"] if not c.isdigit()) for r in rs}
        assert len(skeletons) == 1, f"{family}: text varies beyond the number"


def test_count_never_governs_a_verb(families):
    """x in subject position breaks agreement at x=1 and produces two skeletons.
    Kept as an explicit check because it is the rule most easily dropped when
    the complement is written after the base family."""
    bad = []
    for (emotion, family), rs in families.items():
        tmpl = "".join("{x}" if c.isdigit() else c for c in rs[0]["text"])
        tmpl = re.sub(r"(\{x\})+", "{x}", tmpl)
        after = tmpl.split("{x}", 1)[1] if "{x}" in tmpl else ""
        first = re.findall(r"[a-z]+", after.lower())[:1]
        if first and first[0] in {"are", "were", "have", "has", "is", "was"}:
            bad.append((family, first[0]))
    assert not bad, f"count governs a verb (breaks at x=1): {bad}"
