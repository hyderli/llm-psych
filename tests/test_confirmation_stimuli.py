"""Pre-registration guards for the held-out C2 confirmation stimuli.

These tests enforce the constraints the 2026-07-12 amendment places on the
confirmation set, plus the two disclosed deviations (n=12, coverage). They are
cheap and model-free, and are the reason the set can be frozen with confidence
before any GPU time is spent.

Run:  uv run pytest tests/test_confirmation_stimuli.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_repo_root = Path(__file__).resolve().parents[1]
_SCRIPTS = _repo_root / "scripts"
CONFIRM_JSONL = _repo_root / "data" / "public" / "intensity_confirmation.jsonl"
JUNE_JSONL = _repo_root / "data" / "public" / "intensity_templates.jsonl"

EXPECTED_EMOTIONS = {"joy", "loathing", "sadness"}
MIN_FAMILIES_PER_EMOTION = 3
N_PER_FAMILY = 12


def _load_script(name: str):
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_script("build_confirmation_intensity")


@pytest.fixture(scope="module")
def rows(builder):
    return builder.build()


@pytest.fixture(scope="module")
def june_rows():
    if not JUNE_JSONL.exists():
        pytest.skip("June stimulus set not present")
    return [json.loads(ln) for ln in JUNE_JSONL.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Constraints carried over from the 2026-06-14 set
# ---------------------------------------------------------------------------

def test_no_emotion_label_words(rows, builder):
    """Emotion-label words are never used (frozen constraint)."""
    offenders = [
        (r["id"], bad)
        for r in rows
        for bad in builder.BANNED_SUBSTRINGS
        if bad in r["text"].lower()
    ]
    assert not offenders, f"emotion-label leakage: {offenders[:5]}"


def test_all_emotion_families_are_inverse(rows):
    """Only inverse families are diagnostic; increasing ones have rho(rank)==rho(x)."""
    bad = {(r["emotion"], r["family"], r["direction"])
           for r in rows
           if r["emotion"] != "neutral" and r["direction"] != "decreasing"}
    assert not bad, f"non-inverse emotion families: {bad}"


def test_ranks_invert_the_digit(rows):
    """For every inverse family, semantic rank must fall as x rises."""
    fams: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r in rows:
        if r["direction"] != "decreasing":
            continue
        fams.setdefault((r["emotion"], r["family"]), []).append(
            (r["x"], r["intensity_rank"])
        )
    assert fams
    for key, pairs in fams.items():
        pairs.sort()  # by x ascending
        ranks = [rank for _, rank in pairs]
        assert ranks == sorted(ranks, reverse=True), f"{key}: rank does not invert x"
        assert set(ranks) == set(range(len(ranks))), f"{key}: ranks not a permutation"


def test_neutral_controls_are_flat(rows):
    neutral = [r for r in rows if r["emotion"] == "neutral"]
    assert neutral, "neutral control families are required (number-confound check)"
    assert all(r["direction"] == "flat" for r in neutral)
    assert all(r["intensity_rank"] == 0 for r in neutral)


def test_no_ungrammatical_singular_rows(rows, builder):
    """x=1 before a plural noun ("1 stops") is an artifact at the intense end."""
    offenders = [
        (r["emotion"], r["family"], r["text"])
        for r in rows
        if r["x"] == 1 and builder._plural_after_x(r["text"].replace(" 1 ", " {x} "))
    ]
    assert not offenders, f"ungrammatical singular rows: {offenders}"


def test_plural_guard_detects_a_known_bad_template(builder):
    """The guard itself must work — regression cover for the 'its' false positive."""
    assert builder._plural_after_x("She waited {x} stops from home.") == "stops"
    assert builder._plural_after_x("Once {x} more signatures are in.") == "signatures"
    assert builder._plural_after_x("He kept {x} of its 200 beds open.") is None
    assert builder._plural_after_x("The car is on level {x} of the garage.") is None


def test_template_holds_structure_constant(rows):
    """Within a family only the number may vary."""
    fams: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        fams.setdefault((r["emotion"], r["family"]), []).append(r["text"])
    for key, texts in fams.items():
        skeletons = {"".join(ch for ch in t if not ch.isdigit()) for t in texts}
        assert len(skeletons) == 1, f"{key}: text varies beyond the number"


# ---------------------------------------------------------------------------
# The two disclosed deviations (2026-07-28 amendment)
# ---------------------------------------------------------------------------

def test_twelve_unique_x_per_family(rows):
    fams: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        fams.setdefault((r["emotion"], r["family"]), []).append(r["x"])
    for key, xs in fams.items():
        assert len(xs) == N_PER_FAMILY, f"{key}: {len(xs)} rows, expected {N_PER_FAMILY}"
        assert len(set(xs)) == N_PER_FAMILY, f"{key}: duplicate x values"


def test_coverage_matches_amendment(rows):
    """joy/loathing/sadness confirmed; admiration excluded as a VQ failure."""
    emotions = {r["emotion"] for r in rows} - {"neutral"}
    assert emotions == EXPECTED_EMOTIONS, f"unexpected coverage: {emotions}"
    assert "admiration" not in emotions, (
        "admiration has no locked layer to confirm at (amendment clause 2); "
        "it re-enters only via the residualization experiment"
    )
    per_emotion: dict[str, set[str]] = {}
    for r in rows:
        per_emotion.setdefault(r["emotion"], set()).add(r["family"])
    for e in EXPECTED_EMOTIONS:
        assert len(per_emotion[e]) >= MIN_FAMILIES_PER_EMOTION, (
            f"{e}: {len(per_emotion[e])} families, amendment requires "
            f">={MIN_FAMILIES_PER_EMOTION}"
        )


# ---------------------------------------------------------------------------
# Held-out-ness: nothing may be recycled from the spent June set
# ---------------------------------------------------------------------------

def test_no_family_name_reuse(rows, june_rows):
    reused = ({(r["emotion"], r["family"]) for r in rows}
              & {(r["emotion"], r["family"]) for r in june_rows})
    assert not reused, f"family names reused from the spent June set: {reused}"


def test_no_sentence_reuse(rows, june_rows):
    reused = {r["text"] for r in rows} & {r["text"] for r in june_rows}
    assert not reused, f"sentences reused from the spent June set: {reused}"


def test_no_template_skeleton_reuse(rows, june_rows):
    """Catches the same sentence re-used with a different x-list."""
    def skel(rs):
        return {"".join(ch for ch in r["text"] if not ch.isdigit()) for r in rs}

    reused = skel(rows) & skel(june_rows)
    assert not reused, f"templates reused from the spent June set: {reused}"


# ---------------------------------------------------------------------------
# Freeze integrity
# ---------------------------------------------------------------------------

def test_build_is_deterministic(builder):
    a = json.dumps(builder.build(), sort_keys=True)
    b = json.dumps(builder.build(), sort_keys=True)
    assert a == b


def test_written_file_matches_builder(rows):
    if not CONFIRM_JSONL.exists():
        pytest.skip("confirmation set not built yet")
    on_disk = [json.loads(ln) for ln in CONFIRM_JSONL.read_text().splitlines() if ln.strip()]
    assert on_disk == rows, "data/public/intensity_confirmation.jsonl is stale — rebuild"


def test_md5_matches_frozen_registry():
    """Tamper-evidence: the frozen hash must match the file on disk."""
    if not CONFIRM_JSONL.exists():
        pytest.skip("confirmation set not built yet")
    reg = yaml.safe_load((_repo_root / "configs" / "stimuli_hashes.yaml").read_text()) or {}
    expected = reg.get(CONFIRM_JSONL.name)
    assert expected, "confirmation set is not registered in configs/stimuli_hashes.yaml"
    actual = hashlib.md5(CONFIRM_JSONL.read_bytes()).hexdigest()
    assert actual == expected, "frozen confirmation set changed after locking!"
