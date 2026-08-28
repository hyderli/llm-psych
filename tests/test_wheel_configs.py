"""Tests for the generated Plutchik-wheel emotion configs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
# scripts/ is not an installed package; make it importable without relying
# on the caller's PYTHONPATH.
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_wheel_configs import build_cells, load_spec  # noqa: E402
_EMOTION_DIR = _REPO_ROOT / "configs" / "emotion"
_LABELS_MD = _EMOTION_DIR / "EMOTION_LABELS.md"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    return build_cells(load_spec())


@pytest.fixture(scope="module")
def wheel_files() -> list[Path]:
    return sorted(_EMOTION_DIR.glob("wheel_*.yaml"))


def test_thirty_three_corpora(cells, wheel_files):
    assert len(cells) == 33
    assert len(wheel_files) == 33


def test_ring_cells_cover_eight_axes_by_three_rings(cells):
    ring = [c for c in cells if c["kind"] == "ring"]
    assert len(ring) == 24
    assert len({c["axis"] for c in ring}) == 8
    for axis in {c["axis"] for c in ring}:
        rings = sorted(c["ring"] for c in ring if c["axis"] == axis)
        assert rings == ["high", "low", "middle"]


def test_dyad_components_are_two_distinct_axes(cells):
    axes = {c["axis"] for c in cells if c["kind"] == "ring"}
    dyads = [c for c in cells if c["kind"] == "dyad"]
    assert len(dyads) == 8
    for dyad in dyads:
        assert len(dyad["components"]) == 2
        assert len(set(dyad["components"])) == 2
        assert set(dyad["components"]) <= axes


def test_labels_unique(cells):
    labels = [c["label"] for c in cells]
    assert len(set(labels)) == len(labels)


def test_labels_disjoint_from_legacy_namespace(cells):
    """Wheel labels must not collide with the legacy 1-20 labels."""
    legacy = set()
    for path in _EMOTION_DIR.glob("*.yaml"):
        if path.name.startswith("wheel_"):
            continue
        legacy.add(yaml.safe_load(path.read_text())["label"])
    wheel = {c["label"] for c in cells if c["kind"] != "neutral"}
    assert wheel.isdisjoint(legacy), sorted(wheel & legacy)


def test_documented_legacy_labels_untouched(cells):
    """Every label in EMOTION_LABELS.md stays outside the wheel namespace."""
    documented = {int(m) for m in re.findall(r"^\|\s*(\d+)\s*\|", _LABELS_MD.read_text(), re.M)}
    wheel = {c["label"] for c in cells if c["kind"] != "neutral"}
    assert documented, "no legacy labels parsed from EMOTION_LABELS.md"
    assert documented.isdisjoint(wheel)


def test_every_generated_file_is_prefixed_and_no_legacy_overwritten(wheel_files):
    for path in wheel_files:
        assert path.name.startswith("wheel_")
        bare = _EMOTION_DIR / path.name[len("wheel_"):]
        if bare.exists():
            # A same-named legacy config may exist; it must be a different file.
            assert bare.read_text() != path.read_text()


def test_contempt_dyad_does_not_inherit_the_ekman_config():
    dyad = yaml.safe_load((_EMOTION_DIR / "wheel_contempt.yaml").read_text())
    ekman = yaml.safe_load((_EMOTION_DIR / "contempt.yaml").read_text())
    assert dyad["label"] == 196
    assert ekman["label"] == 16
    assert dyad["label"] != ekman["label"]


def test_generated_files_match_spec():
    """configs/emotion/wheel_*.yaml must be regenerable from configs/wheel.yaml."""
    from scripts.build_wheel_configs import render

    spec = load_spec()
    for cell in build_cells(spec):
        path = _EMOTION_DIR / f"wheel_{cell['name']}.yaml"
        assert path.exists(), path
        assert path.read_text() == render(cell, spec["track"])
