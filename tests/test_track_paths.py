"""Tests for track-aware story-pipeline paths.

The track split exists to stop the Plutchik-wheel run from overwriting
the four-emotion corpora that the locked vectors were derived from: nine
wheel cells share a name with an existing corpus. See
``plans/plutchik-wheel-expansion.md`` §1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_psych.paths import (
    DEFAULT_TRACK,
    resolve_story_corpus,
    story_corpus_path,
    story_dir,
    track_slug,
)

_MODEL = "Llama-3.1-8B-Instruct"
_WHEEL = "story-wheel32"


def test_default_track_reproduces_legacy_layout(tmp_path: Path):
    """The default track must not move any existing artefact."""
    assert track_slug(_MODEL) == f"{_MODEL}-story"
    assert story_dir(tmp_path, _MODEL) == tmp_path / "data" / "derived" / "stories" / _MODEL
    assert story_corpus_path(tmp_path, _MODEL, DEFAULT_TRACK, "joy") == (
        tmp_path / "data" / "derived" / "stories" / _MODEL / "joy.parquet"
    )


def test_non_default_track_is_disjoint_from_the_default(tmp_path: Path):
    """No wheel path may resolve inside the four-emotion track's paths."""
    assert track_slug(_MODEL, _WHEEL) == f"{_MODEL}-story-wheel32"
    assert track_slug(_MODEL, _WHEEL) != track_slug(_MODEL)

    for emotion in ("joy", "sadness", "admiration", "loathing", "neutral"):
        legacy = story_corpus_path(tmp_path, _MODEL, DEFAULT_TRACK, emotion)
        wheel = story_corpus_path(tmp_path, _MODEL, _WHEEL, emotion)
        assert wheel != legacy
        assert legacy.parent != wheel.parent


def test_empty_track_is_rejected():
    with pytest.raises(ValueError):
        track_slug(_MODEL, "")


def test_default_track_falls_back_to_the_flat_layout(tmp_path: Path):
    """Corpora generated before the track split must still resolve."""
    legacy = tmp_path / "data" / "derived" / "stories" / _MODEL
    legacy.mkdir(parents=True)
    (legacy / "joy.parquet").write_text("")

    assert resolve_story_corpus(tmp_path, _MODEL, DEFAULT_TRACK, "joy") == legacy / "joy.parquet"


def test_wheel_track_never_falls_back_to_another_tracks_corpus(tmp_path: Path):
    """The safety property: a missing wheel corpus is an error, not a silent read.

    Falling back here would feed four-emotion stories into the wheel
    derivation and change every wheel vector via the grand mean.
    """
    legacy = tmp_path / "data" / "derived" / "stories" / _MODEL
    legacy.mkdir(parents=True)
    (legacy / "joy.parquet").write_text("")

    assert resolve_story_corpus(tmp_path, _MODEL, _WHEEL, "joy") is None


def test_wheel_track_resolves_its_own_corpus(tmp_path: Path):
    wheel = tmp_path / "data" / "derived" / "stories" / _MODEL / _WHEEL
    wheel.mkdir(parents=True)
    (wheel / "joy.parquet").write_text("")

    assert resolve_story_corpus(tmp_path, _MODEL, _WHEEL, "joy") == wheel / "joy.parquet"
