"""Track-aware artefact paths for the story pipeline.

A *track* is an independent story-method run over its own emotion set:
the four-emotion set (``story``, the default) and the Plutchik-wheel set
(``story-wheel32``) are different tracks. Because the pipeline names
corpora and activations after ``cfg.emotion.name`` alone, two tracks that
share an emotion name — and nine wheel cells do — would collide on disk
and silently overwrite each other. Every path that depends on the emotion
set therefore carries the track.

The default track reproduces the pre-track layout exactly, so existing
artefacts keep resolving:

``activations/<model_key>-story/``, ``steering_vectors/<model_key>-story/``
and ``data/derived/stories/<model_key>/<emotion>.parquet``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DEFAULT_TRACK",
    "track_slug",
    "story_dir",
    "story_corpus_path",
    "resolve_story_corpus",
]

DEFAULT_TRACK = "story"


def track_slug(model_key: str, track: str = DEFAULT_TRACK) -> str:
    """Directory name for per-track model artefacts, e.g. ``Llama-3.1-8B-Instruct-story``."""
    if not track:
        raise ValueError("track must be a non-empty string")
    return f"{model_key}-{track}"


def story_dir(repo_root: Path, model_key: str, track: str = DEFAULT_TRACK) -> Path:
    """Directory holding one track's generated story corpora for a model.

    The default track keeps the historical flat layout; any other track
    gets its own subdirectory, so a wheel run cannot overwrite the
    four-emotion corpora that the locked vectors were derived from.
    """
    base = repo_root / "data" / "derived" / "stories" / model_key
    return base if track == DEFAULT_TRACK else base / track


def story_corpus_path(
    repo_root: Path, model_key: str, track: str, emotion: str
) -> Path:
    """Path this track writes ``<emotion>.parquet`` to."""
    return story_dir(repo_root, model_key, track) / f"{emotion}.parquet"


def resolve_story_corpus(
    repo_root: Path, model_key: str, track: str, emotion: str
) -> Path | None:
    """Locate an existing corpus for reading, or return ``None``.

    Only the default track falls back to the historical flat layout. A
    non-default track must never silently read another track's corpus —
    that is exactly the cross-contamination the track split exists to
    prevent.
    """
    path = story_corpus_path(repo_root, model_key, track, emotion)
    if path.exists():
        return path
    if track != DEFAULT_TRACK:
        return None
    legacy = repo_root / "data" / "derived" / "stories" / model_key / f"{emotion}.parquet"
    return legacy if legacy.exists() else None
