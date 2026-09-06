"""Export a track's *derivation frame* so later runs can reuse it.

The story method turns per-emotion mean activations into vectors with two
track-level constants, per layer:

* the **cross-emotion grand mean**, subtracted from every emotion
  (``derive_story_vectors``), and
* the **neutral PC basis**, projected out of every emotion
  (``fit_neutral_pcs`` on ``neutral.npz``).

Both depend on the track as a whole, not on any one emotion. That is what
makes adding an emotion awkward: recomputing the grand mean over a larger
set translates *every* vector by a shared constant, so a naive "just add a
cell" changes vectors that are already published.

Note what that change actually is, though. Adding ``k`` emotions to a set
of ``E`` moves every vector by the *same* offset

.. math::

    \\delta = \\frac{k\\,(\\bar\\mu_E - \\bar\\mu_{\\text{new}})}{E + k}

— a rigid translation of the whole set, not a per-emotion distortion. Every
pairwise difference ``v_a - v_b`` is exactly unchanged, and because the
neutral projection is linear the translation survives it as ``P(delta)``.

So instead of redefining the frame each time, freeze it once and derive
later emotions *into* it. This script writes that frame; pass it to
``scripts/derive_story_steering_vectors.py`` as
``derivation.frame_from=<path>`` to place new cells in an existing track's
coordinates using only their own activations.

This script is read-only with respect to existing steering vectors: it
never writes into ``<emotion>_layer<L>.npy``.

Outputs
-------
``steering_vectors/<model_key>-<track>/frame.npz``
    ``grand_mean_layer_<L>`` (float64, ``(hidden_dim,)``) and
    ``pcs_layer_<L>`` (float32, ``(k, hidden_dim)``) for every layer.

``steering_vectors/<model_key>-<track>/frame.yaml``
    Human-readable provenance: the emotion set the mean was taken over,
    layers, var_threshold, model SHA, git SHA.

Usage
-----
::

    uv run python scripts/export_derivation_frame.py \\
        model=llama31_8b derivation=story track=story-wheel32

Add ``+push=true`` to upload the frame to the private dataset at
``steering_vectors/<model_key>-<track>/frame.{npz,yaml}`` — the path
``scripts/run_paper_emotions.sh`` fetches from. Worth doing: the frame is
cheap to keep but needs the track's whole activation set to rebuild, so
losing it with an ephemeral pod means re-downloading that set.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

# --- make src/ importable when running without ``pip install -e .``
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from llm_psych.models import ModelConfig, probe_layer_range
from llm_psych.paths import track_slug
from llm_psych.steering import fit_neutral_pcs

log = logging.getLogger(__name__)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_repo_root,
    )
    return result.stdout.strip()


def _discover_emotions(act_dir: Path) -> list[str]:
    """List emotion names present as ``<emotion>.npz`` (excluding neutral)."""
    if not act_dir.is_dir():
        raise FileNotFoundError(f"Activation dir not found: {act_dir}")
    emotions = [
        p.stem for p in sorted(act_dir.glob("*.npz")) if p.stem != "neutral"
    ]
    if not emotions:
        raise FileNotFoundError(
            f"No emotion .npz files in {act_dir} (excluding neutral.npz)."
        )
    return emotions


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.derivation.method != "story":
        raise ValueError(
            "export_derivation_frame.py requires derivation=story; "
            f"got derivation={cfg.derivation.method}."
        )

    model_cfg_raw = cfg.model
    model_key = model_cfg_raw.hf_model_id.split("/")[-1]
    track: str = str(cfg.track)
    slug = track_slug(model_key, track)
    act_dir = _repo_root / cfg.paths.activations_dir / slug

    emotions = _discover_emotions(act_dir)
    neutral_path = act_dir / "neutral.npz"
    if not neutral_path.exists():
        raise FileNotFoundError(
            f"neutral.npz not found in {act_dir}; it defines the "
            "projection-out basis and is part of the frame."
        )

    pseudo_cfg = ModelConfig(
        n_layers=int(model_cfg_raw.n_layers),
        hidden_size=int(model_cfg_raw.hidden_size),
        hf_model_id=str(model_cfg_raw.hf_model_id),
        hf_revision=str(model_cfg_raw.hf_revision) if model_cfg_raw.hf_revision else None,
    )
    layers = probe_layer_range(pseudo_cfg)
    var_threshold = float(cfg.derivation.project_out.var_threshold)

    log.info(
        "Exporting frame for %s: %d emotions, layers %d..%d",
        slug, len(emotions), layers[0], layers[-1],
    )

    npz_handles = {emo: np.load(act_dir / f"{emo}.npz") for emo in emotions}
    neutral_npz = np.load(neutral_path)

    payload: dict[str, np.ndarray] = {}
    n_pcs: dict[int, int] = {}
    for lyr in layers:
        key = f"layer_{lyr}"
        # Grand mean over per-emotion means: each emotion contributes
        # equally, matching derive_story_vectors exactly.
        per_emotion_means = np.stack(
            [
                npz_handles[emo][key].astype(np.float64).mean(axis=0)
                for emo in emotions
            ],
            axis=0,
        )
        payload[f"grand_mean_{key}"] = per_emotion_means.mean(axis=0)

        pcs = fit_neutral_pcs(
            neutral_npz[key].astype(np.float64), var_threshold=var_threshold
        )
        payload[f"pcs_{key}"] = pcs
        n_pcs[lyr] = int(pcs.shape[0])

    out_dir = _repo_root / cfg.paths.steering_vectors_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / "frame.npz"
    np.savez(frame_path, **payload)

    meta = {
        "kind": "derivation_frame",
        "method": "story",
        "track": track,
        "model_id": str(model_cfg_raw.hf_model_id),
        "model_sha": pseudo_cfg.hf_revision or "",
        "git_sha": _git_sha(),
        "layers": list(layers),
        "hidden_dim": int(pseudo_cfg.hidden_size),
        "n_emotions": len(emotions),
        "emotions": list(emotions),
        "n_neutral": int(neutral_npz[f"layer_{layers[0]}"].shape[0]),
        "project_out": {
            "source": str(cfg.derivation.project_out.source),
            "var_threshold": var_threshold,
        },
        "n_pcs_per_layer": n_pcs,
        "note": (
            "Centering constant and projection basis for this track. Derive "
            "additional emotions into these coordinates with "
            "derivation.frame_from=<this file>; the resulting vectors share "
            "this track's frame and the existing vectors stay untouched."
        ),
    }
    meta_path = out_dir / "frame.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    log.info("Wrote %s (%d layers, %d emotions)", frame_path, len(layers), len(emotions))
    log.info("Wrote %s", meta_path)

    if bool(cfg.get("push", False)):
        _push(frame_path, meta_path, slug, cfg)


def _push(frame_path: Path, meta_path: Path, slug: str, cfg: DictConfig) -> None:
    """Upload the frame to the private dataset.

    The repo path is hard-coded to ``steering_vectors/<slug>/`` to match the
    rest of the dataset layout (``scripts/run_wheel.sh``) and the path
    ``scripts/run_paper_emotions.sh`` fetches from; a renamed local
    ``paths.steering_vectors_dir`` must not move it.
    """
    from dotenv import load_dotenv

    load_dotenv(_repo_root / ".env")

    from huggingface_hub import HfApi

    from llm_psych.hf_sync import DEFAULT_DATASET_REPO_ID

    repo = str(cfg.get("dataset_repo", DEFAULT_DATASET_REPO_ID))
    api = HfApi()
    for path in (frame_path, meta_path):
        path_in_repo = f"steering_vectors/{slug}/{path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"export_derivation_frame: {slug} {path.name}",
        )
        log.info("Pushed %s -> %s/%s", path.name, repo, path_in_repo)


if __name__ == "__main__":
    main()
