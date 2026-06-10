"""Derive paper-method (story) steering vectors per layer.

Implements steps 3-5 of the paper-method emotion-vector pipeline
(Sofroniew et al. 2026), composing the pure-NumPy primitives from
``src/llm_psych/steering.py``:

1. Per-emotion mean of pooled story activations.
2. Subtract the cross-emotion grand mean (centering).
3. Project the result orthogonal to the top PCs of the neutral
   activation set (PCs explaining
   ``derivation.project_out.var_threshold`` of variance, default 0.5).

Run once per model after generation and extraction are complete:

.. code-block:: bash

    uv run python scripts/derive_story_steering_vectors.py \\
        model=gemma3_4b derivation=story

This script reads all emotions found at
``activations/<model_key>-story/<emotion>.npz``. The neutral file
(``neutral.npz``) is required and is *not* included in the centered
emotion set — it is used only to fit the projection-out basis.

Outputs
-------
For each candidate layer ``L`` and each non-neutral emotion ``e``:

``steering_vectors/<model_key>-story/<emotion>_layer<L>.npy``
    Float32 array of shape ``(hidden_dim,)``.

``steering_vectors/<model_key>-story/manifest.yaml``
    Run-level metadata: model SHA, n_stories per emotion, var_threshold,
    pool_start_token, git SHA, layers, hidden_dim.
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
from llm_psych.steering import (
    derive_story_vectors,
    fit_neutral_pcs,
    project_out,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pre-flight + helpers
# --------------------------------------------------------------------------

def _check_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=_repo_root,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before "
            "running experiments.\n" + result.stdout
        )


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
    emotions: list[str] = []
    for path in sorted(act_dir.glob("*.npz")):
        name = path.stem
        if name == "neutral":
            continue
        emotions.append(name)
    if not emotions:
        raise FileNotFoundError(
            f"No emotion .npz files in {act_dir} (excluding neutral.npz)."
        )
    return emotions


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.derivation.method != "story":
        raise ValueError(
            "derive_story_steering_vectors.py requires derivation=story; "
            f"got derivation={cfg.derivation.method}."
        )

    _check_clean_git()

    model_cfg_raw = cfg.model
    model_key = model_cfg_raw.hf_model_id.split("/")[-1]
    act_dir = _repo_root / cfg.paths.activations_dir / f"{model_key}-story"

    emotions = _discover_emotions(act_dir)
    neutral_path = act_dir / "neutral.npz"
    if not neutral_path.exists():
        raise FileNotFoundError(
            f"neutral.npz not found in {act_dir}. The paper-method "
            "requires a neutral activation set to fit the projection-out "
            "basis. Run scripts/extract_story_activations.py with "
            "emotion=neutral first."
        )

    log.info(
        "Found %d emotions: %s (model=%s)",
        len(emotions), emotions, model_key,
    )

    # Pseudo ModelConfig just for layer-range derivation (no model load).
    pseudo_cfg = ModelConfig(
        n_layers=int(model_cfg_raw.n_layers),
        hidden_size=int(model_cfg_raw.hidden_size),
        hf_model_id=str(model_cfg_raw.hf_model_id),
        hf_revision=str(model_cfg_raw.hf_revision) if model_cfg_raw.hf_revision else None,
    )
    layers = probe_layer_range(pseudo_cfg)
    log.info(
        "Deriving story steering vectors on layers %d..%d (n=%d)",
        layers[0], layers[-1], len(layers),
    )

    var_threshold = float(cfg.derivation.project_out.var_threshold)

    # Lazy-load .npz handles (mmap-style).
    npz_handles: dict[str, np.lib.npyio.NpzFile] = {
        emo: np.load(act_dir / f"{emo}.npz") for emo in emotions
    }
    neutral_npz = np.load(neutral_path)

    out_dir = (
        _repo_root / cfg.paths.steering_vectors_dir / f"{model_key}-story"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_stories_per_emotion = {
        emo: int(npz_handles[emo][f"layer_{layers[0]}"].shape[0])
        for emo in emotions
    }
    n_neutral = int(neutral_npz[f"layer_{layers[0]}"].shape[0])

    # --- per-layer derivation ---
    for lyr in layers:
        key = f"layer_{lyr}"
        per_emotion_acts = {
            emo: npz_handles[emo][key].astype(np.float64) for emo in emotions
        }
        neutral_acts = neutral_npz[key].astype(np.float64)

        raw_vectors = derive_story_vectors(per_emotion_acts)
        pcs = fit_neutral_pcs(neutral_acts, var_threshold=var_threshold)
        stacked = np.stack(list(raw_vectors.values()), axis=0)
        cleaned = project_out(stacked, pcs)

        for i, emo in enumerate(raw_vectors.keys()):
            out_path = out_dir / f"{emo}_layer{lyr}.npy"
            v = cleaned[i].astype(np.float32)
            np.save(out_path, v)
            log.debug(
                "Saved %s (layer=%d emotion=%s n_pcs=%d norm=%.4f)",
                out_path, lyr, emo, pcs.shape[0], float(np.linalg.norm(v)),
            )
        log.info(
            "Layer %d: derived %d vectors, projected out %d neutral PCs",
            lyr, len(raw_vectors), pcs.shape[0],
        )

    # --- manifest ---
    manifest = {
        "method": "story",
        "model_id": pseudo_cfg.hf_model_id,
        "model_sha": pseudo_cfg.hf_revision or "",
        "git_sha": _git_sha(),
        "layers": layers,
        "hidden_dim": int(pseudo_cfg.hidden_size),
        "n_neutral": n_neutral,
        "n_stories_per_emotion": n_stories_per_emotion,
        "emotions": list(emotions),
        "pool_start_token": int(cfg.derivation.pool_start_token),
        "center": str(cfg.derivation.center),
        "project_out": {
            "source": str(cfg.derivation.project_out.source),
            "var_threshold": var_threshold,
        },
    }
    manifest_path = out_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    log.info("Saved manifest to %s", manifest_path)
    log.info("Done — story-method steering vectors for %s", model_key)


if __name__ == "__main__":
    main()
