"""Train logistic-regression probes from extracted activations.

Implements H1 (linear probe accessibility): for each candidate layer in
``models.probe_layer_range(cfg)``, fit a binary one-vs-rest logistic
probe distinguishing emotion vs neutral activations, evaluate on the
held-out test split with AUC + 1000-bootstrap CI, and save the fitted
probe + metadata.

Usage (Mac dev model)::

    uv run python scripts/train_probes.py model=qwen25_05b emotion=anger

Usage (cloud, primary target)::

    uv run python scripts/train_probes.py model=llama31_8b emotion=anger

Inputs (must exist; produced by ``scripts/extract_activations.py``):
  activations/<model_key>/<emotion>_train.npz
  activations/<model_key>/neutral_train.npz
  activations/<model_key>/<emotion>_test.npz
  activations/<model_key>/neutral_test.npz

Outputs per (model, emotion):
  probes/<model_key>/<emotion>_layer<L>.joblib       (fitted estimator)
  probes/<model_key>/<emotion>_layer<L>.yaml          (ProbeMeta sidecar)
  probes/<model_key>/<emotion>_summary.csv            (AUC per layer)
  steering_vectors/<model_key>/<emotion>_layer<L>.npy (CAA vector at best layer)

Pre-flight checks (aborted if any fails):
  1. git working tree is clean.
  2. Required .npz files for (model, emotion) exist.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

# --- make src/ importable when running without `pip install -e .`
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from llm_psych.models import ModelConfig, probe_layer_range
from llm_psych.probes import (
    ProbeMeta,
    derive_steering_vector,
    evaluate,
    fit,
    save,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pre-flight checks
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


def _check_activations(act_dir: Path, emotion: str) -> None:
    required = [
        f"{emotion}_train.npz",
        f"neutral_train.npz",
        f"{emotion}_test.npz",
        f"neutral_test.npz",
    ]
    missing = [f for f in required if not (act_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing activation files in {act_dir}:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/extract_activations.py first."
        )


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def _stack_split(
    emotion_npz: np.lib.npyio.NpzFile,
    neutral_npz: np.lib.npyio.NpzFile,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for one layer: emotion=1, neutral=0.

    Casts float16 → float64 for sklearn compatibility.
    """
    key = f"layer_{layer}"
    X_emo = emotion_npz[key].astype(np.float64)
    X_neu = neutral_npz[key].astype(np.float64)
    X = np.concatenate([X_emo, X_neu], axis=0)
    y = np.concatenate(
        [np.ones(len(X_emo), dtype=int), np.zeros(len(X_neu), dtype=int)]
    )
    return X, y


# --------------------------------------------------------------------------
# Story-method activations (primary H1, 2026-06-13 amendment)
# --------------------------------------------------------------------------
# The story pipeline writes one pooled-activation file per emotion
# (``activations/<model_key>-story/<emotion>.npz``, keyed ``layer_<L>``,
# shape ``(n_stories, hidden)``) with NO train/test split. We make a
# seeded per-story split here so the probe (unchanged: L2 logistic) trains
# on token-50-mean story activations instead of last-token CAA activations.

def _check_story_activations(act_dir: Path, emotion: str) -> None:
    required = [f"{emotion}.npz", "neutral.npz"]
    missing = [f for f in required if not (act_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing story activation files in {act_dir}:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/extract_story_activations.py first "
            "(derivation=story)."
        )


def _split_indices(n: int, seed: int, test_frac: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Seeded train/test index split (70/30 by default).

    Guarantees at least one test item when ``n >= 2`` so tiny smoke-test
    corpora still yield an evaluable split.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac))) if n >= 2 else 0
    return perm[n_test:], perm[:n_test]  # (train_idx, test_idx)


def _slice_layers(npz, layers: list[int], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Build a layer→array dict for the given row indices (drop-in for NpzFile)."""
    return {f"layer_{lyr}": npz[f"layer_{lyr}"][idx] for lyr in layers}


def _load_story_splits(
    act_dir: Path, emotion: str, layers: list[int], *, seed: int = 42
) -> tuple[dict, dict, dict, dict, list[int]]:
    """Load story activations and produce train/test split dicts.

    Returns ``(emo_train, emo_test, neu_train, neu_test, use_layers)`` where
    each split is a ``{f"layer_<L>": (n, hidden)}`` dict usable by
    ``_stack_split`` / ``derive_steering_vector`` exactly like an NpzFile.
    The neutral split is fixed by ``seed`` so it is identical across emotions.
    """
    emo = np.load(act_dir / f"{emotion}.npz")
    neu = np.load(act_dir / "neutral.npz")
    present = {int(k.split("_")[1]) for k in emo.files if k.startswith("layer_")}
    use_layers = [lyr for lyr in layers if lyr in present]
    if not use_layers:
        raise ValueError(
            f"No requested probe layers {layers} present in {act_dir/f'{emotion}.npz'} "
            f"(found {sorted(present)})."
        )
    n_emo = emo[f"layer_{use_layers[0]}"].shape[0]
    n_neu = neu[f"layer_{use_layers[0]}"].shape[0]
    emo_tr_i, emo_te_i = _split_indices(n_emo, seed)
    neu_tr_i, neu_te_i = _split_indices(n_neu, seed)
    return (
        _slice_layers(emo, use_layers, emo_tr_i),
        _slice_layers(emo, use_layers, emo_te_i),
        _slice_layers(neu, use_layers, neu_tr_i),
        _slice_layers(neu, use_layers, neu_te_i),
        use_layers,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _check_clean_git()

    model_cfg = cfg.model
    model_key = model_cfg.hf_model_id.split("/")[-1]
    emotion_name: str = cfg.emotion.name

    # Primary derivation is the story method (2026-06-13 amendment); CAA is
    # the secondary baseline. Story activations live under "<model_key>-story"
    # and are split per-story here; CAA activations are pre-split on disk.
    is_story = str(cfg.derivation.method) == "story"
    suffix = "-story" if is_story else ""
    act_dir = _repo_root / cfg.paths.activations_dir / (model_key + suffix)

    # Layer range depends only on n_layers, not on a loaded model.
    pseudo_cfg = ModelConfig(
        n_layers=int(model_cfg.n_layers),
        hidden_size=int(model_cfg.hidden_size),
        hf_model_id=str(model_cfg.hf_model_id),
        hf_revision=str(model_cfg.hf_revision) if model_cfg.hf_revision else None,
    )
    layers = probe_layer_range(pseudo_cfg)

    # --- load activations ---
    if is_story:
        _check_story_activations(act_dir, emotion_name)
        emo_train, emo_test, neu_train, neu_test, layers = _load_story_splits(
            act_dir, emotion_name, layers
        )
    else:
        _check_activations(act_dir, emotion_name)
        emo_train = np.load(act_dir / f"{emotion_name}_train.npz")
        neu_train = np.load(act_dir / "neutral_train.npz")
        emo_test = np.load(act_dir / f"{emotion_name}_test.npz")
        neu_test = np.load(act_dir / "neutral_test.npz")

    log.info("Training %s probes for %s × %s on layers %d–%d (%d total)",
             "story" if is_story else "CAA", model_key, emotion_name,
             layers[0], layers[-1], len(layers))

    n_train = emo_train[f"layer_{layers[0]}"].shape[0] + neu_train[f"layer_{layers[0]}"].shape[0]
    n_test = emo_test[f"layer_{layers[0]}"].shape[0] + neu_test[f"layer_{layers[0]}"].shape[0]

    probes_dir = _repo_root / cfg.paths.probes_dir / (model_key + suffix)
    probes_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    fitted: dict[int, tuple] = {}  # layer → (clf, result)

    for layer in layers:
        X_tr, y_tr = _stack_split(emo_train, neu_train, layer)
        X_te, y_te = _stack_split(emo_test, neu_test, layer)

        clf = fit(X_tr, y_tr, C=1.0)
        result = evaluate(clf, X_te, y_te, n_bootstrap=1000)

        meta = ProbeMeta(
            emotion=emotion_name,
            layer=layer,
            model_id=pseudo_cfg.hf_model_id,
            model_sha=pseudo_cfg.hf_revision,
            n_train=n_train,
            n_test=n_test,
            auc=result.auc,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            brier=result.brier,
            sklearn_c=1.0,
        )
        save(clf, meta, probes_dir)

        log.info("layer=%d  %s", layer, result)
        summary_rows.append({
            "layer": layer,
            "auc": result.auc,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "brier": result.brier,
        })
        fitted[layer] = (clf, result)

    # --- summary CSV ---
    summary_df = pd.DataFrame(summary_rows).sort_values("layer")
    summary_path = probes_dir / f"{emotion_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved probe summary to %s", summary_path)

    # --- best layer + steering vector ---
    best_lyr = max(fitted, key=lambda l: fitted[l][1].auc)
    best_auc = fitted[best_lyr][1].auc
    log.info("Best layer: %d (AUC=%.4f)", best_lyr, best_auc)

    if is_story:
        # Story steering vectors are the paper-method vectors (cross-emotion
        # centered + neutral-PC projected) produced by
        # scripts/derive_story_steering_vectors.py — not a CAA emotion−neutral
        # difference. Probes are this script's job; skip vector derivation.
        log.info(
            "Story mode: probes saved to %s. Derive steering vectors with "
            "scripts/derive_story_steering_vectors.py (derivation=story).",
            probes_dir,
        )
    else:
        sv = derive_steering_vector(
            emo_train[f"layer_{best_lyr}"].astype(np.float64),
            neu_train[f"layer_{best_lyr}"].astype(np.float64),
        )
        sv_dir = _repo_root / cfg.paths.steering_vectors_dir / model_key
        sv_dir.mkdir(parents=True, exist_ok=True)
        sv_path = sv_dir / f"{emotion_name}_layer{best_lyr}.npy"
        np.save(sv_path, sv)
        log.info("Saved steering vector to %s (norm=%.4f)", sv_path, float(np.linalg.norm(sv)))

    log.info("Done — probes for %s × %s", model_key, emotion_name)


if __name__ == "__main__":
    main()
