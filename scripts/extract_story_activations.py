"""Extract pooled residual-stream activations from generated stories.

Implements step 2 of the paper-method emotion-vector pipeline
(Sofroniew et al. 2026): for each story written by
``scripts/generate_emotion_stories.py``, run a forward pass and pool
residual-stream activations across token positions
``>= derivation.pool_start_token`` (default 50). One pooled vector per
story per layer.

Usage
-----

.. code-block:: bash

    uv run python scripts/extract_story_activations.py \\
        model=gemma3_4b derivation=story emotion=joy

Outputs per ``(model, emotion)`` pair:

* ``activations/<model_key>-story/<emotion>.npz`` — one array per
  candidate layer, each shaped ``(n_stories, hidden_dim)`` in float16.
* ``activations/<model_key>-story/<emotion>.meta.parquet`` —
  ``story_id`` row index aligning with the .npz rows.

Pre-flight: requires a clean git tree and an existing story parquet
produced by ``scripts/generate_emotion_stories.py`` for the same model
and emotion.
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
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from llm_psych.hooks import ResidualStreamRecorder
from llm_psych.models import load_model, probe_layer_range

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pre-flight
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


# --------------------------------------------------------------------------
# Per-story extraction
# --------------------------------------------------------------------------

def _extract_pooled(
    story_text: str,
    *,
    model,
    tokenizer,
    layers: list[int],
    pool_start_token: int,
    device: str,
    max_length: int = 1024,
) -> dict[int, np.ndarray] | None:
    """Run one forward pass on ``story_text`` and mean-pool from token N.

    Returns ``{layer: (hidden_dim,) float32}`` or ``None`` if the
    tokenized story has fewer than ``pool_start_token + 1`` tokens.
    """
    enc = tokenizer(
        story_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    seq_len = int(enc["input_ids"].shape[-1])
    if seq_len <= pool_start_token:
        return None

    enc = {k: v.to(device) for k, v in enc.items()}

    # Capture the trailing slice; pool inside the script.
    slc = slice(pool_start_token, None)
    with ResidualStreamRecorder(
        model, layers=layers, token_position=slc, dtype=torch.float32
    ) as rec:
        with torch.no_grad():
            model(**enc)
        pooled: dict[int, np.ndarray] = {}
        for lyr in layers:
            # shape: (1, seq_len - pool_start_token, hidden_dim)
            t = rec.activations[lyr]
            v = t.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            pooled[lyr] = v
    return pooled


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.derivation.method != "story":
        raise ValueError(
            "extract_story_activations.py requires derivation=story; "
            f"got derivation={cfg.derivation.method}."
        )

    _check_clean_git()

    emotion_name: str = cfg.emotion.name
    model_cfg_raw = cfg.model
    model_key = model_cfg_raw.hf_model_id.split("/")[-1]
    pool_start_token = int(cfg.derivation.pool_start_token)

    stories_path = (
        _repo_root / "data" / "derived" / "stories" / model_key
        / f"{emotion_name}.parquet"
    )
    if not stories_path.exists():
        raise FileNotFoundError(
            f"Story corpus not found: {stories_path}\n"
            "Run scripts/generate_emotion_stories.py first."
        )
    stories = pd.read_parquet(stories_path)
    log.info(
        "Loaded %d stories from %s (emotion=%s, model=%s)",
        len(stories), stories_path, emotion_name, model_key,
    )

    # --- load model ---
    lm = load_model(
        hf_model_id=model_cfg_raw.hf_model_id,
        revision=model_cfg_raw.hf_revision or None,
        torch_dtype=getattr(torch, model_cfg_raw.torch_dtype),
        device_map=model_cfg_raw.device_map,
        trust_remote_code=model_cfg_raw.trust_remote_code,
    )
    device = next(lm.model.parameters()).device.type
    layers = probe_layer_range(lm.cfg)
    log.info(
        "Pooling activations on layers %d..%d (n=%d) from token %d onward",
        layers[0], layers[-1], len(layers), pool_start_token,
    )

    # --- extract ---
    per_layer_rows: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    kept_ids: list[str] = []
    dropped = 0
    for _, row in tqdm(
        stories.iterrows(), total=len(stories), desc=f"stories[{emotion_name}]"
    ):
        pooled = _extract_pooled(
            row["story_text"],
            model=lm.model,
            tokenizer=lm.tokenizer,
            layers=layers,
            pool_start_token=pool_start_token,
            device=device,
        )
        if pooled is None:
            dropped += 1
            log.debug(
                "Dropped story id=%s: tokenized length <= pool_start_token=%d",
                row["id"], pool_start_token,
            )
            continue
        for lyr, v in pooled.items():
            per_layer_rows[lyr].append(v)
        kept_ids.append(str(row["id"]))

    if not kept_ids:
        raise RuntimeError(
            "All stories were shorter than pool_start_token; refusing "
            "to write empty activations."
        )

    # --- save ---
    out_dir = _repo_root / cfg.paths.activations_dir / f"{model_key}-story"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"layer_{lyr}": np.stack(per_layer_rows[lyr], axis=0).astype(np.float16)
        for lyr in layers
    }
    npz_path = out_dir / f"{emotion_name}.npz"
    np.savez_compressed(npz_path, **arrays)
    log.info("Saved %s (kept %d, dropped %d)", npz_path, len(kept_ids), dropped)

    meta_path = out_dir / f"{emotion_name}.meta.parquet"
    pd.DataFrame({"story_id": kept_ids}).to_parquet(meta_path, index=False)
    log.info("Saved %s", meta_path)


if __name__ == "__main__":
    main()
