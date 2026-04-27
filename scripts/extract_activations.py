"""Extract residual-stream activations from a model and save to .npz.

Usage (Mac M5, small dev model):

    uv run python scripts/extract_activations.py model=qwen25_05b emotion=anger

Usage (cloud, primary target):

    uv run python scripts/extract_activations.py \
        model=llama31_8b emotion=anger \
        extract.batch_size=16

Outputs per (model, emotion, split):
  activations/<model_key>/<emotion>_<split>.npz   — layer arrays (float16)
  activations/<model_key>/<emotion>_<split>.meta.parquet — prompt_id index

The .npz arrays are named ``layer_<L>`` for each candidate layer in
``models.probe_layer_range(cfg)``.

Pre-flight checks (aborted if any fails):
  1. git working tree is clean.
  2. Stimulus parquet files exist.

Notes
-----
This script is the first end-to-end pipeline step; it does not assume
that probes or steering vectors exist yet. Re-running it is safe:
existing .npz files are overwritten (not appended to).
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
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from llm_psych.hooks import ResidualStreamRecorder
from llm_psych.models import load_model, probe_layer_range

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


def _check_stimuli(data_dir: Path, prompt_file: str, neutral_file: str) -> None:
    for fname in (prompt_file, neutral_file):
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Stimulus file not found: {p}\n"
                "Run data preparation scripts first or check paths.data_dir."
            )


# --------------------------------------------------------------------------
# Tokenisation + batched forward passes
# --------------------------------------------------------------------------

def _tokenize_batch(
    texts: list[str],
    tokenizer,
    device: str,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _infer_model_device(model) -> str:
    """Return a device string suitable for tokenizer-output placement."""
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        return "cpu"


# --------------------------------------------------------------------------
# Core extraction
# --------------------------------------------------------------------------

def _extract_split(
    texts: list[str],
    prompt_ids: list[str],
    model,
    tokenizer,
    layers: list[int],
    token_position,
    batch_size: int,
    dtype_str: str,
) -> dict[str, np.ndarray]:
    """Run batched forward passes and return per-layer activation arrays.

    Returns
    -------
    dict mapping ``"layer_<L>"`` → float16 array of shape
    ``(n_prompts, hidden_dim)`` when token_position == "last".
    Also includes ``"prompt_ids"`` as a bytes array for the companion
    parquet.
    """
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.float32
    device = _infer_model_device(model)

    # Pre-allocate result buffers after the first batch tells us hidden_dim.
    layer_buffers: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    with ResidualStreamRecorder(
        model, layers=layers, token_position=token_position, dtype=torch_dtype
    ) as rec:
        for start in tqdm(range(0, len(texts), batch_size), desc="batches"):
            batch_texts = texts[start : start + batch_size]
            enc = _tokenize_batch(batch_texts, tokenizer, device)
            rec.clear()
            with torch.no_grad():
                model(**enc)
            for l in layers:
                layer_buffers[l].append(rec.activations[l].cpu().numpy())

    arrays: dict[str, np.ndarray] = {}
    for l in layers:
        arrays[f"layer_{l}"] = np.concatenate(layer_buffers[l], axis=0)

    return arrays


# --------------------------------------------------------------------------
# Main (Hydra entry point)
# --------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # --- pre-flight ---
    _check_clean_git()

    data_dir = _repo_root / cfg.paths.data_dir
    _check_stimuli(data_dir, cfg.emotion.prompt_file, cfg.emotion.neutral_file)

    # --- load model ---
    model_cfg_raw = cfg.model
    lm = load_model(
        hf_model_id=model_cfg_raw.hf_model_id,
        revision=model_cfg_raw.hf_revision or None,
        torch_dtype=getattr(torch, model_cfg_raw.torch_dtype),
        device_map=model_cfg_raw.device_map,
        trust_remote_code=model_cfg_raw.trust_remote_code,
    )

    layers = probe_layer_range(lm.cfg)
    log.info(
        "Probing layers %d–%d (%d total)", layers[0], layers[-1], len(layers)
    )

    # --- load stimuli ---
    emotion_name: str = cfg.emotion.name
    all_prompts = pd.read_parquet(data_dir / cfg.emotion.prompt_file)
    neutral_prompts = pd.read_parquet(data_dir / cfg.emotion.neutral_file)

    emotion_prompts = all_prompts[all_prompts["emotion"] == emotion_name]

    out_base = _repo_root / cfg.paths.activations_dir
    # Derive a filesystem-safe model key from the model id (e.g. "Qwen2.5-0.5B-Instruct")
    model_key = lm.cfg.hf_model_id.split("/")[-1]
    out_dir = out_base / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- extract per split ---
    for split in ("train", "test"):
        for source_name, df in (
            (emotion_name, emotion_prompts),
            ("neutral", neutral_prompts),
        ):
            split_df = df[df["split"] == split]
            if split_df.empty:
                log.warning(
                    "No %s prompts for source=%s split=%s — skipping.",
                    emotion_name,
                    source_name,
                    split,
                )
                continue

            texts = split_df["text"].tolist()
            prompt_ids = split_df["id"].astype(str).tolist()

            log.info(
                "Extracting: source=%s split=%s n=%d", source_name, split, len(texts)
            )
            arrays = _extract_split(
                texts=texts,
                prompt_ids=prompt_ids,
                model=lm.model,
                tokenizer=lm.tokenizer,
                layers=layers,
                token_position=cfg.extract.token_position,
                batch_size=cfg.extract.batch_size,
                dtype_str=cfg.extract.dtype,
            )

            npz_path = out_dir / f"{source_name}_{split}.npz"
            np.savez_compressed(npz_path, **arrays)
            log.info("Saved %s", npz_path)

            meta_path = out_dir / f"{source_name}_{split}.meta.parquet"
            pd.DataFrame({"prompt_id": prompt_ids}).to_parquet(meta_path, index=False)
            log.info("Saved %s", meta_path)

    log.info("Done — activations for %s × %s", model_key, emotion_name)


if __name__ == "__main__":
    main()
