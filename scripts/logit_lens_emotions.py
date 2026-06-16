#!/usr/bin/env python3
"""Logit-lens analysis of story-derived emotion vectors.

See plans/logit_lens.md for full specification.

Projects each emotion vector through the unembedding matrix W_U (with Gemma's
final RMSNorm applied first) to find which vocabulary tokens each emotion direction
upweights or downweights. Validates semantic content of the vectors.

Two-step workflow to avoid reloading the 4B model each time:

  Step 1 — extract W_U once (needs GPU):
      python scripts/logit_lens_emotions.py --extract-wu

  Step 2 — run logit-lens analysis (CPU-only):
      python scripts/logit_lens_emotions.py --wu-path weights/gemma3_4b_wu.pt

Output:
    results/logit_lens/top_tokens.json
    results/logit_lens/logit_lens_table.csv
    figures/logit_lens/logit_lens_table_layer{L}.png   (layers 8, 16, 22, 28, 32)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["joy", "sadness", "admiration", "loathing"]
PRIMARY_LAYER = 22
PROBE_LAYERS = [8, 16, 22, 28, 32]
TOP_K = 20
N_DISPLAY = 10
MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_REPO_ID = "llm-psych/llm-psych-activations"

EMOTION_COLORS = {
    "joy": "#F4A261",
    "admiration": "#2A9D8F",
    "sadness": "#457B9D",
    "loathing": "#E76F51",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env_token() -> str | None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                val = line[len("HF_TOKEN="):].strip()
                if val:
                    os.environ.setdefault("HF_TOKEN", val)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def apply_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply RMSNorm: x / rms(x) * weight.  Mirrors Gemma's model.model.norm."""
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms * weight.astype(np.float64)).astype(np.float32)


def load_emotion_vectors(
    hf_prefix: str, repo_id: str, token: str | None, work_dir: Path
) -> dict[str, dict[int, np.ndarray]]:
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for emo in EMOTIONS:
        prefix = hf_prefix.strip("/")
        filename = f"{emo}_all_layers.pt"
        hf_path = f"{prefix}/{filename}" if prefix else filename
        local = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=hf_path, local_dir=str(work_dir), token=token,
        )
        layer_dict = torch.load(local, map_location="cpu", weights_only=True)
        vectors[emo] = {l: layer_dict[l].float().numpy() for l in layer_dict}
    return vectors


# ---------------------------------------------------------------------------
# Step 1: extract W_U and norm weight from model
# ---------------------------------------------------------------------------

def extract_and_save_wu(save_path: Path) -> None:
    """Load the Gemma 3 4B model, extract W_U and RMSNorm weight, save to disk."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading {MODEL_ID} (4-bit) to extract W_U ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    model.eval()

    # lm_head is not quantized by bitsandbytes — safe to extract as float
    wu = model.lm_head.weight.detach().cpu().float()  # [vocab_size, hidden_dim]
    norm_weight = model.model.norm.weight.detach().cpu().float()  # [hidden_dim]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"wu": wu, "norm_weight": norm_weight}, save_path)
    print(f"Saved W_U {tuple(wu.shape)} and norm_weight {tuple(norm_weight.shape)} → {save_path}")


# ---------------------------------------------------------------------------
# Step 2: logit-lens computation
# ---------------------------------------------------------------------------

def compute_logit_lens(
    vectors: dict[str, dict[int, np.ndarray]],
    wu: np.ndarray,          # [vocab_size, hidden_dim]
    norm_weight: np.ndarray,  # [hidden_dim]
    layers: list[int],
    top_k: int,
) -> pd.DataFrame:
    """Compute top/bottom tokens per emotion per layer.

    Returns a tidy DataFrame with columns:
        emotion, layer, direction, rank, token_id, token_str, logit
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = []
    for emo in EMOTIONS:
        for l in layers:
            v = vectors[emo][l]
            v_normed = apply_rms_norm(v, norm_weight)
            logits = wu @ v_normed  # [vocab_size]

            for direction, indices in [
                ("up",   np.argsort(logits)[::-1][:top_k]),
                ("down", np.argsort(logits)[:top_k]),
            ]:
                for rank, idx in enumerate(indices):
                    tok = tokenizer.decode([int(idx)]).strip()
                    rows.append({
                        "emotion": emo,
                        "layer": l,
                        "direction": direction,
                        "rank": rank,
                        "token_id": int(idx),
                        "token_str": tok,
                        "logit": float(logits[idx]),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_logit_lens_table(df: pd.DataFrame, layer: int, out_dir: Path, n: int = N_DISPLAY) -> None:
    """Table figure: emotions as columns, top-n up and down tokens as rows."""
    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(3.5 * len(EMOTIONS), n * 0.42 + 2))
    fig.suptitle(f"Logit-lens top tokens — layer {layer}", fontsize=13, y=1.01)

    for ax, emo in zip(axes, EMOTIONS):
        up_tokens   = df[(df["emotion"] == emo) & (df["layer"] == layer) & (df["direction"] == "up")
                        ].nsmallest(n, "rank")["token_str"].tolist()
        down_tokens = df[(df["emotion"] == emo) & (df["layer"] == layer) & (df["direction"] == "down")
                        ].nsmallest(n, "rank")["token_str"].tolist()

        ax.set_xlim(0, 1)
        ax.set_ylim(0, n * 2 + 1)
        ax.axis("off")

        color = EMOTION_COLORS[emo]
        ax.text(0.5, n * 2 + 0.5, emo.upper(), ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

        # Up tokens (top half)
        for i, tok in enumerate(up_tokens):
            y = n * 2 - i
            ax.add_patch(plt.Rectangle((0.05, y - 0.45), 0.9, 0.9,
                                        color=color, alpha=0.15, zorder=0))
            ax.text(0.5, y, f"▲ {tok}", ha="center", va="center", fontsize=8.5)

        # Divider
        ax.axhline(n, color="gray", linewidth=0.8, linestyle="--", xmin=0.05, xmax=0.95)

        # Down tokens (bottom half)
        for i, tok in enumerate(down_tokens):
            y = n - 1 - i
            ax.add_patch(plt.Rectangle((0.05, y - 0.45), 0.9, 0.9,
                                        color="#999999", alpha=0.12, zorder=0))
            ax.text(0.5, y, f"▼ {tok}", ha="center", va="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    path = out_dir / f"logit_lens_table_layer{layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract-wu", action="store_true",
                   help="Extract W_U from model and save (needs GPU). Then exit.")
    p.add_argument("--wu-path", type=Path,
                   default=Path(__file__).parent.parent / "weights" / "gemma3_4b_wu.pt",
                   help="Path to saved W_U file (from --extract-wu).")
    p.add_argument("--hf-prefix", default="steering_vectors/gemma3-4b-story")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--work-dir", type=Path,
                   default=Path(__file__).parent.parent / "steering_vectors" / "gemma3-4b-story")
    p.add_argument("--figures-dir", type=Path,
                   default=Path(__file__).parent.parent / "figures" / "logit_lens")
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "logit_lens")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.extract_wu:
        extract_and_save_wu(args.wu_path)
        return

    # --- Analysis mode (CPU-only) ---
    if not args.wu_path.exists():
        sys.exit(
            f"ERROR: {args.wu_path} not found. "
            "Run with --extract-wu on a GPU machine first."
        )

    token = load_env_token()
    if not token:
        sys.exit("ERROR: HF_TOKEN not found in .env or environment.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading W_U ...")
    saved = torch.load(args.wu_path, map_location="cpu", weights_only=True)
    wu = saved["wu"].numpy()
    norm_weight = saved["norm_weight"].numpy()
    print(f"  W_U shape: {wu.shape}, norm_weight shape: {norm_weight.shape}")

    print("Loading emotion vectors ...")
    vectors = load_emotion_vectors(args.hf_prefix, args.repo_id, token, args.work_dir)

    print("Computing logit lens ...")
    df = compute_logit_lens(vectors, wu, norm_weight, layers=PROBE_LAYERS, top_k=TOP_K)

    # Save tidy CSV
    df.to_csv(args.results_dir / "logit_lens_table.csv", index=False)

    # Save JSON for quick inspection
    top_json: dict = {}
    for layer in PROBE_LAYERS:
        top_json[f"layer_{layer}"] = {}
        for emo in EMOTIONS:
            sub = df[(df["emotion"] == emo) & (df["layer"] == layer)]
            top_json[f"layer_{layer}"][emo] = {
                "up":   sub[sub["direction"] == "up"].nsmallest(N_DISPLAY, "rank")["token_str"].tolist(),
                "down": sub[sub["direction"] == "down"].nsmallest(N_DISPLAY, "rank")["token_str"].tolist(),
            }
    with (args.results_dir / "top_tokens.json").open("w") as f:
        json.dump(top_json, f, indent=2)

    # Print layer-22 summary
    print(f"\n--- Top {N_DISPLAY} tokens at layer {PRIMARY_LAYER} ---")
    for emo in EMOTIONS:
        up   = top_json[f"layer_{PRIMARY_LAYER}"][emo]["up"]
        down = top_json[f"layer_{PRIMARY_LAYER}"][emo]["down"]
        print(f"  {emo:12s}  UP: {up[:5]}  DOWN: {down[:5]}")

    # Plots for each probe layer
    print("\nPlotting ...")
    for layer in PROBE_LAYERS:
        plot_logit_lens_table(df, layer, args.figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
