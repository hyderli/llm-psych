#!/usr/bin/env python3
"""Geometry analysis of story-derived emotion vectors.

See plans/geometry_analysis.md for full specification.

Analyses (all at layer 22 as primary; cross-layer plots use all 34 layers):
  1. Norms per emotion per layer
  2. Pairwise cosine similarity — heatmap at layer 22 + line plot across layers
  3. Hierarchical clustering dendrogram at layer 22

Usage:
    python scripts/analyze_emotion_geometry.py \
        --hf-prefix steering_vectors/gemma3-4b-story

Output:
    figures/geometry/*.png
    results/geometry/geometry_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from huggingface_hub import hf_hub_download
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["joy", "sadness", "admiration", "loathing"]
PRIMARY_LAYER = 22
N_LAYERS = 34
DEFAULT_REPO_ID = "llm-psych/llm-psych-activations"

EMOTION_COLORS = {
    "joy": "#F4A261",
    "admiration": "#2A9D8F",
    "sadness": "#457B9D",
    "loathing": "#E76F51",
}

# Valence grouping for pair line-plot styling
WITHIN_VALENCE_PAIRS = {("joy", "admiration"), ("sadness", "loathing")}


# ---------------------------------------------------------------------------
# I/O helpers
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


def load_vectors(
    hf_prefix: str,
    repo_id: str,
    token: str | None,
    work_dir: Path,
) -> dict[str, dict[int, np.ndarray]]:
    """Download and load emotion vectors from HF.

    Returns
    -------
    dict {emotion: {layer_idx: np.ndarray[hidden_dim]}}
    """
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for emo in EMOTIONS:
        prefix = hf_prefix.strip("/")
        filename = f"{emo}_all_layers.pt"
        hf_path = f"{prefix}/{filename}" if prefix else filename

        print(f"  Downloading {hf_path} ...")
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hf_path,
            local_dir=str(work_dir),
            token=token,
        )
        layer_dict: dict[int, torch.Tensor] = torch.load(
            local, map_location="cpu", weights_only=True
        )
        vectors[emo] = {
            l: layer_dict[l].float().numpy() for l in range(N_LAYERS)
        }
        print(f"  Loaded {emo}: {len(vectors[emo])} layers, hidden_dim={next(iter(vectors[emo].values())).shape[0]}")
    return vectors


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute_norms(
    vectors: dict[str, dict[int, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for emo in EMOTIONS:
        for l in range(N_LAYERS):
            rows.append({
                "emotion": emo,
                "layer": l,
                "norm": float(np.linalg.norm(vectors[emo][l])),
            })
    return pd.DataFrame(rows)


def compute_cosine_similarities(
    vectors: dict[str, dict[int, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    pairs = list(combinations(EMOTIONS, 2))
    for l in range(N_LAYERS):
        for e1, e2 in pairs:
            v1 = vectors[e1][l]
            v2 = vectors[e2][l]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            rows.append({"e1": e1, "e2": e2, "layer": l, "cos_sim": cos})
    return pd.DataFrame(rows)


def cosine_matrix_at_layer(
    vectors: dict[str, dict[int, np.ndarray]], layer: int
) -> np.ndarray:
    """Return [n_emotions, n_emotions] cosine similarity matrix at one layer."""
    n = len(EMOTIONS)
    mat = np.zeros((n, n))
    for i, e1 in enumerate(EMOTIONS):
        for j, e2 in enumerate(EMOTIONS):
            v1, v2 = vectors[e1][layer], vectors[e2][layer]
            mat[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return mat


def compute_linkage(
    vectors: dict[str, dict[int, np.ndarray]], layer: int
) -> np.ndarray:
    """Agglomerative linkage with cosine distance at one layer."""
    vecs = np.stack([vectors[e][layer] for e in EMOTIONS])
    # Normalise to unit norm so cosine distance = euclidean distance on sphere
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = vecs / (norms + 1e-10)
    dist = pdist(vecs_normed, metric="cosine")
    return linkage(dist, method="average")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_norms(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        ax.plot(sub["layer"], sub["norm"], label=emo, color=EMOTION_COLORS[emo], linewidth=2)
    ax.axvline(PRIMARY_LAYER, color="gray", linestyle="--", linewidth=1, label=f"Layer {PRIMARY_LAYER} (2/3)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.set_title("Emotion vector magnitude across layers — Gemma 3 4B")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "norms_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved norms_by_layer.png")


def plot_cosine_heatmap(
    vectors: dict[str, dict[int, np.ndarray]], out_dir: Path
) -> None:
    mat = cosine_matrix_at_layer(vectors, PRIMARY_LAYER)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        linewidths=0.5,
        square=True,
    )
    ax.set_title(f"Pairwise cosine similarity — layer {PRIMARY_LAYER}")
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_sim_layer22_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved cosine_sim_layer22_heatmap.png")


def plot_cosine_by_layer(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for (e1, e2), sub in df.groupby(["e1", "e2"]):
        pair = (e1, e2)
        within = pair in WITHIN_VALENCE_PAIRS or (pair[1], pair[0]) in WITHIN_VALENCE_PAIRS
        linestyle = "-" if within else "--"
        alpha = 0.9 if within else 0.6
        label = f"{e1}–{e2}" + (" (same valence)" if within else "")
        ax.plot(sub["layer"], sub["cos_sim"], linestyle=linestyle, alpha=alpha, label=label, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(PRIMARY_LAYER, color="gray", linestyle="--", linewidth=1, label=f"Layer {PRIMARY_LAYER}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Pairwise emotion vector similarity across layers — Gemma 3 4B")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.set_ylim(-1, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_sim_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved cosine_sim_by_layer.png")


def plot_dendrogram(
    vectors: dict[str, dict[int, np.ndarray]], out_dir: Path
) -> None:
    Z = compute_linkage(vectors, PRIMARY_LAYER)
    fig, ax = plt.subplots(figsize=(5, 4))
    dendrogram(
        Z,
        labels=EMOTIONS,
        ax=ax,
        color_threshold=0,
        above_threshold_color="steelblue",
        leaf_font_size=12,
    )
    ax.set_title(f"Hierarchical clustering (cosine distance) — layer {PRIMARY_LAYER}")
    ax.set_ylabel("Distance")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram_layer22.png", dpi=150)
    plt.close(fig)
    print("  Saved dendrogram_layer22.png")


# ---------------------------------------------------------------------------
# Stats output
# ---------------------------------------------------------------------------

def save_stats(
    norm_df: pd.DataFrame,
    cos_df: pd.DataFrame,
    vectors: dict[str, dict[int, np.ndarray]],
    out_dir: Path,
) -> None:
    mat = cosine_matrix_at_layer(vectors, PRIMARY_LAYER)
    cos_at_22 = {
        f"{EMOTIONS[i]}_{EMOTIONS[j]}": round(float(mat[i, j]), 4)
        for i in range(len(EMOTIONS))
        for j in range(i + 1, len(EMOTIONS))
    }

    norms_at_22 = {
        emo: round(float(norm_df[(norm_df["emotion"] == emo) & (norm_df["layer"] == PRIMARY_LAYER)]["norm"].values[0]), 4)
        for emo in EMOTIONS
    }

    peak_norm_layers = {
        emo: int(norm_df[norm_df["emotion"] == emo]["norm"].idxmax())
        for emo in EMOTIONS
    }
    # idxmax returns index label, get layer value
    peak_norm_layers = {
        emo: int(norm_df[norm_df["emotion"] == emo].loc[
            norm_df[norm_df["emotion"] == emo]["norm"].idxmax(), "layer"
        ])
        for emo in EMOTIONS
    }

    stats = {
        "primary_layer": PRIMARY_LAYER,
        "n_layers": N_LAYERS,
        "emotions": EMOTIONS,
        "norms_at_layer22": norms_at_22,
        "peak_norm_layer": peak_norm_layers,
        "cosine_similarities_at_layer22": cos_at_22,
        "within_valence_mean_cos": round(float(np.mean([
            cos_at_22.get("joy_admiration", cos_at_22.get("admiration_joy", 0)),
            cos_at_22.get("sadness_loathing", cos_at_22.get("loathing_sadness", 0)),
        ])), 4),
        "cross_valence_mean_cos": round(float(np.mean([
            v for k, v in cos_at_22.items()
            if k not in {"joy_admiration", "sadness_loathing", "admiration_joy", "loathing_sadness"}
        ])), 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "geometry_stats.json"
    with path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved geometry_stats.json")

    # Print summary to stdout
    print("\n--- Geometry summary (layer 22) ---")
    print("Norms:", norms_at_22)
    print("Cosine similarities:")
    for k, v in cos_at_22.items():
        print(f"  {k}: {v:+.3f}")
    print(f"Within-valence mean cos: {stats['within_valence_mean_cos']:+.3f}")
    print(f"Cross-valence mean cos:  {stats['cross_valence_mean_cos']:+.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf-prefix",
        default="steering_vectors/gemma3-4b-story",
        help="Path prefix in HF dataset where emotion .pt files live.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).parent.parent / "steering_vectors" / "gemma3-4b-story",
        help="Local cache dir for downloaded .pt files.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).parent.parent / "figures" / "geometry",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "geometry",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = load_env_token()
    if not token:
        sys.exit("ERROR: HF_TOKEN not found in .env or environment.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading emotion vectors from HF ...")
    vectors = load_vectors(args.hf_prefix, args.repo_id, token, args.work_dir)

    print("\nComputing norms and cosine similarities ...")
    norm_df = compute_norms(vectors)
    cos_df = compute_cosine_similarities(vectors)

    print("\nPlotting ...")
    plot_norms(norm_df, args.figures_dir)
    plot_cosine_heatmap(vectors, args.figures_dir)
    plot_cosine_by_layer(cos_df, args.figures_dir)
    plot_dendrogram(vectors, args.figures_dir)

    save_stats(norm_df, cos_df, vectors, args.results_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
