#!/usr/bin/env python3
"""Correlate per-cell J-space fraction with C2 validation metrics.

Reads the decomposed manifests produced by scripts/decompose_emotion_vectors.py
and the C2 sweep tables from results/vector_validation/, then reports Pearson
and Spearman correlations between J-space loading and validation outcomes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_c2_layers import parse_sweep_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(decomp_dir: Path, model_key: str) -> dict:
    path = decomp_dir / f"{model_key}-story" / "manifest.yaml"
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate J-space fractions with C2 validation metrics."
    )
    parser.add_argument(
        "--decomp-dir",
        type=Path,
        default=REPO_ROOT / "results" / "workspace_decomposition_k64",
        help="Directory containing <model_key>-story/manifest.yaml files.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=REPO_ROOT / "results" / "vector_validation",
        help="Directory containing per-model sweep tables.",
    )
    args = parser.parse_args()

    rows = []
    for model_dir in sorted(args.validation_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        implicit_path = model_dir / "implicit_scenarios_sweep.md"
        intensity_path = model_dir / "intensity_semantic_sweep.md"
        if not implicit_path.exists() or not intensity_path.exists():
            print(f"Skipping {model_key}: missing sweep tables")
            continue

        implicit = parse_sweep_table(implicit_path)
        intensity = parse_sweep_table(intensity_path)

        manifest = load_manifest(args.decomp_dir, model_key)
        for emotion, layers in manifest.get("vectors", {}).items():
            if emotion not in implicit:
                continue
            for layer_s, info in layers.items():
                layer = int(layer_s)
                m = info["metrics_pos"]
                rows.append(
                    {
                        "model": model_key,
                        "emotion": emotion,
                        "layer": layer,
                        "jspace_frac": m["frac_norm_squared"],
                        "digit_v_frac": m.get("digit_projection", {}).get(
                            "v_fraction", 0.0
                        ),
                        "implicit_acc": implicit.get(emotion, {}).get(layer),
                        "intensity_rho": intensity.get(emotion, {}).get(layer),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No cells to correlate.")
        return

    print(f"\nCells: {len(df)} across {df['model'].nunique()} models, "
          f"{df['emotion'].nunique()} emotions")
    print("\n=== Per-model emotion mean J-space fraction vs validation ===")
    print(
        df.groupby(["model", "emotion"])
        .agg({"jspace_frac": "mean", "implicit_acc": "mean", "intensity_rho": "mean"})
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    for metric in ("implicit_acc", "intensity_rho"):
        sub = df.dropna(subset=[metric])
        if sub.empty:
            continue
        pearson = stats.pearsonr(sub["jspace_frac"], sub[metric])
        spearman = stats.spearmanr(sub["jspace_frac"], sub[metric])
        print(
            f"\n{metric}: n={len(sub)}, "
            f"Pearson r={pearson.statistic:.3f} p={pearson.pvalue:.4g}, "
            f"Spearman rho={spearman.statistic:.3f} p={spearman.pvalue:.4g}"
        )

    out_csv = args.decomp_dir / "jspace_c2_correlation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
