"""PROVISIONAL descriptive spread of J-space and digit-projection numbers.

This is NOT gate G2. G2 asks whether admiration's digit contamination is an
outlier among validated cells, and no wheel32 cell is validated yet. This script
only draws the picture: how the numbers are distributed across all cells, how
many layers even have digit atoms, and where admiration and loathing fall.

Why run it before validation. If the digit-contamination spread turns out to be
narrow -- every cell within noise of every other -- then G2 cannot distinguish
admiration from anything regardless of which cells C2 validates, and that is
worth knowing before weeks of stimulus authoring rather than after.

PRE-REGISTRATION DISCIPLINE. Looking at this is safe only while no decision
depends on it. Before running, the following are already fixed and must not be
changed in response to what appears here:

  * which cells enter G2 (Tier A, resolved in plans/c2-revalidation-wheel32.md)
  * which layer each cell is read at (the D2 locked layer, per model)
  * the digit-atom criterion (argmax of the atom's unembedding is a numeral)
  * the pass condition for G2 (written in plans/h8-workspace-steering.md)

If any of those change after this runs, the change is a post-hoc choice and must be
recorded as such in the amendment log. Output is written with a PROVISIONAL
banner so it cannot be mistaken for a gate result later.

Normalisation. A random n-dimensional subspace of a d-dimensional space captures
n/d of a random vector's squared norm in expectation. So the null for
``v_fraction`` is ``n_digit_atoms / d_model``, and the interpretable quantity is
the enrichment ratio ``v_fraction / (n_digit_atoms / d_model)``: 1.0 means the
vector sits in the digit subspace no more than chance would put it there.
Raw v_fraction is not comparable across layers, because layers differ in how
many digit atoms they have.

Usage
-----
::

    uv run python scripts/jspace_descriptive_spread.py
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_repo_root = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"]
DEFAULT_TRACKS = ["story-wheel32", "story-wheel32-addon"]
ROOT_K64 = "results/workspace_decomposition_k64"
REPO_ID = "llm-psych/llm-psych-activations"

# Residual-stream width, for the chance-level normalisation above.
D_MODEL = {"Llama-3.1-8B-Instruct": 4096, "Qwen2.5-7B-Instruct": 3584,
           "gemma-2-9b-it": 3584}

MARKED = {"admiration", "loathing"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    p.add_argument("--csv", type=Path, default=_repo_root / "results" / "jspace_gate"
                   / "PROVISIONAL_descriptive_spread.csv")
    return p.parse_args()


def _load(track: str, model: str, token: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    path = f"{ROOT_K64}/{track}/{model}/manifest.yaml"
    try:
        local = hf_hub_download(REPO_ID, path, repo_type="dataset", token=token)
    except (EntryNotFoundError, OSError) as exc:
        print(f"[skip] {path}: {exc}")
        return None
    return yaml.safe_load(open(local))


def _rows(man: dict, model: str, track: str) -> list[dict]:
    out = []
    for emotion, layers in man["vectors"].items():
        for layer, entry in layers.items():
            m = entry.get("metrics_pos") or {}
            if "frac_norm_squared" not in m:
                continue
            dp = m.get("digit_projection") or {}
            out.append({
                "model": model, "track": track, "emotion": emotion,
                "layer": int(layer),
                "frac": float(m["frac_norm_squared"]),
                "n_atoms": int(m.get("n_atoms", 0)),
                "n_digit_atoms": int(dp["n_digit_atoms"]) if "n_digit_atoms" in dp else None,
                "v_fraction": float(dp["v_fraction"]) if "v_fraction" in dp else None,
            })
    return out


def _q(a) -> str:
    a = np.asarray(a, dtype=float)
    return (f"median {np.median(a):6.3f}   IQR {np.percentile(a,25):.3f}-"
            f"{np.percentile(a,75):.3f}   range {a.min():.3f}-{a.max():.3f}")


def main() -> int:
    args = _parse_args()
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found in .env")
        return 1

    rows: list[dict] = []
    for model in args.models:
        for track in args.tracks:
            man = _load(track, model, token)
            if man:
                rows.extend(_rows(man, model, track))
    if not rows:
        print("No manifests loaded.")
        return 1

    print("=" * 78)
    print("PROVISIONAL -- descriptive only. This is not gate G2 and must not be")
    print("cited as one. No wheel32 cell is validated yet.")
    print("=" * 78)

    for model in args.models:
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        d = D_MODEL[model]
        print(f"\n### {model}   (d_model = {d})")
        print(f"  vectors: {len(mrows)}   cells: {len({r['emotion'] for r in mrows})}"
              f"   layers: {sorted({r['layer'] for r in mrows})}")

        print(f"\n  J-fraction over all cells x layers:")
        print(f"    {_q([r['frac'] for r in mrows])}")

        have = [r for r in mrows if r["n_digit_atoms"] is not None]
        if not have:
            print("\n  no digit_projection fields in this manifest")
            continue

        # --- digit-atom availability, per layer -----------------------------
        per_layer = defaultdict(set)
        for r in have:
            per_layer[r["layer"]].add(r["n_digit_atoms"])
        print("\n  digit atoms available per layer (a layer property, "
              "constant across cells):")
        zero_layers = []
        for L in sorted(per_layer):
            vals = sorted(per_layer[L])
            n = vals[0] if len(vals) == 1 else vals
            if vals == [0]:
                zero_layers.append(L)
            print(f"    layer {L:>3}: n_digit_atoms = {n}")
        print(f"    layers with ZERO digit atoms: {zero_layers or 'none'}"
              f"   ({len(zero_layers)}/{len(per_layer)})")
        if zero_layers:
            print("    -> G2 is not computable at these layers at all. If a cell's")
            print("       locked layer is one of them, that cell cannot enter G2.")

        # --- contamination spread, zero-atom layers excluded ----------------
        live = [r for r in have if r["n_digit_atoms"] and r["v_fraction"] is not None]
        if not live:
            print("\n  every layer has zero digit atoms; nothing to rank")
            continue
        print(f"\n  raw v_fraction (zero-atom layers excluded, n={len(live)}):")
        print(f"    {_q([r['v_fraction'] for r in live])}")

        enrich = [r["v_fraction"] / (r["n_digit_atoms"] / d) for r in live]
        print(f"  enrichment over chance  v_fraction / (n_digit_atoms/d_model):")
        print(f"    {_q(enrich)}")
        print("    1.0 = the vector sits in the digit subspace no more than a")
        print("    random vector would. This is the comparable quantity.")

        # --- per-cell ranking ----------------------------------------------
        by_cell = defaultdict(list)
        for r in live:
            by_cell[r["emotion"]].append(r["v_fraction"] / (r["n_digit_atoms"] / d))
        ranked = sorted(((np.median(v), c) for c, v in by_cell.items()), reverse=True)
        print(f"\n  cells by median enrichment (high = more digit-aligned):")
        for i, (val, cell) in enumerate(ranked, 1):
            mark = "  <<<" if cell in MARKED else ""
            print(f"    {i:>3}. {cell:<16s} {val:7.2f}{mark}")
        spread = ranked[0][0] / ranked[-1][0] if ranked[-1][0] > 0 else float("inf")
        print(f"\n  top/bottom ratio: {spread:.2f}x")
        print("    A ratio near 1 means the cells are indistinguishable on this")
        print("    measure, and G2 cannot separate admiration from anything")
        print("    whatever C2 returns.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with args.csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
