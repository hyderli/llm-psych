"""Measure what re-deriving would have done, so the frame choice can be quoted.

Add-on cells are derived into a base track's *frozen frame* rather than
re-centering the whole set (see ``scripts/export_derivation_frame.py``).
The alternative — recomputing the grand mean over base + add-on — would
have translated every vector by one shared offset

.. math::

    \\delta = \\frac{k\\,(\\bar\\mu_E - \\bar\\mu_{\\text{new}})}{E + k}

with ``E`` base emotions and ``k`` add-on emotions, then carried through
the (linear) neutral projection as ``P(delta)``. Pairwise differences are
unaffected either way; individual vectors are not.

This script computes that counterfactual offset from artefacts already on
disk and reports how big it is relative to the vectors themselves, so the
write-up can say "re-deriving would have moved every vector by X% of its
norm and Y degrees" instead of arguing the point.

Two measures, because they answer different questions:

* **norm ratio** ``||P(delta)|| / mean||v||`` — how large the shift is
  next to a typical vector. Relevant wherever magnitude matters.
* **angle** ``median angle(v, v + P(delta))`` in degrees — how much the
  *direction* moves. This is the one that matters for norm-matched
  steering arms, where magnitude is equalised anyway.

Inputs are read from conventional paths and nothing is written back into
any track:

``steering_vectors/<model>-<base-track>/frame.npz``   grand mean + neutral PCs
``activations/<model>-<addon-track>/<emotion>.npz``   add-on activations
``steering_vectors/<model>-<base-track>/*.npy``       base vectors (for scale)

Usage
-----
::

    uv run python scripts/measure_frame_shift.py

    uv run python scripts/measure_frame_shift.py \\
        --models Llama-3.1-8B-Instruct --csv results/frame_shift/shift.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from llm_psych.steering import project_out  # noqa: E402

_DEFAULT_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=_DEFAULT_MODELS)
    p.add_argument("--base-track", default="story-wheel32")
    p.add_argument("--addon-track", default="story-wheel32-addon")
    p.add_argument(
        "--csv",
        type=Path,
        default=_repo_root / "results" / "frame_shift" / "frame_shift.csv",
        help="Where to write the per-layer table.",
    )
    return p.parse_args()


def _layers_from_frame(frame: np.lib.npyio.NpzFile) -> list[int]:
    out = []
    for key in frame.files:
        m = re.fullmatch(r"grand_mean_layer_(\d+)", key)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _base_vectors_at_layer(vec_dir: Path, layer: int) -> np.ndarray:
    """Stack the base track's vectors at one layer -> (n_emotions, d)."""
    paths = sorted(vec_dir.glob(f"*_layer{layer}.npy"))
    if not paths:
        raise FileNotFoundError(f"no *_layer{layer}.npy in {vec_dir}")
    return np.stack([np.load(p).astype(np.float64) for p in paths], axis=0)


def _analyse_model(model: str, base_track: str, addon_track: str) -> list[dict]:
    base_vec_dir = _repo_root / "steering_vectors" / f"{model}-{base_track}"
    frame_path = base_vec_dir / "frame.npz"
    addon_act_dir = _repo_root / "activations" / f"{model}-{addon_track}"

    for path in (frame_path, addon_act_dir):
        if not path.exists():
            raise FileNotFoundError(f"missing: {path}")

    frame = np.load(frame_path)
    layers = _layers_from_frame(frame)

    addon_paths = sorted(p for p in addon_act_dir.glob("*.npz") if p.stem != "neutral")
    if not addon_paths:
        raise FileNotFoundError(f"no add-on activations in {addon_act_dir}")
    addon = {p.stem: np.load(p) for p in addon_paths}

    # E is the base track's emotion count, recorded when the frame was made.
    meta_path = base_vec_dir / "frame.yaml"
    n_base: int | None = None
    if meta_path.exists():
        import yaml

        n_base = int(yaml.safe_load(meta_path.read_text())["n_emotions"])
    if n_base is None:  # fall back to counting distinct emotions on disk
        n_base = len({p.name.rsplit("_layer", 1)[0] for p in base_vec_dir.glob("*.npy")})

    k = len(addon)
    rows: list[dict] = []

    for lyr in layers:
        key = f"layer_{lyr}"
        grand_mean = frame[f"grand_mean_{key}"].astype(np.float64)
        pcs = frame[f"pcs_{key}"]

        # Mean over the add-on emotions' per-emotion means, matching how
        # derive_story_vectors weights emotions equally.
        mu_new = np.mean(
            [addon[e][key].astype(np.float64).mean(axis=0) for e in addon], axis=0
        )

        delta_raw = k * (grand_mean - mu_new) / (n_base + k)
        delta = project_out(delta_raw.astype(np.float32)[None, :], pcs)[0].astype(np.float64)

        V = _base_vectors_at_layer(base_vec_dir, lyr)
        norms = np.linalg.norm(V, axis=1)
        shifted = V + delta
        cos = np.sum(V * shifted, axis=1) / (norms * np.linalg.norm(shifted, axis=1))
        angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        rows.append(
            {
                "model": model,
                "layer": lyr,
                "n_base": n_base,
                "k_addon": k,
                "delta_norm": float(np.linalg.norm(delta)),
                "mean_vector_norm": float(norms.mean()),
                "norm_ratio": float(np.linalg.norm(delta) / norms.mean()),
                "median_angle_deg": float(np.median(angles)),
                "max_angle_deg": float(angles.max()),
            }
        )
    return rows


def main() -> int:
    args = _parse_args()
    all_rows: list[dict] = []

    for model in args.models:
        try:
            rows = _analyse_model(model, args.base_track, args.addon_track)
        except FileNotFoundError as exc:
            print(f"[skip] {model}: {exc}")
            continue
        all_rows.extend(rows)

        ratios = np.array([r["norm_ratio"] for r in rows])
        angles = np.array([r["median_angle_deg"] for r in rows])
        print(f"\n=== {model} ===")
        print(
            f"  E={rows[0]['n_base']} base + k={rows[0]['k_addon']} add-on, "
            f"{len(rows)} layers"
        )
        print(
            f"  ||P(delta)|| / mean||v|| : median {np.median(ratios):.4%}  "
            f"range {ratios.min():.4%}-{ratios.max():.4%}"
        )
        print(
            f"  direction change          : median {np.median(angles):.3f} deg  "
            f"max {max(r['max_angle_deg'] for r in rows):.3f} deg"
        )

    if not all_rows:
        print("\nNothing analysed — are the frames and add-on activations pulled?")
        return 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(all_rows[0].keys())
    with args.csv.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in all_rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {args.csv.relative_to(_repo_root)} ({len(all_rows)} rows)")

    ratios = np.array([r["norm_ratio"] for r in all_rows])
    angles = np.array([r["median_angle_deg"] for r in all_rows])
    print(
        f"\nOverall: re-deriving would have moved every vector by "
        f"{np.median(ratios):.2%} of a typical vector's norm "
        f"(max {ratios.max():.2%}), rotating directions by a median of "
        f"{np.median(angles):.2f} deg (max {angles.max():.2f} deg)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
