"""G1 readout: is the J-space fraction stable across the sparse-pursuit budget?

Gate G1 in ``plans/h8-workspace-steering.md`` asks whether an emotion vector's
J-space fraction survives a change in the pursuit budget. The concern is
concrete: gradient pursuit will fit *any* target given enough atoms, so if
``frac_norm_squared`` moves a lot between k=16 and k=64, the number describes
how much room the pursuit was given rather than a property of the vector --
and every downstream use of it (H8's arm construction, the C2 correlation)
rests on sand.

This reads ``frac_norm_squared`` only. It touches no C2 metric, no digit
projection and no locked layers, so it can run before any of those exist.

What "stable" means here, and why three numbers rather than one:

* **median relative change** -- the headline. |f64 - f16| / f16, per vector.
  A vector whose fraction moves 5% under a 4x budget increase is stable in
  the sense the gate cares about.
* **Spearman rho** -- whether the *ordering* of cells survives. Even if every
  fraction inflates, a rank-preserving inflation still supports statements
  like "loathing carries more J-space content than joy".
* **sign of the change** -- pursuit with a larger dictionary can only fit at
  least as well, so f64 >= f16 is expected. Vectors where f64 < f16 indicate
  a non-monotone solver and would be a red flag about the pursuit itself.

Usage
-----
::

    uv run python scripts/jspace_gate_readout.py
    uv run python scripts/jspace_gate_readout.py --models Llama-3.1-8B-Instruct
    uv run python scripts/jspace_gate_readout.py --emotions loathing sadness
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_repo_root = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"]
DEFAULT_TRACKS = ["story-wheel32", "story-wheel32-addon"]
ROOTS = {16: "results/workspace_decomposition", 64: "results/workspace_decomposition_k64"}
REPO_ID = "llm-psych/llm-psych-activations"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    p.add_argument("--emotions", nargs="+", default=None,
                   help="Restrict the per-cell table to these emotions.")
    p.add_argument("--csv", type=Path,
                   default=_repo_root / "results" / "jspace_gate" / "g1_budget_stability.csv")
    return p.parse_args()


def _load_manifest(track: str, model: str, k: int, token: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    path = f"{ROOTS[k]}/{track}/{model}/manifest.yaml"
    try:
        local = hf_hub_download(REPO_ID, path, repo_type="dataset", token=token)
    except (EntryNotFoundError, OSError) as exc:
        print(f"[skip] {path}: {exc}")
        return None
    return yaml.safe_load(open(local))


def _rows(manifest: dict, model: str, track: str, k: int) -> list[dict]:
    out = []
    for emotion, layers in manifest["vectors"].items():
        for layer, entry in layers.items():
            for sign, key in (("+", "metrics_pos"), ("-", "metrics_neg")):
                m = entry.get(key)
                if not m or "frac_norm_squared" not in m:
                    continue
                out.append({
                    "model": model, "track": track, "k": k, "emotion": emotion,
                    "layer": int(layer), "sign": sign,
                    "frac": float(m["frac_norm_squared"]),
                    "n_atoms": int(m.get("n_atoms", 0)),
                })
    return out


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
            for k in (16, 64):
                man = _load_manifest(track, model, k, token)
                if man is not None:
                    rows.extend(_rows(man, model, track, k))
    if not rows:
        print("No manifests loaded.")
        return 1

    # Join the two budgets on (model, track, emotion, layer, sign).
    by_key: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_key[(r["model"], r["track"], r["emotion"], r["layer"], r["sign"])][r["k"]] = r

    paired = []
    for key, d in by_key.items():
        if 16 in d and 64 in d:
            f16, f64 = d[16]["frac"], d[64]["frac"]
            paired.append({
                "model": key[0], "track": key[1], "emotion": key[2],
                "layer": key[3], "sign": key[4],
                "frac_k16": f16, "frac_k64": f64,
                "abs_delta": f64 - f16,
                "rel_delta": (f64 - f16) / f16 if f16 > 0 else float("nan"),
                "atoms_k16": d[16]["n_atoms"], "atoms_k64": d[64]["n_atoms"],
            })
    if not paired:
        print("No k=16/k=64 pairs found -- are both roots on the dataset?")
        return 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(paired[0].keys())
    with args.csv.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in paired:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    print(f"\n{'=' * 78}\nG1 -- J-space fraction stability, k=16 vs k=64\n{'=' * 78}")
    print(f"{len(paired)} paired vectors (emotion x layer x sign)\n")

    hdr = f"{'model':<24} {'track':<22} {'n':>5} {'med f16':>8} {'med f64':>8} {'med rel':>9} {'rho':>7} {'f64<f16':>8}"
    print(hdr)
    print("-" * len(hdr))
    for model in args.models:
        for track in args.tracks:
            sub = [r for r in paired if r["model"] == model and r["track"] == track]
            if not sub:
                continue
            f16 = np.array([r["frac_k16"] for r in sub])
            f64 = np.array([r["frac_k64"] for r in sub])
            rel = np.array([r["rel_delta"] for r in sub])
            rho = spearmanr(f16, f64).statistic if spearmanr else float("nan")
            n_dec = int((f64 < f16).sum())
            print(f"{model:<24} {track:<22} {len(sub):>5} {np.median(f16):>8.4f} "
                  f"{np.median(f64):>8.4f} {np.median(rel):>8.1%} {rho:>7.3f} {n_dec:>8}")

    focus = args.emotions or ["loathing", "sadness", "admiration", "joy",
                             "desperation", "calm", "nervousness"]
    print(f"\n{'-' * 78}\nPer-cell (+v only), gate-relevant emotions\n{'-' * 78}")
    hdr2 = f"{'model':<24} {'emotion':<14} {'layers':>7} {'med f16':>8} {'med f64':>8} {'med rel':>9} {'max rel':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    for model in args.models:
        for emo in focus:
            sub = [r for r in paired if r["model"] == model and r["emotion"] == emo and r["sign"] == "+"]
            if not sub:
                continue
            f16 = np.array([r["frac_k16"] for r in sub])
            f64 = np.array([r["frac_k64"] for r in sub])
            rel = np.array([r["rel_delta"] for r in sub])
            print(f"{model:<24} {emo:<14} {len(sub):>7} {np.median(f16):>8.4f} "
                  f"{np.median(f64):>8.4f} {np.median(rel):>8.1%} {np.max(np.abs(rel)):>8.1%}")

    print(f"\nWrote {args.csv.relative_to(_repo_root)}")
    print("\nReading the table: 'med rel' is the median relative change in the")
    print("J-space fraction under a 4x budget increase. 'rho' is the Spearman")
    print("correlation between the two budgets' orderings -- high rho with a")
    print("large med rel means the fractions inflate but the ranking of cells")
    print("survives, which is enough for comparative claims but not absolute")
    print("ones. 'f64<f16' should be 0; anything else means the pursuit is")
    print("not monotone in the budget and needs looking at.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
