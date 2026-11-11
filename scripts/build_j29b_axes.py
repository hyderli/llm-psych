"""J29 amendment 2: the shared and difference axes of the pool and coh G parts.

Standard library only. Reads the built j29b_poolG{1..6} and j29b_cohG{1..6}
files (J29 amendment 1), and builds three directions:

    u   = unit(mean of unit poolG_i)      c = unit(mean of unit cohG_i)
    int = unit(u + c)     the shared ("intensity") axis
    valp = unit(u - c)    the difference ("valence") axis, toward pool
    valm = -valp          toward coh

int and valp are orthogonal exactly (|u| = |c|). All three are checked to be
perpendicular to v_hat (they are combinations of G parts that already are).
Each runs at the mean poolG alpha from the amendment-1 build report.

Usage::

    python3 scripts/build_j29b_axes.py            # report only
    python3 scripts/build_j29b_axes.py --write    # report + .npy files
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b", ROOT / "scripts" / "build_j29b_arms.py")
b = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(b)

REPORT = ROOT / "results" / "j29" / "build_j29b_axes_report.json"
TOL_V = 1e-6


def unit(x):
    return b.scale(x, 1 / b.norm(x))


def mean(vs):
    out = [0.0] * len(vs[0])
    for v in vs:
        out = b.add(out, v)
    return b.scale(out, 1 / len(vs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = json.loads(b.REPORT.read_text())
    full, _ = b.load_npy(b.VEC / "j24_full_layer16.npy")
    target = b.norm(full)
    vh = unit(full)
    fam = {f: [unit(b.load_npy(b.VEC / f"{b.TAG}_{f}G{i}_layer16.npy")[0]) for i in range(1, 7)]
           for f in ("pool", "coh")}
    u, c = unit(mean(fam["pool"])), unit(mean(fam["coh"]))
    axes = {"int": unit(b.add(u, c)), "valp": unit(b.sub(u, c))}
    axes["valm"] = b.scale(axes["valp"], -1.0)
    alpha = round(sum(base["arms"][f"{b.TAG}_poolG{i}"]["alpha"] for i in range(1, 7)) / 6, 4)

    rep = {"amendment": "J29 amendment 2", "cos_u_c": b.cos(u, c), "alpha": alpha,
           "cos_int_valp": b.cos(axes["int"], axes["valp"]), "arms": {}}
    for name, x in axes.items():
        cv = b.cos(x, vh)
        if abs(cv) > TOL_V:
            raise SystemExit(f"REFUSING {name}: cos with v_hat {cv:.2e}")
        rep["arms"][f"{b.TAG}_{name}"] = {
            "alpha": alpha, "cos_v": cv,
            "cos_poolG_mean": sum(b.cos(x, g) for g in fam["pool"]) / 6,
            "cos_cohG_mean": sum(b.cos(x, g) for g in fam["coh"]) / 6}
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if args.write:
        for name, x in axes.items():
            out = b.VEC / f"{b.TAG}_{name}_layer16.npy"
            if out.exists():
                raise SystemExit(f"REFUSING to overwrite {out}")
            b.save_npy(out, b.scale(x, target))
        print(f"3 arm files -> {b.VEC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
