"""J30 n=200 confirmatory analysis (post-hoc note to the J30 results).

Runs ONCE after results/j30b/{payloads,judge_scores}.jsonl exist (item B).
Tests, fixed before generation:

  H30a-200: B(full+d) < pooled random ablations AND < each of r1/r2/r3
            (the registered pooled test; per-draw descriptives).
  Fraction vs no-ablation: (B_full - B_full+d) / (B_full - B_none).
  Fraction vs matched ablation: (B_rand - B_full+d) / (B_rand - B_none),
            B_rand = pooled random mean.
  Both fractions with 95% bootstrap CI (10,000 resamples, seed 20261116).
  Degeneration rates reported; non-degenerate sensitivity for the tests.

    uv run python scripts/analyze_j30b.py | tee results/j30b/analysis_stdout.txt
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "j30b"


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


j25 = _load_module("analyze_j25", "scripts/analyze_j25.py")

C = {n: f"j30b_c{n:02d}_{s}" for n, s in
     [(1, "none"), (2, "full"), (3, "full_d"),
      (4, "full_r1"), (5, "full_r2"), (6, "full_r3")]}
RANDS = [C[4], C[5], C[6]]
SEED = 20261116


def boot_frac(k1, n1, k2, n2, k3, n3, rng) -> tuple[float, float]:
    """CI of (p1-p2)/(p1-p3) under binomial resampling of each arm."""
    vals = []
    for _ in range(10_000):
        b1 = rng.binomial(n1, k1 / n1) / n1
        b2 = rng.binomial(n2, k2 / n2) / n2
        b3 = rng.binomial(n3, k3 / n3) / n3
        if b1 != b3:
            vals.append((b1 - b2) / (b1 - b3))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def block(pay, sc, degen, exclude, rng, label: str) -> dict:
    def rr(c):
        return j25.rate(pay, sc, "B", c, exclude)
    r = {k: rr(v) for k, v in C.items()}
    ku, nu = r[1]
    kf, nf = r[2]
    kd, nd = r[3]
    rk = sum(v[0] for k, v in r.items() if k in (4, 5, 6))
    rn = sum(v[1] for k, v in r.items() if k in (4, 5, 6))
    print(f"\n## B rates{label}")
    for n in (1, 2, 3, 4, 5, 6):
        k, m = r[n]
        print(f"  {C[n]:20s} {k:3d}/{m:3d}  ({k / m if m else float('nan'):.3f})")
    p_pool = j25.fisher((kd, nd), (rk, rn))
    per_draw = [j25.fisher((kd, nd), r[i]) for i in (4, 5, 6)]
    h30a = p_pool < 0.05 and all(p < 0.05 for p in per_draw) and kd / nd < rk / rn
    print(f"  H30a-200 pooled p={p_pool:.4g}  per-draw p={per_draw} -> "
          f"{'PASS' if h30a else 'fail'}")
    f_no = (kf / nf - kd / nd) / (kf / nf - ku / nu)
    c1 = boot_frac(kf, nf, kd, nd, ku, nu, rng)
    f_m = (rk / rn - kd / nd) / (rk / rn - ku / nu)
    c2 = boot_frac(rk, rn, kd, nd, ku, nu, rng)
    print(f"  fraction vs no-ablation   = {f_no:.3f}  95% CI [{c1[0]:.3f}, {c1[1]:.3f}]")
    print(f"  fraction vs matched abl.  = {f_m:.3f}  95% CI [{c2[0]:.3f}, {c2[1]:.3f}]")
    return {"h30a": h30a, "f_no": (f_no, c1), "f_m": (f_m, c2)}


def main() -> int:
    pay, sc = j25.load(RES)
    degen, rep_ok = j25.degenerate_flags(pay)
    present = {d["condition"] for d in pay.values()}
    missing = [c for c in C.values() if c not in present]
    if missing:
        raise SystemExit(f"conditions missing: {missing}")
    rng = np.random.default_rng(SEED)
    print("# J30 n=200 confirmatory (post-hoc note, 2026-11-16)")
    print(f"samples {len(pay)}; repetition criterion "
          f"{'validated' if rep_ok else 'NOT validated'}")
    for c in C.values():
        ss = [s for s, d in pay.items() if d["condition"] == c]
        dg = sum(degen[s] for s in ss)
        print(f"  {c:20s} gate {sum(not pay[s]['gate_flags'] for s in ss)}/{len(ss)}  degen {dg}")
    block(pay, sc, degen, None, rng, "")
    block(pay, sc, degen, degen, rng, " (non-degenerate sensitivity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
