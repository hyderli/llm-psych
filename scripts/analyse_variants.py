"""Pooled analysis of the prompt-variant sweep.

The unit that matters here is the VARIANT, not the sample. Fifty-four conditions
at n=5 are not fifty-four measurements; they are nine arms measured on six
prompts. Treating the 270 samples as independent would produce intervals that
ignore the very variance this sweep was run to expose.

Three things, in the order they should be read:

1. Baseline drift. The unsteered rate per item across the six variants. If it
   swings, every rate in the single-scenario figure was measured against one
   arbitrary point on that surface, and that figure needs a caveat.

2. Pooled rates per arm with a CLUSTER bootstrap -- resampling variants, not
   samples -- so the interval reflects prompt variance.

3. The contrast, tested the way clustered data allows: for each item, compute
   resid minus jspace WITHIN each variant, then count how many of the six have
   the same sign. Six of six is p = 0.031 two-sided by exact binomial. This is
   weaker than a pooled t-test and much harder to fool: it asks whether the
   effect reproduces across prompts, which is the question, rather than whether
   a pooled difference clears zero, which sampling noise alone can deliver.

4. The same contrasts, tested by stratified permutation. Section 3 as first
   written was underpowered BY CONSTRUCTION and should not be read alone: with
   six variants the exact-binomial floor is p = 0.031, and the 4 pairs x 7 items
   family is 28 tests, so no effect of any size could clear Holm. Section 4
   fixes that. It keeps variant as a stratum -- arm labels are permuted WITHIN
   each variant, never across -- so clustering is still respected, but the
   statistic uses the counts rather than only the sign of each difference, which
   is where the power went. Every arm is also tested against unsteered, because
   "does this arm do anything at all outside its home cell" turned out to be the
   question that separates the arms, and Section 3 never asked it. Holm is
   applied within each 7-item family.
"""
from __future__ import annotations

import csv, re, sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ITEMS = ["A", "B", "C", "D", "E", "G", "H"]
ARMS = ["unsteered", "full05", "full10", "resid05", "resid10",
        "jspace05", "jspace10", "randatom05", "randatom10"]
PAIRS = [("resid05", "jspace05"), ("resid10", "jspace10"),
         ("resid10", "full10"), ("jspace10", "randatom10")]
RE = re.compile(r"^(.+?)_L22_a[\d.]+_g(\w+?)_u(\w+)$")


def load(d: Path):
    cells = defaultdict(lambda: defaultdict(list))   # (arm, variant) -> item -> [0/1]
    for r in csv.DictReader((d / "scores_judge.csv").open()):
        if r["G0"] != "1":
            continue
        m = RE.match(r["condition"])
        if not m:
            continue
        arm, var = m.group(1), f"{m.group(2)}/{m.group(3)}"
        for it in ITEMS:
            if r[it] in ("0", "1"):
                cells[(arm, var)][it].append(int(r[it]))
    return cells


def rate(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def cluster_boot(per_variant, n_boot=20000, seed=20260927):
    """Resample VARIANTS with replacement; each contributes its own rate."""
    rng = np.random.default_rng(seed)
    v = np.array([x for x in per_variant if not np.isnan(x)])
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_p(k, n):
    """Two-sided exact binomial against p=0.5, for k of n agreeing."""
    k = max(k, n - k)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def strata(cells, variants, arm1, arm2, item):
    """Per-variant 2x2 counts: (k1, n1, k2, n2), dropping variants missing a cell."""
    out = []
    for v in variants:
        a = cells[(arm1, v)][item]
        b = cells[(arm2, v)][item]
        if a and b:
            out.append((sum(a), len(a), sum(b), len(b)))
    return out


def strat_perm(cells_2x2, n_perm=50000, seed=20260927):
    """Stratified permutation test on the Mantel-Haenszel deviation.

    Under the null, arm carries no information, so within each variant the
    successes are exchangeable between the two arms. Shuffling only WITHIN a
    stratum keeps each prompt's own difficulty fixed -- the whole point of
    stratifying -- while destroying the arm labelling. Returns (r1, r2, p);
    p is None when every stratum is degenerate (all 0s or all 1s), where no
    permutation can move the statistic and no test exists.
    """
    if not cells_2x2:
        return float("nan"), float("nan"), None
    t1 = sum(n1 for _, n1, _, _ in cells_2x2)
    t2 = sum(n2 for _, _, _, n2 in cells_2x2)
    r1 = sum(k1 for k1, _, _, _ in cells_2x2) / t1
    r2 = sum(k2 for _, _, k2, _ in cells_2x2) / t2
    live = [(k1, n1, k2, n2) for k1, n1, k2, n2 in cells_2x2
            if 0 < k1 + k2 < n1 + n2]
    if not live:
        return r1, r2, None
    obs = sum(k1 for k1, _, _, _ in cells_2x2)
    exp = sum((k1 + k2) * n1 / (n1 + n2) for k1, n1, k2, n2 in cells_2x2)
    d0 = abs(obs - exp)
    fixed = sum(k1 for k1, n1, k2, n2 in cells_2x2 if not 0 < k1 + k2 < n1 + n2)
    rng = np.random.default_rng(seed)
    pools = [(np.array([1] * (k1 + k2) + [0] * ((n1 + n2) - (k1 + k2))), n1)
             for k1, n1, k2, n2 in live]
    hits = 0
    for _ in range(n_perm):
        s = fixed
        for pool, n1 in pools:
            s += int(rng.permutation(pool)[:n1].sum())
        if abs(s - exp) >= d0 - 1e-9:
            hits += 1
    return r1, r2, (hits + 1) / (n_perm + 1)


def holm(ps):
    """Holm-Bonferroni, returning adjusted p in input order; None passes through."""
    live = [i for i, p in enumerate(ps) if p is not None]
    live.sort(key=lambda i: ps[i])
    out = [None] * len(ps)
    m, run = len(live), 0.0
    for rank, i in enumerate(live):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/scoring_variants")
    cells = load(d)
    variants = sorted({v for (_, v) in cells})
    print(f"{len(variants)} variants: " + ", ".join(variants))

    print("\n" + "=" * 72)
    print("1. BASELINE DRIFT -- unsteered rate per variant")
    print("=" * 72)
    print(f"{'item':6}" + "".join(f"{v.split('/')[0][:3]+'/'+v.split('/')[1][:4]:>12}"
                                  for v in variants) + f"{'range':>9}")
    for it in ITEMS:
        rs = [rate(cells[("unsteered", v)][it]) for v in variants]
        ok = [r for r in rs if not np.isnan(r)]
        print(f"{it:6}" + "".join(f"{r:>12.2f}" for r in rs)
              + f"{(max(ok)-min(ok)):>9.2f}")

    print("\n" + "=" * 72)
    print("2. POOLED RATES -- mean over variants, cluster-bootstrap 95% CI")
    print("=" * 72)
    print(f"{'arm':12}" + "".join(f"{it:>18}" for it in ITEMS))
    for arm in ARMS:
        row = f"{arm:12}"
        for it in ITEMS:
            per = [rate(cells[(arm, v)][it]) for v in variants]
            m = float(np.nanmean(per))
            lo, hi = cluster_boot(per)
            row += f"{f'{m:.2f} [{lo:.2f},{hi:.2f}]':>18}"
        print(row)

    print("\n" + "=" * 72)
    print("3. WITHIN-VARIANT CONTRASTS -- does the sign reproduce across prompts?")
    print("=" * 72)
    for a, b in PAIRS:
        print(f"\n  {a} minus {b}")
        print(f"    {'item':6}{'per-variant differences':>44}{'same sign':>11}{'p':>8}")
        for it in ITEMS:
            diffs = []
            for v in variants:
                ra, rb = rate(cells[(a, v)][it]), rate(cells[(b, v)][it])
                if not (np.isnan(ra) or np.isnan(rb)):
                    diffs.append(ra - rb)
            if len(diffs) < 3:
                continue
            nz = [x for x in diffs if abs(x) > 1e-9]
            if not nz:
                print(f"    {it:6}{'all zero':>44}{'-':>11}{'-':>8}")
                continue
            pos = sum(1 for x in nz if x > 0)
            k, n = max(pos, len(nz) - pos), len(nz)
            p = sign_p(pos, n)
            s = " ".join(f"{x:+.2f}" for x in diffs)
            print(f"    {it:6}{s:>44}{f'{k}/{n}':>11}{p:>8.3f}")
    print("\nSign tests use only non-zero differences; ties carry no information.")
    print("p is exact binomial against 0.5. With six variants the floor is 0.031.")
    print("Section 3 cannot clear Holm over 28 tests at any effect size -- read 4.")

    print("\n" + "=" * 72)
    print("4. STRATIFIED PERMUTATION -- variant as stratum, Holm within each family")
    print("=" * 72)
    fams = ([(a, "unsteered") for a in ARMS if a != "unsteered"]
            + list(PAIRS))
    for a, b in fams:
        res = [(it,) + strat_perm(strata(cells, variants, a, b, it)) for it in ITEMS]
        adj = holm([p for _, _, _, p in res])
        print(f"\n  {a} vs {b}")
        print(f"    {'item':6}{'rate ' + a:>16}{'rate ' + b:>16}"
              f"{'p':>9}{'p_holm':>9}")
        for (it, r1, r2, p), h in zip(res, adj):
            if p is None:
                print(f"    {it:6}{r1:>16.2f}{r2:>16.2f}{'-':>9}{'-':>9}"
                      "   no variation")
                continue
            mark = "  *" if h < 0.05 else ""
            print(f"    {it:6}{r1:>16.2f}{r2:>16.2f}{p:>9.3f}{h:>9.3f}{mark}")
    print("\nArm labels are permuted within a variant, never across it, so each")
    print("prompt's own difficulty is held fixed. '-' means every stratum was")
    print("degenerate (all 0s or all 1s): no permutation moves the statistic, so")
    print("there is no test -- not a null. * is Holm-adjusted p < 0.05.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
