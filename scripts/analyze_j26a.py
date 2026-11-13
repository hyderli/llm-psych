"""J26A analysis: forced-choice readout, judge-free.

Implements plans/j-space-decomposition.md "J26A" (2026-11-10), exactly:

* Non-compliance: a prompt cell is non-compliant if the five IDs' total mass
  is < 0.5; counted per condition and excluded everywhere.
* Per direction (condition): mean P(T), P(W), P(X) over compliant prompts.
* Positive control (required first): full vs unsteered on P(T), paired over
  prompts compliant in both; two-sided sign-flip by Monte Carlo, 100,000
  draws, seed 20261110; passes iff diff > 0 and p < 0.05.
* H26a pool > span on P(T); H26b coh > span on P(W). Direction-level exact
  two-sided permutation (analyze_j24.perm_test, 924 splits) + 95% bootstrap
  CI (10,000 within-class resamples, seed 20261110); Holm across the two;
  a hypothesis passes iff adjusted p < 0.05 AND diff > 0.
* Secondary: pool vs span on P(X), two-sided.
* Reading row per the spec's table, including the withheld and
  control-fails branches.

    uv run python scripts/analyze_j26a.py | tee results/j26a/analysis_stdout.txt
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "j26a"
READOUT = RES / "readout.jsonl"
PUSH_PATH = RES / "draft_push.json"
SEED = 20261110
MC_DRAWS = 100_000
MASS_MIN = 0.5

CLASSES = {"pool": [f"j24_pool{i}" for i in range(1, 7)],
           "span": [f"j24_span{i}" for i in range(1, 7)],
           "coh": [f"j25_coh{i}" for i in range(1, 7)]}
FULL, UNSTEERED = "j24_full", "unsteered"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


j24 = _load("analyze_j24", "scripts/analyze_j24.py")


def load(path: Path = READOUT) -> dict:
    """rows[(condition, order_index)] = row. Missing scores stay missing."""
    rows = {}
    for x in path.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            rows[(r["condition"], r["order_index"])] = r
    return rows


def compliant(rows: dict, cond: str) -> list[dict]:
    return [r for (c, _), r in rows.items()
            if c == cond and r["mass"] >= MASS_MIN]


def mean_p(rows: dict, cond: str, key: str) -> float:
    v = [r["p"][key] for r in compliant(rows, cond)]
    return float(np.mean(v)) if v else float("nan")


def class_means(rows: dict, cls: str, key: str) -> list[float]:
    return [mean_p(rows, c, key) for c in CLASSES[cls]]


def sign_flip_mc(diffs: list[float], seed: int = SEED,
                 draws: int = MC_DRAWS) -> float:
    """Two-sided sign-flip p by Monte Carlo: draws with |mean| >= observed."""
    d = np.asarray(diffs, dtype=float)
    if np.isnan(d).any() or len(d) == 0:
        raise ValueError("paired diffs unavailable")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, len(d)))
    hits = int((np.abs((d * signs).mean(axis=1)) >= obs - 1e-12).sum())
    return hits / draws


def _bootstrap_ci(a: list[float], b: list[float],
                  seed: int = SEED) -> tuple[float, float]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    d = rng.choice(x, size=(10_000, len(x))).mean(axis=1)
    d -= rng.choice(y, size=(10_000, len(y))).mean(axis=1)
    low, high = np.quantile(d, [0.025, 0.975], method="linear")
    return float(low), float(high)


def class_test(rows: dict, pos: str, neg: str, key: str) -> dict:
    a, b = class_means(rows, pos, key), class_means(rows, neg, key)
    if any(np.isnan(x) for x in a + b):
        raise ValueError(f"a direction of {pos}/{neg} has no compliant prompts")
    return {"diff": float(np.mean(a) - np.mean(b)),
            "p": j24.perm_test(a, b), "ci": _bootstrap_ci(a, b),
            "pos_rates": a, "neg_rates": b}


def reading_row(control_pass: bool, h26a_pass: bool, x_raised: bool,
                pool_push_t: float | None,
                pool_push_other_max: float | None) -> str:
    """The spec's readings table as a pure function."""
    if not control_pass:
        return "control fails: steering at this dose does not move a choice " \
               "made without generation; the task cannot separate the accounts"
    t_raised = h26a_pass
    if t_raised and x_raised:
        return "mixed: P(T) and P(X) both raised; use the draft-push numbers " \
               "to apportion, descriptively"
    if t_raised and not x_raised:
        if pool_push_t is not None and pool_push_other_max is not None \
                and pool_push_t <= pool_push_other_max:
            return "conceptual, not lexical: pool selects a threat with no " \
                   "hostile words and ignores hostile words with no threat"
        return "withheld: word matching on T's tokens is not excluded " \
               "(pool's draft push on T exceeds its pushes on E, C, W)"
    if x_raised and not t_raised:
        return "word matching: P(X) raised, P(T) not; the J24 pool effect " \
               "is lexical"
    return "lexical: v moves choices (control) but its hostile-word " \
           "directions do not; the J24 pool effect needs word output"


def main() -> int:
    rows = load()
    push = json.loads(PUSH_PATH.read_text())["arms"] if PUSH_PATH.exists() else {}

    print("# J26A analysis (forced choice, judge-free)")
    print(f"rows {len(rows)}")

    print("\n## Non-compliance per condition (mass < 0.5)")
    conds = sorted({c for c, _ in rows})
    for c in conds:
        n_all = sum(1 for (cc, _) in rows if cc == c)
        n_bad = n_all - len(compliant(rows, c))
        print(f"  {c:12s} {n_bad}/{n_all} non-compliant")

    print("\n## Per-direction means over compliant prompts")
    for c in conds:
        print(f"  {c:12s} P(T) {mean_p(rows, c, 'T'):.3f}  "
              f"P(W) {mean_p(rows, c, 'W'):.3f}  P(X) {mean_p(rows, c, 'X'):.3f}")

    print("\n## POSITIVE CONTROL: full vs unsteered on P(T), paired")
    diffs = []
    both = [o for (c, o) in rows if c == UNSTEERED]
    for oi in both:
        ru, rf = rows.get((UNSTEERED, oi)), rows.get((FULL, oi))
        if (ru is not None and rf is not None
                and ru["mass"] >= MASS_MIN and rf["mass"] >= MASS_MIN):
            diffs.append(rf["p"]["T"] - ru["p"]["T"])
    obs = float(np.mean(diffs)) if diffs else float("nan")
    p_ctrl = sign_flip_mc(diffs)
    ctrl_pass = bool(obs > 0 and p_ctrl < 0.05)
    print(f"paired over {len(diffs)} prompts: mean diff {obs:+.4f}, "
          f"MC sign-flip p = {p_ctrl:.4f} -> {'PASS' if ctrl_pass else 'FAIL'}")

    print("\n## PRIMARIES (direction-level exact permutation, Holm across two)")
    r_a = class_test(rows, "pool", "span", "T")
    r_b = class_test(rows, "coh", "span", "W")
    for name, r, pos, neg, key in (("H26a", r_a, "pool", "span", "T"),
                                   ("H26b", r_b, "coh", "span", "W")):
        print(f"  {name} {pos}>{neg} on P({key}): diff {r['diff']:+.4f} "
              f"CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p = {r['p']:.4f}")
    adj = j24.holm([r_a["p"], r_b["p"]])
    h26a = bool(adj[0] < 0.05 and r_a["diff"] > 0)
    h26b = bool(adj[1] < 0.05 and r_b["diff"] > 0)
    print(f"  Holm-adjusted: H26a {adj[0]:.4f} ({'PASS' if h26a else 'fail'}), "
          f"H26b {adj[1]:.4f} ({'PASS' if h26b else 'fail'})")

    print("\n## SECONDARY: pool vs span on P(X), two-sided")
    r_x = class_test(rows, "pool", "span", "X")
    x_raised = bool(r_x["p"] < 0.05 and r_x["diff"] > 0)
    print(f"  diff {r_x['diff']:+.4f} CI [{r_x['ci'][0]:+.4f},"
          f"{r_x['ci'][1]:+.4f}] p = {r_x['p']:.4f} -> P(X) "
          f"{'raised' if x_raised else 'not raised'}")

    pool_push = {k: float(np.mean([push[c][k] for c in CLASSES["pool"]]))
                 for k in ("T", "E", "C", "W")} if push else {}
    row = reading_row(ctrl_pass, h26a, x_raised,
                      pool_push.get("T"),
                      max((pool_push.get(k, float("nan"))
                           for k in ("E", "C", "W")), default=None)
                      if pool_push else None)
    print(f"\n## READING: {row}")
    print("\nPilot-scale; any claim needs n >= 200.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
