"""Score the filled C2 wheel32 rating sheet.

Reads data/derived/rating/rating_sheet.csv (filled) against rating_key.csv and
reports the four things the rating pass exists to establish:

1. **Monotonicity per arm.** The one failure mode the complement-pair design
   cannot absorb. An arm whose ratings do not move monotonically with x will
   depress |rho| in a way indistinguishable from a weak probe.
2. **Attenuation ceiling per arm** -- |Spearman(rating, x)|. A probe scoring
   0.6 on an arm whose ceiling is 0.65 is a different finding from one scoring
   0.6 against a ceiling of 0.95, and the C2 gate's 0.60 threshold is otherwise
   a number against nothing.
3. **Neutral flatness.** Neutrals should show near-zero |rho| on both scales.
   A neutral that moves with x is carrying construct and the ratio diagnostic
   in A4 will misread the residual.
4. **Test-retest agreement** on the duplicated items -- the only reliability
   estimate available from a single rater.

Reading the ceiling from a single rater who authored the items: a LOW ceiling
is strong evidence against that arm, since the author could not recover their
own intended ordering from the shuffled sentence. A HIGH ceiling is weak
evidence, since the author knows what each item was built to say. The asymmetry
is why this is worth running anyway, and why it is reported as an author's
monotonicity check rather than as a human ceiling.

Run:  uv run python scripts/score_rating_sheet.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_repo_root = Path(__file__).resolve().parents[1]
RATING_DIR = _repo_root / "data" / "derived" / "rating"
SHEET = RATING_DIR / "rating_sheet.csv"
KEY = RATING_DIR / "rating_key.csv"

TARGET_SCALE = {"desperation": "desperate", "nervousness": "nervous"}


def _read_sheet() -> dict[str, dict[str, float]]:
    rows = {}
    with SHEET.open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        if not r["desperate"].strip() or not r["nervous"].strip():
            raise SystemExit(f"item {r['item_id']} is not rated; fill every row")
        rows[r["item_id"]] = {"desperate": float(r["desperate"]),
                              "nervous": float(r["nervous"])}
    return rows


def main() -> None:
    ratings = _read_sheet()
    with KEY.open() as f:
        key = list(csv.DictReader(f))

    # --- test-retest on duplicated sentences -------------------------------
    by_item: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for k in key:
        by_item[(k["family"], int(k["x"]))].append(ratings[k["item_id"]])
    diffs = [abs(v[0][s] - v[1][s]) for v in by_item.values() if len(v) > 1
             for s in ("desperate", "nervous")]
    print("=" * 78)
    print("TEST-RETEST (duplicated sentences, both scales)")
    if diffs:
        print(f"  n={len(diffs)//2} sentences   mean |difference| = {np.mean(diffs):.2f} "
              f"   max = {max(diffs):.0f}")
        print("  A mean above ~1.5 points means the scale is not being applied stably;")
        print("  ceilings below are then upper bounds with meaningful noise on them.")
    else:
        print("  no duplicates found")

    # --- per-arm monotonicity and ceiling ----------------------------------
    arms: dict[tuple[str, str, str, str], list[tuple[int, dict]]] = defaultdict(list)
    for k in key:
        if int(k["is_duplicate"]):
            continue
        arms[(k["emotion"], k["scenario"], k["family"], k["direction"])].append(
            (int(k["x"]), ratings[k["item_id"]])
        )

    print()
    print("=" * 78)
    print("PER-ARM MONOTONICITY AND CEILING")
    print(f"{'family':22s} {'dir':>10s} {'ceiling':>8s} {'off-scale':>10s}  monotone")
    problems = []
    for (emotion, scenario, family, direction), pairs in sorted(arms.items()):
        pairs.sort()
        xs = [x for x, _ in pairs]
        if emotion == "neutral":
            continue
        scale = TARGET_SCALE[emotion]
        other = "nervous" if scale == "desperate" else "desperate"
        vals = [r[scale] for _, r in pairs]
        off = [r[other] for _, r in pairs]

        rho = float(spearmanr(vals, xs).statistic)
        want = -1 if direction == "decreasing" else +1
        # monotone in the intended direction, ties allowed
        deltas = np.diff(vals)
        ok = all(d <= 0 for d in deltas) if want < 0 else all(d >= 0 for d in deltas)
        flag = "yes" if ok else "NO"
        if not ok or abs(rho) < 0.6 or np.sign(rho) != want:
            problems.append((family, flag, round(rho, 2)))
        print(f"{family:22s} {direction:>10s} {abs(rho):8.2f} {np.mean(off):10.1f}"
              f"  {flag}")

    print()
    print("  ceiling   = |Spearman(target rating, x)|; the most a probe could score here")
    print("  off-scale = mean rating on the OTHER construct; should sit well below target")

    # --- neutral flatness ---------------------------------------------------
    print()
    print("=" * 78)
    print("NEUTRAL FLATNESS (want both near zero)")
    print(f"{'family':22s} {'|rho| desperate':>16s} {'|rho| nervous':>15s}")
    for (emotion, scenario, family, direction), pairs in sorted(arms.items()):
        if emotion != "neutral":
            continue
        pairs.sort()
        xs = [x for x, _ in pairs]
        rd = abs(float(spearmanr([r["desperate"] for _, r in pairs], xs).statistic))
        rn = abs(float(spearmanr([r["nervous"] for _, r in pairs], xs).statistic))
        print(f"{family:22s} {rd:16.2f} {rn:15.2f}")
        if max(rd, rn) > 0.5:
            problems.append((family, "neutral moves with x", round(max(rd, rn), 2)))

    # --- verdict ------------------------------------------------------------
    print()
    print("=" * 78)
    if problems:
        print("ARMS NEEDING ATTENTION BEFORE THE SET IS FROZEN")
        for p in problems:
            print(f"  {p}")
        print()
        print("A non-monotone arm must be reworded or dropped: it depresses |rho|")
        print("in a way that is indistinguishable from a weak probe, which is the")
        print("one failure the complement-pair design cannot absorb.")
    else:
        print("All arms monotone in the intended direction, all neutrals flat.")
    print("=" * 78)


if __name__ == "__main__":
    main()
