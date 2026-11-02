#!/usr/bin/env python3
"""EXPLORATORY follow-up to the J19 pre-registered analysis. NOT pre-registered.

Nothing here may be read as a test. It does three jobs:
  1. bounds the gate-attrition selection instead of hand-waving it, by scoring
     the two extreme imputations of the missing samples;
  2. breaks the plane D rates out by dose, to look at the 80 deg pin cell;
  3. lays the four attrition arms side by side on A, B, D.
Run scripts/analyze_j19.py for the pre-registered numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recut_outcomes import fisher
from analyze_j19 import (ALL_DEG, ATTR, DOSES, PLANE, pct, pool, rows_for, wilson)


def cell(rows, arm, item, dose=None, deg=None):
    """(k, n, n_fail) -- coherent scored, and how many samples the gate removed."""
    sub = [r for r in rows if r["arm"] == arm
           and (dose is None or r["dose"] == dose)
           and (deg is None or r["deg"] == deg)]
    sc = [r[item] for r in sub if r["gate"] and r[item] is not None]
    fail = sum(1 for r in sub if not r["gate"])
    miss = sum(1 for r in sub if r["gate"] and r[item] is None)
    return sum(sc), len(sc), fail + miss


def bounds(rows, arm_hi, arm_lo, item, claim):
    """Fisher p under: as-specified, ITT (all missing = 0), and the ADVERSARIAL
    imputation -- every missing sample filled against `claim`."""
    kh, nh, fh = cell(rows, arm_hi, item)
    kl, nl, fl = cell(rows, arm_lo, item)
    print(f"\n  {item}: {arm_hi} vs {arm_lo}   (claim: {claim})")
    print(f"    as-specified  {arm_hi} {pct(kh, nh)}  {arm_lo} {pct(kl, nl)}"
          f"   p={fisher(kh, nh-kh, kl, nl-kl):.4f}")
    print(f"    ITT (all 0)   {arm_hi} {pct(kh, nh+fh)}  {arm_lo} {pct(kl, nl+fl)}"
          f"   p={fisher(kh, nh+fh-kh, kl, nl+fl-kl):.4f}")
    if claim == "higher":        # worst case: hi loses all, lo gains all
        a, na, b, nb = kh, nh + fh, kl + fl, nl + fl
    else:                        # claim "lower": hi gains all, lo loses all
        a, na, b, nb = kh + fh, nh + fh, kl, nl + fl
    print(f"    ADVERSARIAL   {arm_hi} {pct(a, na)}  {arm_lo} {pct(b, nb)}"
          f"   p={fisher(a, na-a, b, nb-b):.4f}")
    gap_as = kh/nh - kl/nl if nh and nl else float('nan')
    gap_ad = a/na - b/nb if na and nb else float('nan')
    kept = (gap_as > 0) == (gap_ad > 0)
    print(f"    gap {gap_as:+.3f} -> {gap_ad:+.3f} under the worst case: "
          f"direction {'SURVIVES' if kept else 'REVERSES'}")


def main() -> int:
    rows = rows_for(PLANE, "plane")
    print("=" * 72)
    print("EXPLORATORY FOLLOW-UP -- not pre-registered, not a test")
    print("=" * 72)

    print("\n--- plane D pooled over ALL 7 angles (for the record) ---")
    for itt in (False, True):
        kp, np_ = pool(rows, "D", ALL_DEG, "pin", itt=itt)
        ko, no = pool(rows, "D", ALL_DEG, "pout", itt=itt)
        print(f"  {'ITT        ' if itt else 'as-specified'}  pin {pct(kp, np_)}"
              f"   pout {pct(ko, no)}   p={fisher(kp, np_-kp, ko, no-ko):.4f}")

    print("\n--- plane D by angle x dose (the 80 deg pin cell) ---")
    print(f"{'angle':>6}  {'pin a0.10':>13}  {'pin a0.15':>13}  "
          f"{'pout a0.10':>13}  {'pout a0.15':>13}")
    for deg in ALL_DEG:
        out = f"{deg:>6}"
        for arm in ("pin", "pout"):
            for dose in DOSES:
                k, n, _ = cell(rows, arm, "D", dose=dose, deg=deg)
                out += f"  {pct(k, n):>13}"
        print(out)

    print("\n--- plane A, B by angle (pin / pout), as-specified ---")
    print(f"{'angle':>6}  {'A pin':>13}  {'A pout':>13}  {'B pin':>13}  {'B pout':>13}")
    for deg in ALL_DEG:
        out = f"{deg:>6}"
        for item in ("A", "B"):
            for arm in ("pin", "pout"):
                k, n, _ = cell(rows, arm, item, deg=deg)
                out += f"  {pct(k, n):>13}"
        print(out)

    arows = rows_for(ATTR, "unit")
    print("\n" + "=" * 72)
    print("ATTRITION ARMS -- four arms side by side, with Wilson 95%")
    print("=" * 72)
    print(f"{'arm':<10} {'gate fail':>12}  {'A':>22}  {'B':>22}  {'D':>22}")
    for arm in ("resid", "jspace", "randatom", "faratom"):
        sub = [r for r in arows if r["arm"] == arm]
        f = sum(1 for r in sub if not r["gate"])
        line = f"{arm:<10} {pct(f, len(sub)):>12}"
        for item in ("A", "B", "D"):
            k, n, _ = cell(arows, arm, item)
            lo, hi = wilson(k, n)
            line += f"  {k:>2}/{n:<2} {k/n if n else float('nan'):.3f} [{lo:.2f},{hi:.2f}]"
        print(line)

    print("\n--- selection bounds on the J18 legs ---")
    bounds(arows, "jspace", "resid", "A", "higher")
    bounds(arows, "jspace", "resid", "B", "lower")
    bounds(arows, "jspace", "resid", "D", "lower")

    print("\n--- refit arms pooled vs resid (exploratory; the refit axis) ---")
    for item in ("A", "B", "D"):
        kr = nr = 0
        for arm in ("jspace", "randatom", "faratom"):
            k, n, _ = cell(arows, arm, item)
            kr += k
            nr += n
        k0, n0, _ = cell(arows, "resid", item)
        print(f"  {item}: refits {pct(kr, nr)}   resid {pct(k0, n0)}"
              f"   p={fisher(kr, nr-kr, k0, n0-k0):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
