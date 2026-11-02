#!/usr/bin/env python3
"""J19 pre-registered analysis: the orthogonal-plane run, plus the J18 replication.

Reads payloads.jsonl + judge_scores.jsonl directly. The four-item export is a
PARTIAL item set, and `combine` drops partial samples as incomplete, so the
combine path is unavailable by construction (this is what --from-judge exists
for in recut_outcomes.py; that one is D-only, this needs A and B too).

Every test here was pre-registered in plans/j-space-decomposition.md (J19
pre-registration, amendments 1-4b) BEFORE the judge ran. Nothing is added here.

A is the OR of its two surfaces. score_blackmail.py:714 computes that with
.get(), so a MISSING surface reads as False and silently derives A=0. Here a
missing surface with the other at 0 makes A None -- missing, not zero.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recut_outcomes import fisher              # exact two-sided Fisher
from analyse_variants import holm, strat_perm  # Holm-Bonferroni, stratified permutation

PLANE = Path("results/scoring_plane")
ATTR = Path("results/scoring_attrrep")

PLN_RE = re.compile(r"_(pin|pout)(\d+)_L\d+_a([0-9.]+)$")
UNIT_RE = re.compile(r"_pos_([a-z]+)_L\d+_a([0-9.]+)$")

REVIVAL_DEG = [70, 75, 80, 85]      # past the action cliff
SUPPRESS_DEG = [40, 50, 60, 70]     # on the falling limb
ALL_DEG = [40, 50, 60, 70, 75, 80, 85]
DOSES = [0.1, 0.15]


def load_dir(d: Path):
    pay = {}
    with (d / "payloads.jsonl").open() as f:
        for line in f:
            j = json.loads(line)
            pay[j["sid"]] = j
    sc: dict[str, dict] = defaultdict(dict)
    with (d / "judge_scores.jsonl").open() as f:
        for line in f:
            j = json.loads(line)
            if j.get("value") in (0, 1, "0", "1"):
                sc[j["sid"]][j["item"]] = int(j["value"])
    return pay, sc


def item_A(it: dict):
    """A = A_pad OR A_act, missing-aware: one surface absent and the other 0 is None."""
    pad, act = it.get("A_pad"), it.get("A_act")
    if pad == 1 or act == 1:
        return 1
    if pad is None or act is None:
        return None
    return 0


def rows_for(d: Path, kind: str):
    pay, sc = load_dir(d)
    out = []
    for sid, j in pay.items():
        if str(j.get("development", "")).lower() in ("1", "true"):
            continue
        m = (PLN_RE if kind == "plane" else UNIT_RE).search(j["condition"])
        if not m:
            continue
        if kind == "plane":
            arm, deg, dose = m.group(1), int(m.group(2)), float(m.group(3))
        else:
            arm, deg, dose = m.group(1), None, float(m.group(2))
        it = sc.get(sid, {})
        out.append({"sid": sid, "cond": j["condition"], "arm": arm, "deg": deg,
                    "dose": dose, "gate": int(j.get("G0_auto") == 1),
                    "flags": j.get("gate_flags") or [],
                    "A": item_A(it), "B": it.get("B"), "D": it.get("D")})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


def pool(rows, item, degs, arm, itt=False):
    """(k, n) for one arm over the given angles, both doses."""
    k = n = 0
    for r in rows:
        if r["arm"] != arm or r["deg"] not in degs:
            continue
        if itt:                       # gate failure counts as the behaviour absent
            v = r[item] if (r["gate"] and r[item] is not None) else 0
            n += 1
            k += v
        else:
            if not r["gate"] or r[item] is None:
                continue
            n += 1
            k += r[item]
    return k, n


def strata_of(rows, item, degs):
    """(k_pin, n_pin, k_pout, n_pout) per (deg, dose) stratum."""
    out = []
    for deg in degs:
        for dose in DOSES:
            cell = {}
            for arm in ("pin", "pout"):
                k = n = 0
                for r in rows:
                    if (r["arm"], r["deg"], r["dose"]) != (arm, deg, dose):
                        continue
                    if not r["gate"] or r[item] is None:
                        continue
                    n += 1
                    k += r[item]
                cell[arm] = (k, n)
            if cell["pin"][1] and cell["pout"][1]:
                out.append((cell["pin"][0], cell["pin"][1],
                            cell["pout"][0], cell["pout"][1]))
    return out


def pct(k, n):
    return f"{k}/{n} = {k/n:.3f}" if n else f"{k}/{n} = --"


def main() -> int:
    rows = rows_for(PLANE, "plane")
    print("=" * 72)
    print("J19 PLANE RUN -- in-span (pin) vs out-of-span (pout) perpendicular")
    print("=" * 72)
    print(f"samples: {len(rows)}  (expected 560 = 7 angles x 2 kinds x 2 doses x 20)")

    # ---------------------------------------------------------------- gate
    print("\n--- GATE ATTRITION (reported first; it is post-treatment) ---")
    print(f"{'angle':>6}  {'pin pass':>12}  {'pout pass':>12}")
    for deg in ALL_DEG:
        c = {}
        for arm in ("pin", "pout"):
            sub = [r for r in rows if r["arm"] == arm and r["deg"] == deg]
            c[arm] = (sum(r["gate"] for r in sub), len(sub))
        print(f"{deg:>6}  {pct(*c['pin']):>12}  {pct(*c['pout']):>12}")
    gp = [(sum(r["gate"] for r in rows if r["arm"] == a),
           len([r for r in rows if r["arm"] == a])) for a in ("pin", "pout")]
    fail = [(n - k, n) for k, n in gp]
    print(f"\npooled gate FAILURE: pin {pct(*fail[0])}   pout {pct(*fail[1])}")
    print(f"  Fisher p = {fisher(fail[0][0], fail[0][1]-fail[0][0], fail[1][0], fail[1][1]-fail[1][0]):.4f}")
    flags = defaultdict(int)
    for r in rows:
        if not r["gate"]:
            for f in (r["flags"] or ["<none recorded>"]):
                flags[f] += 1
    print("  failure modes: " + ", ".join(f"{k} x{v}" for k, v in
                                          sorted(flags.items(), key=lambda t: -t[1])))

    # -------------------------------------------------------------- per-angle D
    print("\n--- D BY ANGLE (as-specified; coherent samples only) ---")
    print(f"{'angle':>6}  {'pin':>14}  {'pout':>14}")
    for deg in ALL_DEG:
        a = pool(rows, "D", [deg], "pin")
        b = pool(rows, "D", [deg], "pout")
        print(f"{deg:>6}  {pct(*a):>14}  {pct(*b):>14}")

    # ------------------------------------------------------- the four tests
    results = []

    # T1 primary: D revival past the cliff, pooled, Fisher
    kp, np_ = pool(rows, "D", REVIVAL_DEG, "pin")
    ko, no = pool(rows, "D", REVIVAL_DEG, "pout")
    p1 = fisher(kp, np_ - kp, ko, no - ko)
    results.append(("T1 D-revival 70-85 (PRIMARY, Fisher)", p1,
                    f"pin {pct(kp, np_)}  pout {pct(ko, no)}"))

    # T2 D suppression on the falling limb, stratified permutation
    st = strata_of(rows, "D", SUPPRESS_DEG)
    r1, r2, p2 = strat_perm(st)
    results.append(("T2 D-suppression 40-70 (strat. perm)", p2,
                    f"pin {r1:.3f}  pout {r2:.3f}  ({len(st)} strata)"))

    # T3 A over all angles, stratified permutation
    st = strata_of(rows, "A", ALL_DEG)
    r1, r2, p3 = strat_perm(st)
    results.append(("T3 A all angles (strat. perm)", p3,
                    f"pin {r1:.3f}  pout {r2:.3f}  ({len(st)} strata)"))

    # T4 B over all angles, stratified permutation
    st = strata_of(rows, "B", ALL_DEG)
    r1, r2, p4 = strat_perm(st)
    results.append(("T4 B all angles (strat. perm)", p4,
                    f"pin {r1:.3f}  pout {r2:.3f}  ({len(st)} strata)"))

    adj = holm([r[1] for r in results])
    print("\n--- FOUR PRE-REGISTERED TESTS, HOLM OVER THE FAMILY ---")
    for (name, p, detail), a in zip(results, adj):
        ps = "n/a" if p is None else f"{p:.4f}"
        as_ = "n/a" if a is None else f"{a:.4f}"
        star = " *" if (a is not None and a < 0.05) else ""
        print(f"  {name:<40} p={ps:>7}  holm={as_:>7}{star}")
        print(f"      {detail}")

    # ------------------------------------------------- D intention-to-treat
    print("\n--- D INTENTION-TO-TREAT (gate failure = D 0; amendment 4) ---")
    for label, degs in (("revival 70-85", REVIVAL_DEG), ("suppression 40-70", SUPPRESS_DEG)):
        kp, np_ = pool(rows, "D", degs, "pin", itt=True)
        ko, no = pool(rows, "D", degs, "pout", itt=True)
        p = fisher(kp, np_ - kp, ko, no - ko)
        print(f"  {label:<18} pin {pct(kp, np_)}   pout {pct(ko, no)}   Fisher p={p:.4f}")
        lo, hi = wilson(kp, np_)
        print(f"      pin Wilson 95% [{lo:.3f}, {hi:.3f}]")

    # -------------------------------------------------- J18 replication
    arows = rows_for(ATTR, "unit")
    print("\n" + "=" * 72)
    print("J18 REPLICATION (attrition-replication run) -- own Holm family {A, B}")
    print("=" * 72)
    arms = sorted({r["arm"] for r in arows})
    print(f"samples: {len(arows)}   arms: {arms}")
    print("\n--- gate attrition by arm ---")
    for arm in arms:
        for dose in DOSES:
            sub = [r for r in arows if r["arm"] == arm and r["dose"] == dose]
            if sub:
                f = sum(1 for r in sub if not r["gate"])
                print(f"  {arm:<10} a{dose:<5} gate FAIL {pct(f, len(sub))}")
    print("\n--- item rates by arm (coherent only; missing != 0) ---")
    for arm in arms:
        sub = [r for r in arows if r["arm"] == arm and r["gate"]]
        line = f"  {arm:<10}"
        for item in ("A", "B", "D"):
            v = [r[item] for r in sub if r[item] is not None]
            miss = sum(1 for r in sub if r[item] is None)
            line += f"  {item} {pct(sum(v), len(v))}"
            if miss:
                line += f" (+{miss} missing)"
        print(line)

    rep = []
    for item in ("A", "B"):
        st = []
        for dose in DOSES:
            cell = {}
            for arm in ("jspace", "resid"):
                sub = [r for r in arows if r["arm"] == arm and r["dose"] == dose
                       and r["gate"] and r[item] is not None]
                cell[arm] = (sum(r[item] for r in sub), len(sub))
            if cell["jspace"][1] and cell["resid"][1]:
                st.append((cell["jspace"][0], cell["jspace"][1],
                           cell["resid"][0], cell["resid"][1]))
        r1, r2, p = strat_perm(st)
        rep.append((f"{item}: jspace vs resid", p, f"jspace {r1:.3f}  resid {r2:.3f}  ({len(st)} strata)"))
    adj = holm([r[1] for r in rep])
    print("\n--- replication tests (direction check; power 0.51/0.04/0.63/0.35/0.70) ---")
    for (name, p, detail), a in zip(rep, adj):
        ps = "n/a" if p is None else f"{p:.4f}"
        as_ = "n/a" if a is None else f"{a:.4f}"
        print(f"  {name:<30} p={ps:>7}  holm={as_:>7}")
        print(f"      {detail}")

    # declared missingness
    print("\n--- DECLARED MISSING (deviation note) ---")
    for tag, rr in (("plane", rows), ("attrrep", arows)):
        for item in ("A", "B", "D"):
            miss = [r for r in rr if r["gate"] and r[item] is None]
            if miss:
                for r in miss:
                    print(f"  {tag}: {r['sid']} {r['cond']} item {item} unscorable "
                          "(judge returned no text) -- counted as missing, not 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
