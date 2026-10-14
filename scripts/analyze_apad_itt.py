#!/usr/bin/env python3
"""A_pad over the FULL randomised set -- no conditioning on the coherence gate.

Why this is identified where every other A analysis in this project is not:
A_pad is scored on the SCRATCHPAD, and a sample that produced no tool call still
has one. So A_pad can be measured on all 20 samples of every cell, and the
comparison conditions on nothing downstream of treatment. Denominator = 40 per
arm (attrrep) and 280 per arm (plane). Missing data: none.

Status, stated plainly. The DIRECTION is pre-registered -- J18 claimed A higher
under `jspace`, and amendment 4b made that a confirmatory replication. The
OUTCOME SURFACE is not: `A_pad` alone was chosen after seeing that the composite
A = A_pad OR A_act cancels its two surfaces. So this is a confirmatory direction
test on a post-hoc surface, and it is reported as that.

Convention: an empty or near-empty scratchpad scores 0 (ingest enforces it). A
scratchpad with no text expresses no contempt, so 0 is the measurement and not a
failure of it; the per-arm counts of empty pads are printed so the reader can see
where that convention bites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recut_outcomes import fisher
from analyse_variants import holm, sign_p, strat_perm
from analyze_j19 import (ALL_DEG, ATTR, DOSES, PLANE, PLN_RE, UNIT_RE, pct, wilson)


def rows_all(d: Path, kind: str):
    """Every sample, gate ignored, with its A_pad score. No filtering at all."""
    pay = {}
    with (d / "payloads.jsonl").open() as f:
        for line in f:
            j = json.loads(line)
            pay[j["sid"]] = j
    pad = {}
    with (d / "judge_scores.jsonl").open() as f:
        for line in f:
            j = json.loads(line)
            if j.get("item") == "A_pad" and j.get("value") in (0, 1, "0", "1"):
                pad[j["sid"]] = int(j["value"])
    out = []
    for sid, j in pay.items():
        if str(j.get("development", "")).lower() in ("1", "true"):
            continue
        m = (PLN_RE if kind == "plane" else UNIT_RE).search(j["condition"])
        if kind == "plane" and not m:
            arm, deg = "full", None          # the positive control in the sweep
            dose = float(j["condition"].rsplit("_a", 1)[1])
        elif kind == "plane":
            arm, deg, dose = m.group(1), int(m.group(2)), float(m.group(3))
        else:
            arm, deg, dose = m.group(1), None, float(m.group(2))
        assert sid in pad, f"no A_pad for {sid} -- coverage is not complete"
        out.append({"sid": sid, "arm": arm, "deg": deg, "dose": dose,
                    "gate": int(j.get("G0_auto") == 1),
                    "pad_empty": int(not (j.get("pad") or "").strip()),
                    "A_pad": pad[sid]})
    return out


def tally(rows, **sel):
    sub = [r for r in rows
           if all(r[k] == v for k, v in sel.items() if v is not None)]
    return sum(r["A_pad"] for r in sub), len(sub)


def main() -> int:
    print("=" * 76)
    print("A_pad ON THE FULL RANDOMISED SET -- identified, not pre-registered")
    print("=" * 76)

    rows = rows_all(ATTR, "unit")
    print(f"\nattrition run: {len(rows)} samples, all with an A_pad score")
    print(f"\n{'arm':<10} {'A_pad, ALL 40':>22}  {'gate-passing only':>22}  "
          f"{'gate lost':>10}  {'empty pads':>10}")
    full_rates = {}
    for arm in ("resid", "jspace", "randatom", "faratom"):
        k, n = tally(rows, arm=arm)
        kg, ng = tally(rows, arm=arm, gate=1)
        lo, hi = wilson(k, n)
        full_rates[arm] = (k, n)
        emp = sum(r["pad_empty"] for r in rows if r["arm"] == arm)
        print(f"{arm:<10} {pct(k,n):>14} [{lo:.2f},{hi:.2f}]  {pct(kg,ng):>22}  "
              f"{n-ng:>10}  {emp:>10}")

    print("\n--- jspace vs resid on A_pad, the pre-registered DIRECTION ---")
    kj, nj = full_rates["jspace"]
    kr, nr = full_rates["resid"]
    p_f = fisher(kj, nj - kj, kr, nr - kr)
    print(f"  full set      jspace {pct(kj,nj)}   resid {pct(kr,nr)}   Fisher p={p_f:.4f}")
    st = []
    for dose in DOSES:
        a = tally(rows, arm="jspace", dose=dose)
        b = tally(rows, arm="resid", dose=dose)
        st.append((a[0], a[1], b[0], b[1]))
        print(f"    a{dose:<5} jspace {pct(*a):>12}   resid {pct(*b):>12}")
    r1, r2, p_s = strat_perm(st)
    print(f"  stratified by dose: jspace {r1:.3f}  resid {r2:.3f}  "
          f"perm p={'n/a' if p_s is None else f'{p_s:.4f}'}  ({len(st)} strata)")
    kg_j, ng_j = tally(rows, arm="jspace", gate=1)
    kg_r, ng_r = tally(rows, arm="resid", gate=1)
    print(f"  for contrast, gate-conditioned: jspace {pct(kg_j,ng_j)}  "
          f"resid {pct(kg_r,ng_r)}  Fisher p={fisher(kg_j,ng_j-kg_j,kg_r,ng_r-kg_r):.4f}")
    print("  the two differ only by the samples the gate removed; the full-set")
    print("  number needs no bound, because nothing was removed.")

    print("\n--- all six pairs among the four arms, Holm over the family ---")
    arms = ("resid", "jspace", "randatom", "faratom")
    pairs, ps = [], []
    for i, a in enumerate(arms):
        for b in arms[i+1:]:
            ka, na = full_rates[a]
            kb, nb = full_rates[b]
            p = fisher(ka, na - ka, kb, nb - kb)
            pairs.append((a, b, ka/na, kb/nb))
            ps.append(p)
    for (a, b, ra_, rb_), p, adj in zip(pairs, ps, holm(ps)):
        star = " *" if adj < 0.05 else ""
        print(f"  {a:<9} {ra_:.3f}  vs  {b:<9} {rb_:.3f}   p={p:.4g}  holm={adj:.4g}{star}")

    # ---------------------------------------------------------------- plane
    prow = rows_all(PLANE, "plane")
    print("\n" + "=" * 76)
    print("PLANE RUN, A_pad on all 280 per arm")
    print("=" * 76)
    for arm in ("pin", "pout", "full"):
        k, n = tally(prow, arm=arm)
        lo, hi = wilson(k, n)
        kg, ng = tally(prow, arm=arm, gate=1)
        print(f"  {arm:<5} full set {pct(k,n):>16} [{lo:.2f},{hi:.2f}]"
              f"   gate-passing {pct(kg,ng):>16}")
    kp, np_ = tally(prow, arm="pin")
    ko, no = tally(prow, arm="pout")
    print(f"  pin vs pout, full set: Fisher p={fisher(kp,np_-kp,ko,no-ko):.4f}")

    print(f"\n{'angle':>6}  {'pin (40)':>13}  {'pout (40)':>13}")
    dn = up = tie = 0
    for deg in ALL_DEG:
        a = tally(prow, arm="pin", deg=deg)
        b = tally(prow, arm="pout", deg=deg)
        print(f"{deg:>6}  {pct(*a):>13}  {pct(*b):>13}")
        ra_, rb_ = a[0]/a[1], b[0]/b[1]
        tie += abs(ra_-rb_) < 1e-9
        dn += ra_ < rb_ - 1e-9
        up += ra_ > rb_ + 1e-9
    n = up + dn
    print(f"  draw-level: pin lower at {dn}/{n} angles (ties {tie}), "
          f"sign p={sign_p(dn, n):.4f}" if n else "  draw-level: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
