"""J23 amendment 1 analysis: the Qwen L16 main run, Task 0 reused as comparators.

Reads the ingested judge scores from the Task 0 and main-run scoring dirs
(``score_blackmail.py parse/export/ingest`` output) and prints the per-condition
table, predictions 2-5 and the Holm correction over the three primaries. This is
the single source for the numbers in the J23 main-run result entry.

Definitions follow J23 amendment 1. ``pad_empty`` is the parsed scratchpad empty
after stripping. Items are gate-conditioned ("spec"), with J20's bounds: ITT
(missing = 0) and adversarial (gate-failed / unscorable filled against the
claim). Arm angles are measured from the L16 ``.npy`` arms against ``full``.

Usage::

    uv run python scripts/analyze_j23_main.py
    uv run python scripts/analyze_j23_main.py results/t0_qwen results/j23_main
"""
import collections
import json
import sys

from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

ANGLE = {"full": 0.0, "lad10": 10.0, "lad20": 20.0, "lad40": 40.0, "lad60": 60.0,
         "lad70": 70.0, "lad80": 80.0, "lad85": 85.0, "ladstar": 74.84,
         "jspace": 74.84, "resid": 15.16, "randatom": 78.78, "faratom": 89.95,
         "gauss": 90.19, "shuffle": 90.50}  # measured from the L16 .npy arms


def _score(scores: dict, item: str) -> int | None:
    x = scores.get(item)
    return None if x in (None, "None", "MISSING") else int(x)


def load(d: str) -> list[dict]:
    """One record per sample: arm, dose, gate, pad_empty and the four items."""
    payloads = {r["sid"]: r for r in map(json.loads, open(f"{d}/payloads.jsonl"))}
    judged = collections.defaultdict(dict)
    for r in map(json.loads, open(f"{d}/judge_scores.jsonl")):
        judged[r["sid"]][r["item"]] = r["value"]
    out = []
    for sid, p in payloads.items():
        arm, dose = p["condition"].replace("ca_unit_pos_", "").split("_L16_a")
        sc = judged[sid]
        out.append(dict(arm=arm, dose=float(dose), gate=p["G0_auto"],
                        pad_empty=int(not p["pad"].strip()),
                        A_pad=_score(sc, "A_pad"), A_act=_score(sc, "A_act"),
                        B=_score(sc, "B"), D=_score(sc, "D")))
    return out


DIRS = sys.argv[1:] or ["results/t0_qwen", "results/j23_main"]
S = [s for d in DIRS for s in load(d)]


def cell(arm, doses):
    return [s for s in S if s["arm"] == arm and s["dose"] in doses]


def rate(ss, it, mode="spec", claim_lower=True):
    """spec: gate-passing scorable only; itt: missing=0; adv: missing against claim.

    claim_lower=True means the claim is that THIS arm is the lower one, so the
    adversarial fill for this arm is 1; for the comparator arm pass the opposite.
    """
    k = n = 0
    for s in ss:
        ok = s["gate"] == 1 and s[it] is not None
        if mode == "spec":
            if ok:
                k += s[it]
                n += 1
        else:
            n += 1
            if ok:
                k += s[it]
            elif mode == "adv":
                k += 1 if claim_lower else 0
    return k, n


def fisher(a, b):
    return fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]


def fmt(kn):
    return f"{kn[0]}/{kn[1]}={kn[0] / kn[1]:.3f}" if kn[1] else "0/0"


BOTH, D25 = (0.2, 0.25), (0.25,)
print("# Per-condition table (gate, pad_empty, items as-specified)")
print(f"{'arm':9s} {'dose':5s} {'angle':>6s} gate  pad_empty  A_act   B      A_pad  D")
for arm in ["full"] + [a for a in ANGLE if a != "full"]:
    for d in (0.0, 0.2, 0.25):
        ss = cell(arm, (d,))
        if not ss:
            continue
        g = sum(s["gate"] for s in ss)
        pe = sum(s["pad_empty"] for s in ss)
        r = {it: rate(ss, it) for it in ("A_act", "B", "D")}
        ap = rate(ss, "A_pad", "itt")
        print(f"{arm:9s} {d:<5} {ANGLE[arm] if d else 0:6.2f} {g:2d}/20 {pe:2d}/20     "
              f"{r['A_act'][0]:2d}/{r['A_act'][1]:<3d} {r['B'][0]:2d}/{r['B'][1]:<3d} "
              f"{ap[0]:2d}/20  {r['D'][0]:2d}/{r['D'][1]}")

print("\n# Prediction 2: randatom vs resid, pooled over 0.20+0.25")
ra, rs = cell("randatom", BOTH), cell("resid", BOTH)
g = ((sum(s["gate"] for s in ra), 40), (sum(s["gate"] for s in rs), 40))
p2_gate = fisher(*g)
print(f"  gate   randatom {fmt(g[0])}  resid {fmt(g[1])}  p={p2_gate:.4g}  [PRIMARY]")
aa, ar = rate(ra, "A_act"), rate(rs, "A_act")
print(f"  A_act  randatom {fmt(aa)}  resid {fmt(ar)}  delta={aa[0]/aa[1]-ar[0]/ar[1]:+.3f} "
      f"(predicted |delta|<=0.15; descriptive)  p={fisher(aa, ar):.4g}")
for mode in ("spec", "itt", "adv"):
    b1, b2 = rate(ra, "B", mode, True), rate(rs, "B", mode, False)
    print(f"  B {mode:4s} randatom {fmt(b1)}  resid {fmt(b2)}  p={fisher(b1, b2):.4g}  (claim randatom<resid)")

print("\n# Prediction 3: angle trend at 0.25 over full(0), lad10..lad85, gauss (descriptive)")
trend_arms = ["full", "lad10", "lad20", "lad40", "lad60", "lad70", "lad80", "lad85", "gauss"]
for it in ("B", "A_act"):
    xs, ys = [], []
    for a in trend_arms:
        k, n = rate(cell(a, D25), it)
        xs.append(ANGLE[a])
        ys.append(k / n)
    rho, p = spearmanr(xs, ys)
    print(f"  {it:5s} " + "  ".join(f"{a}:{y:.2f}" for a, y in zip(trend_arms, ys, strict=True))
          + f"   Spearman rho={rho:+.3f} p={p:.4g}")
u = cell("full", (0.0,))
print(f"  unsteered: B {fmt(rate(u, 'B'))}  A_act {fmt(rate(u, 'A_act'))}")

print("\n# Prediction 4: jspace vs ladstar at 0.25 (both 74.84 deg)")
js, ls = cell("jspace", D25), cell("ladstar", D25)
print(f"  gate   jspace {sum(s['gate'] for s in js)}/20  ladstar {sum(s['gate'] for s in ls)}/20 "
      f"(predicted jspace <= ladstar)")
p4 = {}
for mode in ("spec", "itt", "adv"):
    b1, b2 = rate(js, "B", mode, True), rate(ls, "B", mode, False)
    p4[mode] = fisher(b1, b2)
    print(f"  B {mode:4s} jspace {fmt(b1)}  ladstar {fmt(b2)}  p={p4[mode]:.4g}"
          f"{'  [PRIMARY]' if mode == 'spec' else ''}  (claim jspace<ladstar)")
a1, a2 = rate(js, "A_act"), rate(ls, "A_act")
print(f"  A_act  jspace {fmt(a1)}  ladstar {fmt(a2)}  p={fisher(a1, a2):.4g} (two-sided, no direction)")

print("\n# Prediction 5: pad_empty")
def pe(ss: list[dict]) -> tuple[int, int]:
    return sum(s["pad_empty"] for s in ss), len(ss)



p5 = fisher(pe(rs), pe(cell("jspace", BOTH)))
print(f"  resid {fmt(pe(rs))}  jspace {fmt(pe(cell('jspace', BOTH)))}  p={p5:.4g}  "
      f"(predicted resid >= jspace)  [PRIMARY]")
lad = ["full", "lad10", "lad20", "lad40", "lad60", "lad70", "lad80", "lad85"]
xs = [ANGLE[a] for a in lad]
ys = [pe(cell(a, D25))[0] / 20 for a in lad]
rho, p = spearmanr(xs, ys)
print("  ladder@0.25 " + "  ".join(f"{a}:{y:.2f}" for a, y in zip(lad, ys, strict=True))
      + f"   unsteered {fmt(pe(u))}   Spearman rho={rho:+.3f} p={p:.4g} (descriptive)")

print("\n# Holm over the three primaries (raw p two-sided Fisher)")
raw = {"P2 gate randatom vs resid": p2_gate, "P4 B jspace vs ladstar": p4["spec"],
       "P5 pad_empty resid vs jspace": p5}
_, holm, _, _ = multipletests(list(raw.values()), method="holm")
for (k, v), h in zip(raw.items(), holm, strict=True):
    print(f"  {k:30s} raw p={v:.4g}  Holm p={h:.4g}")
