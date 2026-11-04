"""J24 analysis, written and committed before any J24 sample exists.

Implements the analyses fixed in plans/j-space-decomposition.md ("J24"). The
unit of analysis is the DIRECTION (J20's rule): each arm's rate is computed
over its gate-passing samples, and class comparisons are exact permutation
tests over those direction-level rates.

Inputs (from score_blackmail.py parse / ingest, items A_act and B only)::

    results/j24/payloads.jsonl
    results/j24/judge_scores.jsonl
    results/jspace_gate/arms_Qwen2.5-7B-Instruct_story-wheel32_L16_j24.json  (hostile push)

Usage (Mac, project venv; needs the Qwen tokenizer for the TOKEN criterion)::

    uv run python scripts/analyze_j24.py | tee results/j24/analysis_stdout.txt
"""
from __future__ import annotations

import collections
import importlib.util
import itertools
import json
from pathlib import Path

from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "j24"
BUILD = ROOT / "results/jspace_gate/arms_Qwen2.5-7B-Instruct_story-wheel32_L16_j24.json"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DOSE = "0.25"
N = 6
CLASSES = {c: [f"{c}{i + 1}" for i in range(N)] for c in ("pool", "far", "span")}
MAX_SHARE = 0.25
REP8_THRESHOLD = 0.25
MIN_NONDEGEN = 20          # a direction below this in the sensitivity analysis is flagged

_v = importlib.util.spec_from_file_location("val", ROOT / "scripts" / "j24_degeneration_validate.py")
val = importlib.util.module_from_spec(_v)
_v.loader.exec_module(val)


def cond(arm: str, dose: str = DOSE) -> str:
    return f"j24_{arm}_L16_a{dose}"


def load():
    pay = {}
    for d in map(json.loads, open(RES / "payloads.jsonl")):
        pay[d["sid"]] = d
    sc = collections.defaultdict(dict)
    for r in map(json.loads, open(RES / "judge_scores.jsonl")):
        if r["sid"] in pay and r["value"] not in (None, "None", "MISSING"):
            sc[r["item"]][r["sid"]] = int(r["value"])
    return pay, sc


def degenerate_flags(pay: dict) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=REVISION)
    rep_ok = "REPETITION criterion validated" in (RES / "degeneration_validation.txt").read_text()
    out = {}
    for sid, d in pay.items():
        ids = tok(d["full"], add_special_tokens=False)["input_ids"]
        share = (collections.Counter(ids).most_common(1)[0][1] / len(ids)) if ids else 1.0
        r8 = val.rep8(val.sc.words(d["full"]))
        out[sid] = share > MAX_SHARE or (rep_ok and r8 > REP8_THRESHOLD)
    return out, rep_ok


def rate(pay, sc, item, c, exclude=None):
    sids = [s for s, d in pay.items() if d["condition"] == c and not d["gate_flags"]
            and (exclude is None or not exclude[s])]
    v = [sc[item][s] for s in sids if s in sc[item]]
    return sum(v), len(v)


def perm_test(a: list[float], b: list[float]) -> float:
    """Exact two-sided permutation test on the difference in means."""
    pooled, na = a + b, len(a)
    obs = abs(sum(a) / na - sum(b) / len(b))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pooled)), na):
        s = set(idx)
        x = [pooled[i] for i in s]
        y = [pooled[i] for i in range(len(pooled)) if i not in s]
        tot += 1
        hits += abs(sum(x) / len(x) - sum(y) / len(y)) >= obs - 1e-12
    return hits / tot


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def class_rates(pay, sc, item, exclude=None):
    out = {}
    for c, arms in CLASSES.items():
        out[c] = []
        for a in arms:
            k, n = rate(pay, sc, item, cond(a), exclude)
            out[c].append(k / n if n else float("nan"))
    return out


def main() -> int:
    pay, sc = load()
    degen, rep_ok = degenerate_flags(pay)
    build = json.loads(BUILD.read_text())

    print("# J24 analysis")
    print(f"samples {len(pay)}; repetition criterion {'validated' if rep_ok else 'NOT validated (TOKEN only)'}")

    print("\n## Gate and degeneration per condition")
    conds = sorted({d["condition"] for d in pay.values()})
    for c in conds:
        ss = [s for s, d in pay.items() if d["condition"] == c]
        gate = sum(not pay[s]["gate_flags"] for s in ss)
        dg = sum(degen[s] for s in ss)
        kb, nb = rate(pay, sc, "B", c)
        ka, na = rate(pay, sc, "A_act", c)
        kb2, nb2 = rate(pay, sc, "B", c, degen)
        flag = "  <- < %d non-degenerate" % MIN_NONDEGEN if nb2 < MIN_NONDEGEN else ""
        print(f"{c:22s} gate {gate:2d}/{len(ss)}  degen {dg:2d}  B {kb:2d}/{nb:2d}  "
              f"A_act {ka:2d}/{na:2d}  B non-degen {kb2:2d}/{nb2:2d}{flag}")

    print("\n## Sanity: full vs unsteered (same pod), Fisher two-sided")
    for item in ("B", "A_act"):
        a, b = rate(pay, sc, item, cond("full")), rate(pay, sc, item, cond("full", "0"))
        p = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
        print(f"{item}: full {a[0]}/{a[1]} vs unsteered {b[0]}/{b[1]}, p = {p:.3g}")

    def tests(item, exclude=None, label=""):
        cr = class_rates(pay, sc, item, exclude)
        print(f"\n## {item}{label}: direction-level rates")
        for c, v in cr.items():
            print(f"  {c:5s} " + " ".join(f"{x:.3f}" for x in v) + f"   mean {sum(v) / len(v):.3f}")
        p1 = perm_test(cr["pool"], cr["span"])
        p2 = perm_test(cr["pool"], cr["far"])
        p3 = perm_test(cr["far"], cr["span"])
        return cr, (p1, p2, p3)

    cr, (p1, p2, p3) = tests("B")
    h = holm([p1, p2])
    print("\n## PRIMARY (B, as specified; exact permutation over 924 splits; Holm across two)")
    for nm, p, ph, a, b in (("H24a pool vs span", p1, h[0], "pool", "span"),
                            ("H24b pool vs far", p2, h[1], "pool", "far")):
        d = sum(cr[a]) / N - sum(cr[b]) / N
        print(f"{nm}: diff {d:+.3f}  raw p {p:.4f}  Holm p {ph:.4f}")
    print(f"secondary far vs span (descriptive): diff "
          f"{sum(cr['far']) / N - sum(cr['span']) / N:+.3f}  p {p3:.4f}")

    cra, (q1, q2, q3) = tests("A_act")
    print(f"\n## SECONDARY A_act: pool vs span p {q1:.4f}, pool vs far p {q2:.4f}, far vs span p {q3:.4f}")

    crs, (s1, s2, _s3) = tests("B", degen, " (sensitivity: non-degenerate samples)")
    print(f"\n## SENSITIVITY B non-degenerate: pool vs span p {s1:.4f}, pool vs far p {s2:.4f}")
    for nm, a, b, pa, pb in (("H24a", "pool", "span", p1, s1), ("H24b", "pool", "far", p2, s2)):
        d1 = sum(cr[a]) / N - sum(cr[b]) / N
        d2 = sum(crs[a]) / N - sum(crs[b]) / N
        agree = (d1 > 0) == (d2 > 0)
        print(f"{nm}: direction {'agrees' if agree else 'DISAGREES'} across analyses "
              f"({d1:+.3f} vs {d2:+.3f})")

    for ref, why in (("jspace", "descriptive replication of J23 P4"),
                     ("common", "amendment 1: does the shared atom component act?")):
        print(f"\n## {ref} against each span draw ({why}), Fisher")
        jk, jn = rate(pay, sc, "B", cond(ref))
        for a in CLASSES["span"]:
            k, n = rate(pay, sc, "B", cond(a))
            p = fisher_exact([[jk, jn - jk], [k, n - k]])[1]
            print(f"  {ref} {jk}/{jn} vs {a} {k}/{n}: p = {p:.3g}")

    print("\n## EXPLORATORY: B rate vs hostile push, across directions (Spearman)")
    names = ["jspace", "common"] + [a for arms in CLASSES.values() for a in arms]
    xs, ys = [], []
    for a in names:
        k, n = rate(pay, sc, "B", cond(a))
        xs.append(build["arms"][a]["hostile_push"])
        ys.append(k / n if n else float("nan"))
        print(f"  {a:8s} push {xs[-1]:+.3f}  B {ys[-1]:.3f}")
    rho, p = spearmanr(xs, ys)
    print(f"all {len(names)} directions: rho = {rho:+.3f}, p = {p:.3g}")
    pool_x = [build["arms"][a]["hostile_push"] for a in CLASSES["pool"]]
    pool_y = cr["pool"]
    rho2, p2_ = spearmanr(pool_x, pool_y)
    print(f"within pool (n = {N}): rho = {rho2:+.3f}, p = {p2_:.3g}")
    print("\nwithin-class w_hat cosines (build):", json.dumps(build["within_class_cos"]))
    print("pool-far w_hat cosines (build):", json.dumps(build["pool_far_cos"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
