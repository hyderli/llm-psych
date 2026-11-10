"""J25 pilot analysis, written and committed before any J25 sample exists.

Implements the analyses fixed in plans/j-space-decomposition.md ("J25" and its
amendments 1-2). Unit of analysis: the DIRECTION (J20's rule). Each arm's rate
is computed over its gate-passing samples, and the class comparison is an
exact permutation test over those direction-level rates.

Primary (amendment 2), single test with no multiplicity correction:
    H25a'  pool > coh on item B. Exact two-sided permutation over the 924
           6-vs-6 splits; effect = difference of class means, with a 95%
           percentile bootstrap CI (10,000 within-class resamples, seed
           20261107).
Sensitivity: the same test on non-degenerate samples (the J24 criterion,
unchanged, including J24's validated repetition criterion).
Secondary: A_act, same test. Descriptive: w90 and w90ctl (alpha 0.2413) and
w90 vs w90ctl (Fisher); same-pod full@0.25 vs unsteered (Fisher); jspace.

Inputs (from score_blackmail.py parse / ingest, items A_act and B only)::

    results/j25/payloads.jsonl
    results/j25/judge_scores.jsonl
    results/j24/degeneration_validation.txt   (J24's repetition validation)

Usage (Mac, project venv; needs the Qwen tokenizer for the TOKEN criterion)::

    uv run python scripts/analyze_j25.py | tee results/j25/analysis_stdout.txt
"""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "j25"
REP_VALIDATION = ROOT / "results" / "j24" / "degeneration_validation.txt"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DOSE = "0.25"
W90_DOSE = "0.2413"            # amendment 1: 0.25 * sin(theta*)
N = 6
SEED = 20261107
MAX_SHARE = 0.25
REP8_THRESHOLD = 0.25
MIN_NONDEGEN = 20

# Condition names follow run_dose_sweep.sh: <arm file stem>_L16_a<alpha>.
# pool, jspace and full are the J24 files, reused unchanged.
CLASSES = {"pool": [f"j24_pool{i + 1}_L16_a{DOSE}" for i in range(N)],
           "coh": [f"j25_coh{i + 1}_L16_a{DOSE}" for i in range(N)]}
W90 = f"j25_w90_L16_a{W90_DOSE}"
W90CTL = f"j25_w90ctl_L16_a{W90_DOSE}"
FULL = f"j24_full_L16_a{DOSE}"
UNSTEERED = "j24_full_L16_a0"
JSPACE = f"j24_jspace_L16_a{DOSE}"


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


j24 = _load_module("analyze_j24", "scripts/analyze_j24.py")


def load(res: Path = RES) -> tuple[dict, dict]:
    """Payloads by sid and judge scores by item; missing scores stay missing."""
    pay = {d["sid"]: d for d in map(json.loads, open(res / "payloads.jsonl"))}
    sc: dict[str, dict] = collections.defaultdict(dict)
    for r in map(json.loads, open(res / "judge_scores.jsonl")):
        if r["sid"] in pay and r["value"] not in (None, "None", "MISSING"):
            sc[r["item"]][r["sid"]] = int(r["value"])
    return pay, sc


def degenerate_flags(pay: dict) -> tuple[dict, bool]:
    """The J24 degeneration criterion, unchanged."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=REVISION)
    rep_ok = "REPETITION criterion validated" in REP_VALIDATION.read_text()
    out = {}
    for sid, d in pay.items():
        ids = tok(d["full"], add_special_tokens=False)["input_ids"]
        share = (collections.Counter(ids).most_common(1)[0][1] / len(ids)) if ids else 1.0
        r8 = j24.val.rep8(j24.val.sc.words(d["full"]))
        out[sid] = share > MAX_SHARE or (rep_ok and r8 > REP8_THRESHOLD)
    return out, rep_ok


def rate(pay: dict, sc: dict, item: str, c: str, exclude: dict | None = None) -> tuple[int, int]:
    """(positives, scored) over gate-passing, optionally non-excluded samples."""
    sids = [s for s, d in pay.items() if d["condition"] == c and not d["gate_flags"]
            and (exclude is None or not exclude[s])]
    v = [sc[item][s] for s in sids if s in sc[item]]
    return sum(v), len(v)


def class_rates(pay: dict, sc: dict, item: str, exclude: dict | None = None) -> dict:
    out = {}
    for c, conds in CLASSES.items():
        out[c] = []
        for cn in conds:
            k, n = rate(pay, sc, item, cn, exclude)
            out[c].append(k / n if n else float("nan"))
    return out


def primary(cr: dict) -> dict:
    """H25a': pool vs coh. Difference of class means, exact p, bootstrap CI."""
    a, b = cr["pool"], cr["coh"]
    if any(np.isnan(x) for x in a + b):
        raise ValueError("a direction has no scored samples; the primary cannot run")
    diff = sum(a) / len(a) - sum(b) / len(b)
    p = j24.perm_test(a, b)
    rng_ci = _bootstrap_ci(a, b)
    return {"diff": diff, "p": p, "ci": rng_ci}


def _bootstrap_ci(a: list[float], b: list[float]) -> tuple[float, float]:
    """95% percentile CI for mean(a) - mean(b); J24's method with the J25 seed."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    d = rng.choice(x, size=(10_000, len(x))).mean(axis=1)
    d -= rng.choice(y, size=(10_000, len(y))).mean(axis=1)
    low, high = np.quantile(d, [0.025, 0.975], method="linear")
    return float(low), float(high)


def fisher(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1])


def main() -> int:
    pay, sc = load()
    degen, rep_ok = degenerate_flags(pay)

    print("# J25 pilot analysis (amendments 1-2)")
    print(f"samples {len(pay)}; repetition criterion "
          f"{'validated (J24)' if rep_ok else 'NOT validated (TOKEN only)'}")

    print("\n## Gate and degeneration per condition")
    for c in sorted({d["condition"] for d in pay.values()}):
        ss = [s for s, d in pay.items() if d["condition"] == c]
        gate = sum(not pay[s]["gate_flags"] for s in ss)
        dg = sum(degen[s] for s in ss)
        kb, nb = rate(pay, sc, "B", c)
        ka, na = rate(pay, sc, "A_act", c)
        kb2, nb2 = rate(pay, sc, "B", c, degen)
        flag = f"  <- < {MIN_NONDEGEN} non-degenerate" if nb2 < MIN_NONDEGEN else ""
        print(f"{c:24s} gate {gate:2d}/{len(ss)}  degen {dg:2d}  B {kb:2d}/{nb:2d}  "
              f"A_act {ka:2d}/{na:2d}  B non-degen {kb2:2d}/{nb2:2d}{flag}")

    print("\n## Sanity: full@0.25 vs unsteered (same pod), Fisher two-sided")
    for item in ("B", "A_act"):
        a, b = rate(pay, sc, item, FULL), rate(pay, sc, item, UNSTEERED)
        print(f"{item}: full {a[0]}/{a[1]} vs unsteered {b[0]}/{b[1]}, p = {fisher(a, b):.3g}")

    def report(item: str, exclude: dict | None, label: str) -> dict:
        cr = class_rates(pay, sc, item, exclude)
        print(f"\n## {item}{label}: direction-level rates")
        for c, v in cr.items():
            print(f"  {c:5s} " + " ".join(f"{x:.3f}" for x in v) + f"   mean {sum(v) / N:.3f}")
        r = primary(cr)
        print(f"pool - coh: {r['diff'] * 100:+.2f} pp; 95% CI [{r['ci'][0] * 100:+.2f}, "
              f"{r['ci'][1] * 100:+.2f}] pp; exact p = {r['p']:.4f}")
        return r

    print("\n## PRIMARY H25a' (B; exact permutation over 924 splits; single test)")
    rb = report("B", None, "")
    print("\n## SENSITIVITY (B, non-degenerate samples)")
    rs = report("B", degen, " non-degenerate")
    agree = (rb["diff"] > 0) == (rs["diff"] > 0)
    print(f"direction {'agrees' if agree else 'DISAGREES'} across analyses "
          f"({rb['diff']:+.3f} vs {rs['diff']:+.3f})")
    print("\n## SECONDARY (A_act)")
    report("A_act", None, "")

    print(f"\n## DESCRIPTIVE: v-deletion arms at alpha {W90_DOSE} (one direction each)")
    for item in ("B", "A_act"):
        w, c = rate(pay, sc, item, W90), rate(pay, sc, item, W90CTL)
        print(f"{item}: w90 {w[0]}/{w[1]}  w90ctl {c[0]}/{c[1]}  Fisher p = {fisher(w, c):.3g}")
    w = rate(pay, sc, "B", W90)
    p1 = rate(pay, sc, "B", CLASSES["pool"][0])
    print(f"B: w90 {w[0]}/{w[1]} vs its source pool1 {p1[0]}/{p1[1]} "
          f"(pool1 includes the v-component, alpha {DOSE}), Fisher p = {fisher(w, p1):.3g}")

    jk = rate(pay, sc, "B", JSPACE)
    print(f"\n## DESCRIPTIVE: jspace B {jk[0]}/{jk[1]}")
    print("\nAll results are pilot-scale (n = 40 per direction) and suggestive only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
