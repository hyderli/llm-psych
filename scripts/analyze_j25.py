"""J25 pilot analysis, written and committed before any J25 sample exists.

Implements the analyses fixed in plans/j-space-decomposition.md ("J25" and its
amendments 1-3). Unit of analysis: the DIRECTION (J20's rule). Each arm's rate
is computed over its gate-passing samples, and the class comparison is an
exact permutation test over those direction-level rates.

Primaries (amendment 3, Holm across three at FWER 0.05):
    H25a'  pool > coh on item B. Exact two-sided permutation over the 924
           6-vs-6 splits; effect = difference of class means, with a 95%
           percentile bootstrap CI (10,000 within-class resamples, seed
           20261107).
    H25c   coh > pool on item P (praise; new extra item, amendment 3).
    H25d   w90 class != w90ctl class on item B (v-deletion as a class).
Sensitivity: each primary rerun on non-degenerate samples (the J24 criterion,
unchanged, including J24's validated repetition criterion).
Secondary: A_act for H25a' and H25d; w90_i vs pool_i paired by source
direction (exact two-sided sign-flip over the 64 assignments). Descriptive:
the 2x2 v/w interaction on B (risk-difference scale with 95% bootstrap CI,
and a +0.5 log-odds version), the pool/full dose point at alpha 0.15 with a
coherent-regime rule (highest of {0.15, 0.25} whose pool class-mean
degeneration is <= 10%, vs unsteered by Fisher), P rates for every condition,
coh vs unsteered on P (pooled Fisher), and the reading row selected by
`dissociation_row`.

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
import itertools
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

# Condition names follow run_dose_sweep.sh: <arm file stem>_L16_a<alpha>,
# with the alpha string taken verbatim from DOSES (so 0.15 -> "a0.15",
# 0.0654 -> "a0.0654"). pool, jspace and full are the J24 files, reused.
LOW_DOSE = "0.15"                 # amendment 3 lower pool/full dose point
V_DOSE = "0.0654"                 # amendment 3: 0.25 * cos(theta*), the v-only cell
CLASSES = {"pool": [f"j24_pool{i + 1}_L16_a{DOSE}" for i in range(N)],
           "coh": [f"j25_coh{i + 1}_L16_a{DOSE}" for i in range(N)]}
W90 = f"j25_w90_L16_a{W90_DOSE}"
W90CTL = f"j25_w90ctl_L16_a{W90_DOSE}"
# Amendment 3: w90/w90ctl become classes; the i = 1 members are the amendment-1
# files (no index in their names), i = 2..6 come from build_j25_w90class.py.
W90CLASSES = {"w90": [W90] + [f"j25_w90_{i}_L16_a{W90_DOSE}" for i in range(2, N + 1)],
              "w90ctl": [W90CTL] + [f"j25_w90ctl_{i}_L16_a{W90_DOSE}"
                                    for i in range(2, N + 1)]}
FULL = f"j24_full_L16_a{DOSE}"
UNSTEERED = "j24_full_L16_a0"
JSPACE = f"j24_jspace_L16_a{DOSE}"
V_ONLY = f"j24_full_L16_a{V_DOSE}"
POOL_LOW = [f"j24_pool{i + 1}_L16_a{LOW_DOSE}" for i in range(N)]
FULL_LOW = f"j24_full_L16_a{LOW_DOSE}"
MAX_DEGEN = 0.10                  # coherent-regime rule, amendment 3


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


def item_values(pay: dict, sc: dict, item: str, c: str,
                exclude: dict | None = None) -> list[int]:
    """Per-sample scores over gate-passing, optionally non-excluded samples."""
    sids = [s for s, d in pay.items() if d["condition"] == c and not d["gate_flags"]
            and (exclude is None or not exclude[s])]
    return [sc[item][s] for s in sids if s in sc[item]]


def class_rates(pay: dict, sc: dict, item: str, exclude: dict | None = None,
                classes: dict | None = None) -> dict:
    out = {}
    for c, conds in (classes or CLASSES).items():
        out[c] = []
        for cn in conds:
            k, n = rate(pay, sc, item, cn, exclude)
            out[c].append(k / n if n else float("nan"))
    return out


def primary(cr: dict, pos: str = "pool", neg: str = "coh") -> dict:
    """Difference of class means, exact two-sided p, bootstrap CI."""
    a, b = cr[pos], cr[neg]
    if any(np.isnan(x) for x in a + b):
        raise ValueError("a direction has no scored samples; the primary cannot run")
    diff = sum(a) / len(a) - sum(b) / len(b)
    p = j24.perm_test(a, b)
    rng_ci = _bootstrap_ci(a, b)
    return {"diff": diff, "p": p, "ci": rng_ci}


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values; J24's implementation."""
    return j24.holm(ps)


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip p on paired direction differences (2^6)."""
    d = np.asarray(diffs, dtype=float)
    if np.isnan(d).any():
        raise ValueError("a paired direction has no scored samples")
    obs = abs(d.mean())
    tot = hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(d)):  # noqa: E501
        tot += 1
        hits += abs((d * np.asarray(signs)).mean()) >= obs - 1e-12
    return hits / tot


def interaction(both: float, w_only: float, v_only: float, none: float) -> float:
    """2x2 interaction on the risk-difference scale: both - w - v + none."""
    return both - w_only - v_only + none


def coherent_dose(degen_rates: dict[float, float],
                  max_degen: float = MAX_DEGEN) -> float | None:
    """Highest dose whose class-mean degeneration rate is <= max_degen."""
    ok = [d for d, r in degen_rates.items() if r <= max_degen]
    return max(ok) if ok else None


def dissociation_row(h25a_pass: bool, h25c_pass: bool, coh_rate: float,
                     unsteered_rate: float, fisher_p: float) -> str:
    """The amendment-3 reading-row rule, as a pure function."""
    if h25a_pass and h25c_pass:
        if coh_rate > unsteered_rate and fisher_p < 0.05:
            return "double dissociation"
        return "not a double dissociation"
    if h25a_pass:
        return "H25c fails"
    return "other"


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


def _class_values(pay: dict, sc: dict, item: str, conds: list[str],
                  exclude: dict | None = None) -> list[list[int]]:
    return [item_values(pay, sc, item, c, exclude) for c in conds]


def interaction_ci(pool_cls: list[list[int]], w90_cls: list[list[int]],
                   v_vals: list[int], none_vals: list[int],
                   seed: int = SEED) -> tuple[float, float]:
    """95% percentile CI for the 2x2 interaction on B.

    Resamples directions within each class and samples within each
    condition/direction (10,000 resamples, seed 20261107).
    """
    rng = np.random.default_rng(seed)

    def cls_mean(cls: list[list[int]]) -> float:
        idx = rng.integers(0, len(cls), size=len(cls))
        return float(np.mean([
            np.mean(rng.choice(cls[j], size=len(cls[j]), replace=True))
            for j in idx]))

    def cond_mean(vals: list[int]) -> float:
        return float(np.mean(rng.choice(vals, size=len(vals), replace=True)))

    draws = np.array([interaction(cls_mean(pool_cls), cls_mean(w90_cls),
                                  cond_mean(v_vals), cond_mean(none_vals))
                      for _ in range(10_000)])
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _log_odds(k: int, n: int) -> float:
    """Pooled-count log-odds with a +0.5 (Haldane) correction on every count."""
    return float(np.log((k + 0.5) / (n - k + 0.5)))


def main() -> int:
    pay, sc = load()
    degen, rep_ok = degenerate_flags(pay)
    present = {d["condition"] for d in pay.values()}

    def missing(classes: dict) -> list[str]:
        return [c for conds in classes.values() for c in conds if c not in present]

    print("# J25 pilot analysis (amendments 1-3)")
    print(f"samples {len(pay)}; repetition criterion "
          f"{'validated (J24)' if rep_ok else 'NOT validated (TOKEN only)'}")

    print("\n## Gate and degeneration per condition")
    for c in sorted(present):
        ss = [s for s, d in pay.items() if d["condition"] == c]
        gate = sum(not pay[s]["gate_flags"] for s in ss)
        dg = sum(degen[s] for s in ss)
        kb, nb = rate(pay, sc, "B", c)
        ka, na = rate(pay, sc, "A_act", c)
        kp, np_ = rate(pay, sc, "P", c)
        kb2, nb2 = rate(pay, sc, "B", c, degen)
        flag = f"  <- < {MIN_NONDEGEN} non-degenerate" if nb2 < MIN_NONDEGEN else ""
        print(f"{c:24s} gate {gate:2d}/{len(ss)}  degen {dg:2d}  B {kb:2d}/{nb:2d}  "
              f"A_act {ka:2d}/{na:2d}  P {kp:2d}/{np_:2d}  "
              f"B non-degen {kb2:2d}/{nb2:2d}{flag}")

    print("\n## Sanity: full@0.25 vs unsteered (same pod), Fisher two-sided")
    for item in ("B", "A_act"):
        a, b = rate(pay, sc, item, FULL), rate(pay, sc, item, UNSTEERED)
        print(f"{item}: full {a[0]}/{a[1]} vs unsteered {b[0]}/{b[1]}, p = {fisher(a, b):.3g}")

    def report(item: str, classes: dict, pos: str, neg: str,
               exclude: dict | None, label: str) -> dict:
        cr = class_rates(pay, sc, item, exclude, classes)
        print(f"\n## {item} {pos} vs {neg}{label}: direction-level rates")
        for c, v in cr.items():
            print(f"  {c:6s} " + " ".join(f"{x:.3f}" for x in v)
                  + f"   mean {np.nanmean(v):.3f}")
        r = primary(cr, pos, neg)
        print(f"{pos} - {neg}: {r['diff'] * 100:+.2f} pp; 95% CI "
              f"[{r['ci'][0] * 100:+.2f}, {r['ci'][1] * 100:+.2f}] pp; "
              f"exact p = {r['p']:.4f}")
        return r

    # --- primaries: fail loudly if any condition is absent -------------------
    primaries = [("H25a'", "B", CLASSES, "pool", "coh"),
                 ("H25c", "P", CLASSES, "coh", "pool"),
                 ("H25d", "B", W90CLASSES, "w90", "w90ctl")]
    for name, _item, classes, _p, _n in primaries:
        m = missing(classes)
        if m:
            raise SystemExit(f"PRIMARY {name} cannot run: "
                             f"condition(s) missing from payloads: {m}")
    if V_ONLY not in present or UNSTEERED not in present:
        print("NOTE: v-only or unsteered condition absent; the 2x2 cells will "
              "report 'condition missing'.")

    print("\n## PRIMARIES (exact permutation over 924 splits each; "
          "Holm across three, FWER 0.05)")
    res: dict[str, dict] = {}
    for name, item, classes, pos, neg in primaries:
        print(f"\n### {name}: {pos} > {neg} on {item}" if name != "H25d"
              else f"\n### {name}: {pos} != {neg} on {item}")
        res[name] = report(item, classes, pos, neg, None, "")
    adj = holm([res[n]["p"] for n, *_ in primaries])
    passes = {}
    print("\nHolm-adjusted (FWER 0.05):")
    for (name, *_), a in zip(primaries, adj, strict=True):
        # H25a' and H25c are directional predictions tested two-sided: a
        # significant difference in the wrong direction is not a pass.
        passes[name] = a < 0.05 and (name == "H25d" or res[name]["diff"] > 0)
        print(f"  {name}: raw p = {res[name]['p']:.4f}   adjusted p = {a:.4f}   "
              f"{'PASS' if passes[name] else 'fail'}")

    print("\n## SENSITIVITY: same tests on non-degenerate samples (J24 criterion)")
    for name, item, classes, pos, neg in primaries:
        try:
            rs = report(item, classes, pos, neg, degen, " non-degenerate")
            agree = (rs["diff"] > 0) == (res[name]["diff"] > 0)
            print(f"  {name}: direction {'agrees' if agree else 'DISAGREES'} "
                  f"({res[name]['diff']:+.3f} vs {rs['diff']:+.3f})")
        except ValueError as e:
            print(f"  {name}: sensitivity cannot run ({e})")

    # --- secondary ------------------------------------------------------------
    print("\n## SECONDARY (A_act): H25a' and H25d")
    report("A_act", CLASSES, "pool", "coh", None, "")
    report("A_act", W90CLASSES, "w90", "w90ctl", None, "")

    print("\n## SECONDARY (B): w90_i vs pool_i paired by source direction, "
          "exact two-sided sign-flip over 64 assignments")
    cr = class_rates(pay, sc, "B", None, {**W90CLASSES, "pool": CLASSES["pool"]})
    diffs = [w - p for w, p in zip(cr["w90"], cr["pool"], strict=True)]
    print("paired diffs (w90 - pool): "
          + " ".join(f"{x:+.3f}" for x in diffs))
    print(f"mean diff {np.mean(diffs):+.3f}; sign-flip p = {sign_flip_p(diffs):.4f}")

    # --- descriptive: the 2x2 on B -------------------------------------------
    print("\n## DESCRIPTIVE: 2x2 interaction on B "
          "(v-hat 0.0654 x w-hat 0.2413 cells)")
    if any(c not in present for c in CLASSES["pool"] + W90CLASSES["w90"]
           + [V_ONLY, UNSTEERED]):
        print("  condition missing: 2x2 interaction not computed")
    else:
        pool_cls = _class_values(pay, sc, "B", CLASSES["pool"])
        w90_cls = _class_values(pay, sc, "B", W90CLASSES["w90"])
        v_vals = item_values(pay, sc, "B", V_ONLY)
        none_vals = item_values(pay, sc, "B", UNSTEERED)
        both_m = float(np.mean([np.mean(vv) for vv in pool_cls]))
        w_m = float(np.mean([np.mean(vv) for vv in w90_cls]))
        v_m = float(np.mean(v_vals)) if v_vals else float("nan")
        n_m = float(np.mean(none_vals)) if none_vals else float("nan")
        est = interaction(both_m, w_m, v_m, n_m)
        lo, hi = interaction_ci(pool_cls, w90_cls, v_vals, none_vals)
        print(f"  cells: both {both_m:.3f}  w-only {w_m:.3f}  "
              f"v-only {v_m:.3f}  none {n_m:.3f}")
        print(f"  I = {est * 100:+.2f} pp; 95% bootstrap CI "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}] pp")
        kb_ = sum(sum(vv) for vv in pool_cls)
        nb_ = sum(len(vv) for vv in pool_cls)
        kw = sum(sum(vv) for vv in w90_cls)
        nw = sum(len(vv) for vv in w90_cls)
        kv, nv = len([x for x in v_vals if x]), len(v_vals)
        kn, nn = len([x for x in none_vals if x]), len(none_vals)
        i_lo = (interaction(_log_odds(kb_, nb_), _log_odds(kw, nw),
                            _log_odds(kv, nv), _log_odds(kn, nn)))
        print(f"  log-odds interaction (+0.5 on every pooled count): {i_lo:+.3f} "
              "(no claim attached; pool is near the ceiling)")

    # --- descriptive: lower dose ----------------------------------------------
    print("\n## DESCRIPTIVE: dose point at alpha 0.15 vs the registered 0.25")
    dose_sets = {0.15: (POOL_LOW, FULL_LOW), 0.25: (CLASSES["pool"], FULL)}
    degen_rates: dict[float, float] = {}
    for dose, (pool_conds, full_c) in dose_sets.items():
        miss = [c for c in pool_conds + [full_c] if c not in present]
        if miss:
            print(f"  dose {dose}: condition missing ({miss})")
            continue
        rates_b, drates = [], []
        for cn in pool_conds:
            k, n = rate(pay, sc, "B", cn)
            rates_b.append(k / n if n else float("nan"))
            ss = [s for s, d in pay.items()
                  if d["condition"] == cn and not d["gate_flags"]]
            drates.append(np.mean([degen[s] for s in ss]) if ss else float("nan"))
        knd = []
        for cn in pool_conds:
            k2, n2 = rate(pay, sc, "B", cn, degen)
            knd.append(k2 / n2 if n2 else float("nan"))
        kf, nf = rate(pay, sc, "B", full_c)
        degen_rates[dose] = float(np.nanmean(drates))
        print(f"  dose {dose}: pool class-mean B {np.nanmean(rates_b):.3f}   "
              f"full B {kf}/{nf}   pool non-degen class-mean B "
              f"{np.nanmean(knd):.3f}   pool class-mean degeneration "
              f"{degen_rates[dose]:.3f}")
        print(f"    pool - full: {np.nanmean(rates_b) - (kf / nf if nf else float('nan')):+.3f}")
    if len(degen_rates) == 2 and UNSTEERED in present:
        d = coherent_dose(degen_rates)
        if d is None:
            print("  coherent-regime effect: neither dose has pool class-mean "
                  "degeneration <= 10%; the pool effect is confined to the "
                  "degenerate regime and every pool claim carries that caveat.")
        else:
            pool_conds = dose_sets[d][0]
            k = sum(rate(pay, sc, "B", cn)[0] for cn in pool_conds)
            n = sum(rate(pay, sc, "B", cn)[1] for cn in pool_conds)
            u = rate(pay, sc, "B", UNSTEERED)
            print(f"  coherent-regime effect: dose {d} (degeneration "
                  f"{degen_rates[d]:.3f} <= {MAX_DEGEN}); pooled pool B "
                  f"{k}/{n} vs unsteered {u[0]}/{u[1]}, "
                  f"Fisher p = {fisher((k, n), u):.3g}")
    elif UNSTEERED not in present:
        print("  coherent-regime effect: condition missing (unsteered)")

    # --- descriptive: P rates --------------------------------------------------
    print("\n## DESCRIPTIVE: P (praise) rates per condition")
    for c in sorted(present):
        k, n = rate(pay, sc, "P", c)
        print(f"  {c:26s} P {k:2d}/{n:2d}  {k / n if n else float('nan'):.3f}")
    print("  2x2 cells (pooled):")
    for label, conds in (("both (pool@0.25)", CLASSES["pool"]),
                         ("w only (w90@0.2413)", W90CLASSES["w90"]),
                         ("v only (full@0.0654)", [V_ONLY]),
                         ("none (full@0)", [UNSTEERED])):
        if any(c not in present for c in conds):
            print(f"    {label}: condition missing")
            continue
        k = sum(rate(pay, sc, "P", cn)[0] for cn in conds)
        n = sum(rate(pay, sc, "P", cn)[1] for cn in conds)
        print(f"    {label}: {k}/{n}  {k / n if n else float('nan'):.3f}")

    # --- coh vs unsteered on P, and the reading row ----------------------------
    print("\n## DESCRIPTIVE: coh vs unsteered on P (same pod), "
          "pooled Fisher two-sided")
    u = rate(pay, sc, "P", UNSTEERED)
    u_rate = u[0] / u[1] if u[1] else float("nan")
    k = sum(rate(pay, sc, "P", cn)[0] for cn in CLASSES["coh"])
    n = sum(rate(pay, sc, "P", cn)[1] for cn in CLASSES["coh"])
    fp = fisher((k, n), u)
    cr_p = class_rates(pay, sc, "P")["coh"]
    n_above = sum(x > u_rate for x in cr_p)
    print(f"  coh pooled {k}/{n} vs unsteered {u[0]}/{u[1]}  "
          f"Fisher p = {fp:.3g};  {n_above}/6 coh directions above unsteered")
    row = dissociation_row(passes.get("H25a'", False), passes.get("H25c", False),
                           k / n if n else float("nan"), u_rate, fp)
    print(f"\n## READING: {row}")

    jk = rate(pay, sc, "B", JSPACE) if JSPACE in present else (0, 0)
    print(f"\n## DESCRIPTIVE: jspace B {jk[0]}/{jk[1]}"
          + ("" if JSPACE in present else "  (condition missing)"))
    print("\nAll results are pilot-scale (n = 40 per direction) and suggestive only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
