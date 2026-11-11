"""J29 analysis: G/specific split of the pool word direction.

Implements the analyses fixed in plans/j-space-decomposition.md ("J29",
2026-11-11), run ONCE after J29's payloads and judge scores exist. Same
conventions as analyze_j25.py: direction-level rates, exact 6v6 permutation
on class means, pooled Fisher vs unsteered for the "acts" rule, J24
degeneration criterion for the sensitivity.

Classes (all at alpha 0.15, L16):
    poolG  j29_poolG{1..6}  -- the G component g_i = P_G(w_i), in-arm dose
    poolS  j29_poolS{1..6}  -- the specific component s_i = w_i - g_i
    farG   j29_farG{1..6}   -- far's G component (secondary: is "generic"
                               generic?)
Reference (reused from results/j25):
    pool   j24_pool{1..6} @ 0.15 ; unsteered j24_full_L16_a0

Primary H29: the 2x2-style reading
    s acts & g doesn't -> hostile-specific content carries the effect
    g acts & s doesn't -> generic English push carries it
    both               -> additive/interaction, compared against pool itself
    neither            -> the split destroys what acts
Secondary H29s: farG vs unsteered.

Inputs: results/j29/{payloads,judge_scores}.jsonl plus the J25 files for the
reference conditions. Usage::

    uv run python scripts/analyze_j29.py | tee results/j29/analysis_stdout.txt
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES29 = ROOT / "results" / "j29"
RES25 = ROOT / "results" / "j25"


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


j25 = _load_module("analyze_j25", "scripts/analyze_j25.py")
j24 = j25.j24

UNSTEERED = "j24_full_L16_a0"
CLASSES = {
    "poolG": [f"j29_poolG{i}_L16_a0.15" for i in range(1, 7)],
    "poolS": [f"j29_poolS{i}_L16_a0.15" for i in range(1, 7)],
    "farG": [f"j29_farG{i}_L16_a0.15" for i in range(1, 7)],
    "pool": [f"j24_pool{i}_L16_a0.15" for i in range(1, 7)],
}


def load_both() -> tuple[dict, dict]:
    """J29 payloads + scores merged with the J25 reference conditions."""
    pay, sc = j25.load(RES29)
    pay25, sc25 = j25.load(RES25)
    ref_conds = {UNSTEERED, *CLASSES["pool"]}
    ref_sids = {s for s, d in pay25.items() if d["condition"] in ref_conds}
    pay.update({s: pay25[s] for s in ref_sids})
    for item, m in sc25.items():
        sc[item].update({s: v for s, v in m.items() if s in ref_sids})
    return pay, sc


def acts(pay: dict, sc: dict, conds: list[str], exclude: dict | None) -> dict:
    """Pooled B vs unsteered, Fisher two-sided; the fixed 'acts' rule."""
    vals = [j25.item_values(pay, sc, "B", c, exclude) for c in conds]
    k, n = sum(sum(v) for v in vals), sum(len(v) for v in vals)
    ku, nu = j25.rate(pay, sc, "B", UNSTEERED, exclude)
    return {"k": k, "n": n, "p": j25.fisher((k, n), (ku, nu)),
            "rate": k / n if n else float("nan"), "unsteered": (ku, nu)}


def main() -> int:
    pay, sc = load_both()
    degen, rep_ok = j25.degenerate_flags(pay)
    present = {d["condition"] for d in pay.values()}
    missing = [c for conds in CLASSES.values() for c in conds if c not in present]
    if missing:
        raise SystemExit(f"conditions missing from payloads: {missing}")

    print("# J29 analysis (G/specific split; prereg 2026-11-11)")
    print(f"samples {len(pay)}; repetition criterion "
          f"{'validated (J24)' if rep_ok else 'NOT validated (TOKEN only)'}")

    print("\n## Gate and degeneration per condition")
    for c in sorted(present):
        ss = [s for s, d in pay.items() if d["condition"] == c]
        gate = sum(not pay[s]["gate_flags"] for s in ss)
        dg = sum(degen[s] for s in ss)
        kb, nb = j25.rate(pay, sc, "B", c)
        kb2, nb2 = j25.rate(pay, sc, "B", c, degen)
        print(f"{c:26s} gate {gate:2d}/{len(ss)}  degen {dg:2d}  "
              f"B {kb:2d}/{nb:2d}  B non-degen {kb2:2d}/{nb2:2d}")

    def table(exclude: dict | None, label: str) -> dict:
        cr = j25.class_rates(pay, sc, "B", exclude, CLASSES)
        print(f"\n## B rates per direction{label}")
        for c, v in cr.items():
            print(f"  {c:6s} " + " ".join(f"{x:.3f}" for x in v)
                  + f"   mean {np.nanmean(v):.3f}")
        return cr

    out = {}
    for exclude, label in [(None, ""), (degen, " (non-degenerate sensitivity)")]:
        cr = table(exclude, label)
        print(f"\n## 'Acts' tests{label}: pooled B vs unsteered "
              f"({j25.rate(pay, sc, 'B', UNSTEERED, exclude)})")
        for c in ("pool", "poolG", "poolS", "farG"):
            r = acts(pay, sc, CLASSES[c], exclude)
            out[(c, bool(exclude))] = r
            print(f"  {c:6s} {r['k']:3d}/{r['n']:3d} ({r['rate']:.3f}) vs "
                  f"unsteered, Fisher p = {r['p']:.4g}  "
                  f"{'acts' if r['p'] < 0.05 else 'inert'}")
        # H29 primary: poolS vs poolG class means, exact permutation
        r = j25.primary(cr, "poolS", "poolG")
        print(f"  H29 poolS - poolG: {r['diff'] * 100:+.2f} pp; 95% CI "
              f"[{r['ci'][0] * 100:+.2f}, {r['ci'][1] * 100:+.2f}] pp; "
              f"exact p = {r['p']:.4f}")
        out[("h29", bool(exclude))] = r

    sg = out[("poolS", False)]["p"] < 0.05
    gg = out[("poolG", False)]["p"] < 0.05
    fg = out[("farG", False)]["p"] < 0.05
    row = ("hostile-specific content carries it" if sg and not gg else
           "generic English push carries it" if gg and not sg else
           "both act (additive/interaction)" if sg and gg else
           "neither acts; the split destroys the effect")
    print(f"\n## REGISTERED READING: s {'acts' if sg else 'inert'}, g "
          f"{'acts' if gg else 'inert'} -> {row}")
    print(f"## H29s (farG vs unsteered): {'acts' if fg else 'inert'} "
          f"-> {'generic push alone is not sufficient' if not fg else 'generic is generic'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
