"""J23 degeneration check (exploratory): are the inert arms inert because their text is broken?

Definition fixed in plans/j-space-decomposition.md ("J23 degeneration check") at
commit 18931c8, before anything here was computed. A sample is DEGENERATE if it
fails J22's coherence rule (< 20 words or unique-word ratio < 0.40) OR its single
most frequent Qwen token is > 25% of its tokens. faratom @0.25 is the positive
control: >= 15 of its 18 samples containing '==(' must be flagged.

Usage::

    uv run python scripts/j23_degeneration_check.py | tee results/j23_lexical/degeneration_stdout.txt
"""
import collections
import importlib.util
import json
from pathlib import Path

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_SHARE = 0.25
INERT_DRAWS = ["lad70", "lad80", "lad85", "ladstar", "gauss", "shuffle"]
ARMS = ["full", "resid", "jspace", "randatom", "faratom", "gauss", "shuffle", "lad10",
        "lad20", "lad40", "lad60", "lad70", "lad80", "lad85", "ladstar"]

spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "score_j22_diagnostic.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=REVISION)
FAR_TOKEN = tok.convert_tokens_to_ids("==(")


def load(d: Path) -> list[dict]:
    b = {}
    for r in map(json.loads, open(d / "judge_scores.jsonl")):
        if r["item"] == "B":
            b[r["sid"]] = None if r["value"] in (None, "None", "MISSING") else int(r["value"])
    out = []
    for p in map(json.loads, open(d / "payloads.jsonl")):
        arm, dose = p["condition"].replace("ca_unit_pos_", "").split("_L16_a")
        ws = sc.words(p["full"])
        ids = tok(p["full"], add_special_tokens=False).input_ids
        share = collections.Counter(ids).most_common(1)[0][1] / len(ids) if ids else 1.0
        short = len(ws) < sc.MIN_WORDS
        loop = len(set(ws)) / max(len(ws), 1) < sc.MIN_UNIQUE_RATIO
        word_fail = short or loop
        out.append(dict(arm=arm, dose=float(dose), gate=p["G0_auto"], B=b.get(p["sid"]),
                        word_fail=word_fail, short=short, loop=loop, share=share,
                        tok_fail=share > MAX_SHARE, degen=word_fail or share > MAX_SHARE,
                        has_far=FAR_TOKEN in ids,
                        far_share=ids.count(FAR_TOKEN) / len(ids) if ids else 0.0))
    return out


S = load(ROOT / "results/t0_qwen") + load(ROOT / "results/j23_main")


def cell(arm, dose):
    return [s for s in S if s["arm"] == arm and s["dose"] == dose]


def b_rate(ss):
    ok = [s["B"] for s in ss if s["gate"] == 1 and s["B"] is not None]
    return f"{sum(ok)}/{len(ok)}"


print("# J23 degeneration check (definition 18931c8)")
unst = cell("full", 0.0)
shares = sorted(s["share"] for s in unst)
print(f"unsteered max-token share: median {shares[len(shares) // 2]:.3f}, max {shares[-1]:.3f} "
      f"(threshold {MAX_SHARE})")

far = [s for s in cell("faratom", 0.25) if s["has_far"]]
n_flag = sum(s["degen"] for s in far)
valid = n_flag >= 15
print(f"\n# Positive control: faratom @0.25 samples containing '==(': {n_flag}/{len(far)} flagged "
      f"(words {sum(s['word_fail'] for s in far)}, tokens {sum(s['tok_fail'] for s in far)}) "
      f"-> rule {'VALID' if valid else 'INSENSITIVE (passes below say nothing)'}")

print("\n# Per condition: degenerate /20 (words, tokens), Fisher vs unsteered, BH-FDR q=0.10")
rows, ps = [], []
u_deg = sum(s["degen"] for s in unst)
for arm in ARMS:
    for d in (0.2, 0.25):
        ss = cell(arm, d)
        if not ss:
            continue
        k = sum(s["degen"] for s in ss)
        p = fisher_exact([[k, len(ss) - k], [u_deg, len(unst) - u_deg]])[1]
        rows.append((arm, d, ss, k))
        ps.append(p)
reject, q, _, _ = multipletests(ps, alpha=0.10, method="fdr_bh")
print(f"  unsteered      {u_deg:2d}/20 (words {sum(s['word_fail'] for s in unst)}, "
      f"tokens {sum(s['tok_fail'] for s in unst)})   B {b_rate(unst)}")
print(f"  {'arm':9s} dose  degen  words tokens  raw p     BH q    B as-spec  B coherent")
for (arm, d, ss, k), p, qq, rj in zip(rows, ps, q, reject, strict=True):
    coh = [s for s in ss if not s["degen"]]
    print(f"  {arm:9s} {d:<5} {k:2d}/20  {sum(s['word_fail'] for s in ss):5d} "
          f"{sum(s['tok_fail'] for s in ss):6d}  {p:<8.3g}  {qq:<6.3g}{'*' if rj else ' '}  "
          f"{b_rate(ss):9s}  {b_rate(coh)}")

print("\n# The draw count (inert draws past 70 deg, dose 0.25)")
kept = []
for arm in INERT_DRAWS:
    coh = [s for s in cell(arm, 0.25) if not s["degen"]]
    keep = len(coh) >= 10
    kept += [arm] if keep else []
    print(f"  {arm:8s} coherent {len(coh):2d}/20   B among coherent gate-passing {b_rate(coh):6s}"
          f"{'' if keep else '   -> DROPPED (< 10 coherent)'}")
print(f"  inert draws retained: {len(kept)}/6 {kept}")
if len(kept) < 6:
    p = fisher_exact([[2, 0], [0, len(kept)]])[1]
    print(f"  re-reported draw count: 2/2 vs 0/{len(kept)}, exact p = {p:.3g}")

print("\n# POST-HOC (not in definition 18931c8; added after the output above)")
print("  faratom @0.25, '==(' share of tokens in each sample containing it, flagged marked *:")
print("   " + " ".join(f"{s['far_share']:.3f}{'*' if s['degen'] else ''}"
                       for s in sorted(far, key=lambda s: s["far_share"])))
print("  word-criterion failures split: short (< 20 words) / looping (unique ratio < 0.40)")
for arm, d, ss, _ in rows:
    n_w = sum(s["word_fail"] for s in ss)
    if n_w:
        print(f"    {arm:9s} {d:<5} words {n_w:2d}: short {sum(s['short'] for s in ss):2d}, "
              f"looping {sum(s['loop'] for s in ss):2d}")
print(f"    unsteered       words {sum(s['word_fail'] for s in unst):2d}: short "
      f"{sum(s['short'] for s in unst):2d}, looping {sum(s['loop'] for s in unst):2d}")
