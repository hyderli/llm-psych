"""J24 degeneration criterion: validate the repetition arm on J23 data, before J24 runs.

J24's degeneration rule (plans/j-space-decomposition.md, "J24") flags a sample
if EITHER
  * TOKEN: its single most frequent Qwen token is > 25% of its tokens (J23's
    token criterion, unchanged; validated in J23 result 4: it flags every
    faratom 0.25 sample above 25% and the unsteered maximum share is 0.053), or
  * REPETITION: rep8 > 0.25, where rep8 is the share of the sample's word
    8-grams that occur more than once in it. This replaces J22's unique-word
    ratio, which J23 result 4 showed is confounded with length.

This script validates REPETITION on both sides with data that already exists:
  negative control  J23 unsteered + Task 0 full (60 samples): at most 1 flagged;
  positive control  each of the 20 unsteered transcripts with a verbatim loop
                    appended (its last 30 words, repeated until the loop is
                    >= 30% of all words): all 20 flagged.
The acceptance rules are fixed in the J24 amendment. If either fails, J24's
sensitivity analysis uses TOKEN alone and says so.

Standard library only::

    python3 scripts/j24_degeneration_validate.py | tee results/j24/degeneration_validation.txt
"""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "score_j22_diagnostic.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

REP8_THRESHOLD = 0.25
LOOP_WORDS = 30
LOOP_SHARE = 0.30
MAX_NEG_FLAGS = 1


def rep8(ws: list[str]) -> float:
    grams = [tuple(ws[i:i + 8]) for i in range(len(ws) - 7)]
    if not grams:
        return 0.0
    c = collections.Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)


def with_loop(ws: list[str]) -> list[str]:
    """Append copies of the last LOOP_WORDS words until they are >= LOOP_SHARE of all words."""
    chunk = ws[-LOOP_WORDS:]
    out, added = list(ws), 0
    while added < LOOP_SHARE * len(out):
        out += chunk
        added += len(chunk)
    return out


def main() -> int:
    t0 = [json.loads(l) for l in open(ROOT / "results/t0_qwen/payloads.jsonl")]
    neg = [d for d in t0 if d["condition"].startswith("ca_unit_pos_full_L16_a")]
    unst = [d for d in t0 if d["condition"] == "ca_unit_pos_full_L16_a0"]
    neg_vals = [rep8(sc.words(d["full"])) for d in neg]
    n_neg = sum(v > REP8_THRESHOLD for v in neg_vals)
    pos_vals = []
    for d in unst:
        ws = with_loop(sc.words(d["full"]))
        pos_vals.append(rep8(ws))
    n_pos = sum(v > REP8_THRESHOLD for v in pos_vals)

    print("# J24 degeneration criterion, REPETITION arm (rep8 > %.2f)" % REP8_THRESHOLD)
    print(f"negative control: {n_neg}/{len(neg)} flagged (max rep8 {max(neg_vals):.3f}); "
          f"accept if <= {MAX_NEG_FLAGS} -> {'PASS' if n_neg <= MAX_NEG_FLAGS else 'FAIL'}")
    print(f"positive control: {n_pos}/{len(unst)} flagged (min rep8 {min(pos_vals):.3f}); "
          f"accept if {len(unst)}/{len(unst)} -> {'PASS' if n_pos == len(unst) else 'FAIL'}")
    ok = n_neg <= MAX_NEG_FLAGS and n_pos == len(unst)
    print("VERDICT:", "REPETITION criterion validated" if ok
          else "REPETITION criterion NOT validated; J24 sensitivity uses TOKEN alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
