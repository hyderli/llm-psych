"""J23 degeneration check, POST-HOC validity probe of the word criterion.

Not part of the definition at 18931c8; written after degeneration_stdout.txt
existed. Question: on 300-800-word blackmail transcripts, does J22's
unique-word-ratio < 0.40 test detect looping, or does it partly measure length?
J22 calibrated it on 200-token completions, and unique-word ratio falls with
length in ordinary text.

Reports, from the same payloads the check used:
* Spearman rho between word count and unique-word ratio, all 460 samples;
* word-count median per condition (the inert arms' outputs are short);
* for word-flagged samples: length, unique ratio, and the share of word
  8-grams that recur in the sample ("rep8", a length-robust verbatim-repetition
  measure);
* per condition: word-flagged count vs rep8 > 0.25. The 0.25 threshold is
  chosen post-hoc to mirror the token criterion's and is NOT validated.

Standard library only. Usage::

    python3 scripts/j23_word_criterion_validity.py | tee results/j23_lexical/word_criterion_stdout.txt
"""
import collections
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "score_j22_diagnostic.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
REP8_THRESHOLD = 0.25  # post-hoc, unvalidated


def rep8(ws: list[str]) -> float:
    grams = [tuple(ws[i:i + 8]) for i in range(len(ws) - 7)]
    if not grams:
        return 0.0
    c = collections.Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


rows = collections.defaultdict(list)
for f in ("results/t0_qwen/payloads.jsonl", "results/j23_main/payloads.jsonl"):
    for d in map(json.loads, open(ROOT / f)):
        ws = sc.words(d["full"])
        n = len(ws)
        ur = len(set(ws)) / n if n else 0.0
        cond = d["condition"].replace("ca_unit_pos_", "").replace("_L16", "")
        cond = "unsteered" if cond == "full_a0" else cond
        rows[cond].append({"n": n, "ur": ur, "rep8": rep8(ws),
                           "word_flag": n < sc.MIN_WORDS or ur < sc.MIN_UNIQUE_RATIO})

allr = [r for v in rows.values() for r in v]
print("# POST-HOC: validity of the word criterion on J23 transcripts (not in 18931c8)")
print(f"samples {len(allr)}; Spearman rho(word count, unique ratio) = "
      f"{spearman([r['n'] for r in allr], [r['ur'] for r in allr]):+.3f}")
print(f"\n{'condition':15s} {'n_words med':>11s} {'word-flagged':>12s} "
      f"{'rep8>' + str(REP8_THRESHOLD):>10s}   flagged samples (words, unique ratio, rep8)")
for c in sorted(rows, key=lambda c: (c != "unsteered", c)):
    v = rows[c]
    med = sorted(r["n"] for r in v)[len(v) // 2]
    fl = [r for r in v if r["word_flag"]]
    hi = sum(r["rep8"] > REP8_THRESHOLD for r in v)
    detail = " ".join(f"({r['n']},{r['ur']:.2f},{r['rep8']:.2f})" for r in sorted(fl, key=lambda r: r["rep8"]))
    print(f"{c:15s} {med:11d} {len(fl):9d}/{len(v)} {hi:7d}/{len(v)}   {detail}")
