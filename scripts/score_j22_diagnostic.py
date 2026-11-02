"""J22 step 1 scorer: apply the frozen decision rule to the diagnostic JSONL.

Standard library only, so it runs on the Mac or the device VM. Everything that
decides the outcome -- the lexicon, the thresholds, the selection rule -- is
fixed here and in the J22 amendment (plans/j-space-decomposition.md) BEFORE the
diagnostic runs. Changing any of it after the JSONL exists requires a new
amendment that records the change and the reason.

Usage::

    python3 scripts/score_j22_diagnostic.py outputs/j22_diag_<...>.jsonl

Rule (J22, verbatim in the amendment):

* A completion is COHERENT if it has >= 20 words and a unique-word ratio
  >= 0.40 (find_alpha.py's looping threshold).
* A completion is HOSTILE if its lexicon hits per 100 words exceed the
  unsteered completion of the same prompt by >= 1.0, AND it hits >= 2
  distinct lexicon entries.
* A cell (layer, alpha), site=post, ACTS if both prompts are coherent and at
  least one is hostile.
* band(L) = number of acting alphas at layer L.
* If no cell acts anywhere: outcome NO-ACT, no Task 0 re-run.
* Otherwise L* = argmax band; ties -> closest to 52% relative depth (Gemma's
  L22/42), then the shallower layer. d1 = lowest acting alpha at L*, d2 = the
  next acting alpha above it; if band(L*) == 1, a single dose at 40 samples.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

# --- frozen by J22 -----------------------------------------------------------
MIN_WORDS = 20
MIN_UNIQUE_RATIO = 0.40
MIN_RATE_LIFT = 1.0          # hits per 100 words above the unsteered completion
MIN_DISTINCT = 2
GEMMA_REL_DEPTH = 22 / 42    # tie-break target

# Prefix stems (matched at word start; all >= 5 characters so they do not
# swallow ordinary words) and exact words (short forms a prefix would overmatch,
# e.g. "hell" -> "hello").
STEMS = (
    "contempt", "despis", "disgust", "loath", "scorn", "sneer", "pathetic",
    "worthless", "incompeten", "inferior", "stupid", "idiot", "foolish",
    "pitiful", "laughabl", "ridiculous", "shameful", "disgrac", "coward",
    "arrogan", "insolen", "furious", "angri", "hostil", "outrag", "reveng",
    "vengean", "punish", "ultimatum", "threat", "insult", "humiliat", "crush",
    "destroy", "obliterat", "annihilat", "attack", "dominat", "demand",
    "fight", "hatred", "useless", "unacceptabl", "inexcusabl", "careless",
)
WORDS = frozenset((
    "hate", "hated", "hates", "hateful", "rage", "raged", "fury", "anger",
    "angry", "fool", "fools", "damn", "vile", "scum",
))
# -----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def lexicon_hits(ws: list[str]) -> tuple[int, set[str]]:
    """Return (total hits, distinct lexicon entries hit)."""
    n, distinct = 0, set()
    for w in ws:
        if w in WORDS:
            n += 1
            distinct.add(w)
            continue
        for s in STEMS:
            if w.startswith(s):
                n += 1
                distinct.add(s)
                break
    return n, distinct


def describe(text: str) -> dict:
    ws = words(text)
    n = len(ws)
    hits, distinct = lexicon_hits(ws)
    return {
        "n_words": n,
        "unique_ratio": (len(set(ws)) / n) if n else 0.0,
        "hits": hits,
        "rate": (100.0 * hits / n) if n else 0.0,
        "distinct": sorted(distinct),
    }


def coherent(d: dict) -> bool:
    return d["n_words"] >= MIN_WORDS and d["unique_ratio"] >= MIN_UNIQUE_RATIO


def hostile(d: dict, baseline_rate: float) -> bool:
    return (d["rate"] - baseline_rate) >= MIN_RATE_LIFT and len(d["distinct"]) >= MIN_DISTINCT


def select(band: dict[int, list[float]], n_layers: int) -> dict:
    """Apply the J22 selection rule to {layer: [acting alphas]}."""
    best = max((len(v) for v in band.values()), default=0)
    if best == 0:
        return {"outcome": "NO-ACT"}
    target = GEMMA_REL_DEPTH * n_layers
    tied = [L for L, v in band.items() if len(v) == best]
    L_star = min(tied, key=lambda L: (abs(L - target), L))
    acting = sorted(band[L_star])
    if len(acting) == 1:
        doses, n_per = [acting[0]], 40
    else:
        doses, n_per = acting[:2], 20
    return {"outcome": "ACTS", "layer": L_star, "doses": doses,
            "samples_per_dose": n_per, "band_size": best, "tied_layers": sorted(tied)}


def main(path: str) -> None:
    header, gens = None, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "header":
                header = r
            elif r.get("record") == "gen":
                gens.append(r)
    if header is None:
        sys.exit("ERROR: no header record; not a j22_layer_diagnostic.py output")
    prompts = list(header["prompts"])
    n_layers = int(header["n_layers"])

    base = {g["prompt_id"]: g for g in gens if g["site"] == "none"}
    if set(base) != set(prompts):
        sys.exit(f"ERROR: unsteered completions missing for {set(prompts) - set(base)}")
    base_d = {p: describe(base[p]["text"]) for p in prompts}

    cell = defaultdict(dict)          # (site, layer, alpha) -> {prompt: desc}
    changed = defaultdict(int)        # (site, layer) -> n alphas where any text != baseline
    for g in gens:
        if g["site"] == "none":
            continue
        key = (g["site"], g["layer"], g["alpha"])
        d = describe(g["text"])
        d["hostile"] = hostile(d, base_d[g["prompt_id"]]["rate"])
        d["coherent"] = coherent(d)
        d["differs"] = g["text"] != base[g["prompt_id"]]["text"]
        cell[key][g["prompt_id"]] = d

    print(f"J22 diagnostic: {header['model']}  ({path})")
    print(f"unsteered: " + "  ".join(
        f"{p} rate={base_d[p]['rate']:.2f} words={base_d[p]['n_words']}" for p in prompts))
    print()
    print("site layer alpha | " + " | ".join(f"{p[:11]:>11} coh host rate" for p in prompts)
          + " | ACTS")
    band = defaultdict(list)
    for (site, L, al) in sorted(cell, key=lambda k: (k[0] != "post", k[1], k[2])):
        ds = cell[(site, L, al)]
        if len(ds) != len(prompts):
            sys.exit(f"ERROR: incomplete cell {(site, L, al)}")
        acts = all(d["coherent"] for d in ds.values()) and any(d["hostile"] for d in ds.values())
        if any(d["differs"] for d in ds.values()):
            changed[(site, L)] += 1
        if site == "post":
            band.setdefault(L, [])
            if acts:
                band[L].append(al)
        row = " | ".join(f"{'':>11} {'Y' if d['coherent'] else 'n':>3} "
                         f"{'Y' if d['hostile'] else 'n':>4} {d['rate']:4.1f}"
                         for d in (ds[p] for p in prompts))
        print(f"{site:>4} {L:>5} {al:5.2f} | {row} | {'ACTS' if acts else ''}"
              + ("" if site == "post" else "   (site check, not selectable)"))

    print("\nbands (post):", {L: band[L] for L in sorted(band)})
    print("text changed vs unsteered (alphas, of", len(header["alphas"]), "):",
          {f"{s}@L{L}": n for (s, L), n in sorted(changed.items())})

    sel = select(dict(band), n_layers)
    print("\n=== J22 rule outcome ===")
    print(json.dumps(sel, indent=2))
    if sel["outcome"] == "NO-ACT":
        if not any(changed.values()):
            print("No text changed at any layer, site or dose: the hook is not landing. "
                  "Debug the injection; this outcome says NOTHING about the vector.")
        else:
            print("Steering changes the text but never produces hostility: J21 "
                  "explanation (i), the vector. Llama closes; no Task 0 re-run.")
    else:
        L28 = band.get(28, [])
        if sel["layer"] != 28 and not L28:
            print("L28 never acts; a shallower layer does: J21 explanation (ii), layer.")
        elif L28 and min(L28) > 0.15:
            print("L28 acts only above J21's doses: J21 explanation (iii), dose, contributed.")
    print("\nHuman read: may VETO an acting cell (record text + reason in J22); "
          "may not add one.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
