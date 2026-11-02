"""Does each model tokenize the C2 numeral grid uniformly?

Gating check for the wheel32 stimulus grid (plans/c2-stimuli-drafts.md).

If 0-9 are single tokens but 10, 11, 12 split into two, then any readout at or
near the numeral position has a step discontinuity at x=10 -- and x=10..12 sit
at the FLAT end of an inverse family's curve, where the design can least afford
extra variance. Three of twelve grid points on the far side of a tokenization
boundary is enough to manufacture or destroy a Spearman result.

Outcome decides the grid:
  * uniform across 0..12  -> keep the twelve-point grid as drafted
  * splits at 10          -> either cap x at 9 (n=10 per family, a disclosed
                             deviation from N_PER_FAMILY=12) or read out at a
                             fixed late position several tokens downstream of
                             the numeral

Run:  uv run python scripts/check_numeral_tokenization.py
"""

from __future__ import annotations

from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
]

# A real carrier frame, so the numeral is scored in context rather than bare.
FRAME = "Of the twenty places she applied to, she is still in the running at {}."
XS = list(range(0, 21))

# Spelled-out denominators tokenize differently from one another and sit at
# different positions across the set; a uniform numeral grid does not imply a
# uniform sentence length.
DENOMINATORS = ["nine", "eleven", "twelve", "fourteen", "nineteen", "twenty"]


def main() -> None:
    for repo in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as exc:  # gated / offline
            print(f"\n{repo}: SKIPPED ({type(exc).__name__}: {exc})")
            continue

        print(f"\n=== {repo} ===")
        print(f"{'x':>3}  {'n_tok':>5}  {'sent_len':>7}  pieces")
        counts, total = {}, {}
        for x in XS:
            text = FRAME.format(x)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            i = text.index(str(x))
            j = i + len(str(x))
            cov = [
                tok.convert_ids_to_tokens(t)
                for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])
                if a < j and b > i
            ]
            counts[x] = len(cov)
            total[x] = len(enc["input_ids"])
            print(f"{x:>3}  {len(cov):>5}  {total[x]:>7}  {cov}")

        singles = sorted(x for x, n in counts.items() if n == 1)
        multis = sorted(x for x, n in counts.items() if n > 1)
        print(f"  single-token x: {singles}")
        print(f"  split x:        {multis}")
        if multis:
            print(f"  -> BOUNDARY at x={multis[0]}; grid must stay below it "
                  f"or the readout must move off the numeral.")
        else:
            print("  -> uniform over the tested range; grid unconstrained here.")

        lens = sorted(set(total.values()))
        print(f"  sentence lengths across the grid: {lens}")
        if len(lens) > 1:
            print("  -> sentence length is NOT invariant across x. Harmless for a "
                  "readout at the final position or a fixed offset from the end; "
                  "corrupting for a fixed absolute index or a fixed window.")

        print("  denominator words:")
        for w in DENOMINATORS:
            ids = tok(w, add_special_tokens=False)["input_ids"]
            print(f"    {w:>9}: {len(ids)} tok {tok.convert_ids_to_tokens(ids)}")


if __name__ == "__main__":
    main()
