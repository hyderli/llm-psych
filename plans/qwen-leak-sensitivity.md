# Plan: Qwen 2.5 7B label-leak sensitivity analysis (C2 at locked layers)

**Date:** 2026-07-31
**Status:** Proposed. Steps 1–2 are done; steps 3–6 need the 7B weights.
**Prereg status:** **Sensitivity analysis — descriptive, not confirmatory.**
No amendment required *provided the locked layers are not changed*. See
§Guardrail.

---

## Motivation

`scripts/screen_story_corpora.py` found that Qwen 2.5 7B violates the
generation constraint ("do not use the word '{emotion}' or any of its
direct synonyms") at high and **differential** rates:

| emotion | label-leak rate | BH-FDR q |
|---|---|---|
| sadness | 27.0% (87/322) | 2.4e-21 |
| admiration | 14.9% (48/322) | 0.010 |
| joy | 9.6% (31/322) | — |
| loathing | 1.6% (5/322) | 1.9e-11 |
| neutral | 0.0% | (structural — see below) |

Llama 3.1 8B and Gemma 2 9B are clean (≤2.5%), so this is a Qwen
instruction-following failure, not a pipeline bug. All 175 matches were
manually verified as genuine emotion words — no `admiral`/`saddle`/`Joyce`
false positives.

**The concern.** Qwen's two best C2 results sit on its two most
contaminated corpora: sadness (L14, intensity ρ +0.94) at 27.0% leak, and
joy (L26, ρ +0.83) at 9.6%. A vector that has partly learned "contains the
token *sorrow*" would track a sadness-intensity ladder very well. The
locked layers for Qwen (2026-07-12 amendment) may therefore be validated
by lexical leakage rather than by an emotion concept.

## Question

Does Qwen's C2 performance at the **already-locked** layers survive
removal of contaminated stories, once sample size is held constant?

## Guardrail (do not violate)

This analysis evaluates C2 **only at the layers already locked in
`configs/vector_validation/layers.yaml`** — Qwen sadness L14, joy L26,
loathing L19. It does **not** sweep layers and does **not** re-apply the
selection rule. That keeps it descriptive and immune to layer-shopping.

If the filtered vectors fail at the locked layers, the correct response is
a dated HYPOTHESES.md amendment authorising re-derivation and
re-selection — **not** a quiet search for a layer that works.

## Design: three arms

Comparing filtered-vs-original alone is confounded, because filtering also
drops n from 322 to 217 per emotion. A ρ drop would be ambiguous between
"leak removed" and "less data". Hence a matched-n control:

| arm | corpus | n/emotion | purpose |
|---|---|---|---|
| A. original | unfiltered | 322 | the published numbers |
| B. filtered | leaks removed, topic-rebalanced | 217 | the treatment |
| C. matched-n control | **unfiltered**, topic-rebalanced random subsample, seed pinned | 217 | isolates n from leak removal |

**The comparison that matters is B vs C**, not B vs A. Arm A is reported
for continuity only.

## Steps

### 1. Screen the corpus — DONE

`results/story_screening/report.md`, screener version 1.0.

### 2. Build the exclusion list — DONE

`uv run python scripts/build_story_exclusions.py --model Qwen2.5-7B-Instruct`

Topic-rebalanced rule: for each topic `t`, keep `k(t) = min_e surviving(e,t)`
stories per emotion. Result: 217/emotion (from 322), 2 of 46 topics dropped
entirely, topic distribution identical across emotions by construction.

* `results/story_screening/Qwen2.5-7B-Instruct/exclusions.json`
* Freeze MD5: `138b110cece2b23b98a1ce3d91200abd`

Quote that MD5 in any result derived from this list.

### 3. Pull Qwen story activations (~300 MB, no GPU)

```
uv run python scripts/sync_hf.py pull activations --model Qwen2.5-7B-Instruct-story
```

`<emotion>.meta.parquet` carries the `story_id` aligned row-wise with the
`.npz` activation rows — this is what makes filtering possible without
re-extraction. **No regeneration and no forward passes are needed for
steps 3–4.**

### 4. Add row filtering to the derivation (CPU only)

Add an optional `--exclude-ids <json>` to
`scripts/derive_story_steering_vectors.py`: load `dropped_story_ids`, drop
the matching rows via `meta.parquet` before the per-emotion mean, and
record the MD5 + surviving n per emotion in the output metadata.

Additive and off by default — arm A stays bit-identical. Per CLAUDE.md §3,
flag in the PR that this file produced already-reported numbers.

Build arm C the same way, with a pinned-seed random topic-rebalanced
subsample to 217/emotion drawn from the unfiltered corpus (seed recorded in
the output metadata).

Outputs:
`steering_vectors/Qwen2.5-7B-Instruct-story-filtered/`
`steering_vectors/Qwen2.5-7B-Instruct-story-matchedn/`

### 5. Re-run C2 at the locked layers only (needs weights)

```
uv run python scripts/validate_intensity_semantic.py  --model qwen25_7b --vectors-dir <arm>
uv run python scripts/validate_implicit_scenarios.py  --model qwen25_7b --vectors-dir <arm>
uv run python scripts/validate_logit_lens.py          --model qwen25_7b --vectors-dir <arm>
```

Use the **frozen confirmation** intensity families (2026-07-28 amendment),
not the sweep stimuli. Restrict to L14 (sadness), L26 (joy), L19 (loathing).
Admiration is already a vector-quality failure and is out of scope here —
it goes to residualization regardless.

### 6. Report

`results/story_screening/Qwen2.5-7B-Instruct/c2_sensitivity.md`: arms A/B/C
side by side per emotion — intensity ρ with bootstrap 95% CI (n=10_000),
implicit accuracy, logit-lens top-5 tokens, and cosine between the arm-A and
arm-B vectors at the locked layer.

## Pre-specified interpretation

Decided before the numbers exist:

* **Survives** — arm B intensity ρ ≥ 0.6 at the locked layer *and* within
  the bootstrap CI of arm C. The locked layers stand; leakage was not
  load-bearing. Report as a passed robustness check.
* **Collapses** — arm B ρ < 0.6 while arm C ρ ≥ 0.6. The C2 validation was
  leak-driven. Qwen's affected emotions are reclassified as not robustly
  validated, and re-derivation/re-selection needs an amendment.
* **Underpowered** — arms B *and* C both drop below 0.6. The loss is from n,
  not leakage; the design cannot answer the question at 217/emotion, and the
  honest report is "inconclusive at reduced n."
* **Cosine check** — if cos(arm A, arm B) at the locked layer is ≈ 1.0, the
  leak was not shifting the vector meaningfully regardless of what ρ does.

## Cost

Steps 3–4 are CPU-only (minutes on the Mac). Step 5 needs Qwen 2.5 7B
weights: ~1 h on one 4090, roughly $1, or a slow local MPS run. Well inside
the remaining budget.

## Known limitations

1. **Screener recall is incomplete.** Detectors are high-precision and
   deliberately narrow (label morphology + a fixed synonym list). Stories
   with subtler lexical tells survive filtering, so arm B is *less* leaky,
   not clean.
2. **Neutral's 0.0% leak rate is structural**, not empirical —
   `LABEL_STEMS["neutral"]` is empty. The neutral rows in the differential
   table are artifacts and were ignored here.
3. **Gemma's 7 extra emotions are unscreened** (no stems defined), so
   Gemma's clean verdict covers only the four primaries.
4. Filtering cannot fix a generation-side problem. If Qwen's emotion stories
   are lexically distinctive in ways beyond the banned words, only
   regeneration or residualization addresses it.

## Related open item (separate decision, not this plan)

The HF dataset holds **12** Gemma emotions — admiration, anger, anxious,
calm, desperate, fear, joy, loathing, loving, nervous, neutral, sadness —
not the 4 in the locked set. Three (`anxious`, `nervous`, `loving`) have no
config in `configs/emotion/` and cannot be regenerated from committed
configs; `desperate`/`calm`/`loving` are emotions the 2026-06-12 amendment
explicitly excluded. `plans/emotion-set-expansion-design.md` states it
becomes a pre-registration "before any expanded-set data is collected" —
that data already exists, so the eventual amendment must disclose post-hoc
timing. The manifests carry no git SHA or timestamp, so the run cannot be
dated. Raise with the team.

## First action if approved

Per CLAUDE.md, start a **new session reading only this file**. Begin at
step 3 (pull activations), then step 4.
