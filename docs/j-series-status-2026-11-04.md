# Project Status — J-series (emotion × J-lens steering)

**Date:** 2026-11-04  ·  **Owner:** Haydar  ·  **Audience:** project team

---

This is a short map of where the emotion × J-lens steering work stands. The authoritative record is `plans/j-space-decomposition.md` (branch `feat/paper-emotion-cells`). Every claim below is in that file with its provenance.

## Per model

| model | layer steered | status |
|---|---|---|
| Gemma 2 9B | L22 (52% depth) | Finished (J17–J20). |
| Llama 3.1 8B | L28 (lens-chosen) | **Closed.** J21 Task 0 failed. The J22 diagnostic found no layer and dose that met its rule. Hostility appeared only at L16, near the coherence edge. One lexicon miss was recorded and not used to reopen. |
| Qwen 2.5 7B | L16 (57%, chosen by the J23 rule; the lens-chosen L24 was inert) | Task 0 passed. Main run done (400 samples), judge pass done, follow-up checks done (J23 results 1–4). |

## Results that hold so far (all n = 20 per cell, suggestive)

1. **The layer the lens decodes best is not the layer where steering acts.** This held on both Llama (L28) and Qwen (L24): the lens-chosen layer did nothing up to α 0.40. Steering acts at about 52–57% depth.
2. **The angle plateau-and-cliff replicates.** For span-free directions, threats fall with angle from v (Qwen ρ ≈ −0.89). The cliff is at 40–60° on Qwen and 60–70° on Gemma.
3. **At a matched angle, construction decides whether a direction acts** (Qwen, 74.8°). The J-lens component (6.8% of the squared norm) produces threats at the level of the full vector. A span-free direction at the same angle produces almost none (16/19 vs 2/20). Counting directions rather than samples past 70°: 2/2 atom-built directions act, against 0/5–0/6 span-free ones. That count is post-hoc. On affect, the effect points the same way on Gemma.

## What failed or reversed

- On Qwen, pre-registered predictions 2, 4 and 5 failed. P4 and P5 reversed significantly (Holm p 9.4e-06).
- Gemma's dissociation (refit damages acting, not affect) does not replicate on Qwen. Qwen never loses the coherence gate.
- Scratchpad suppression is mostly generic. Random directions do it too.

## Open confounds

- **Lexical promotion.** By construction, J-lens directions push the model toward particular output words. The promoted tokens do mark the threats, at about one per threat. That cannot separate "the workspace carries the emotion" from "pushing hostile words produces hostile actions".
- **Degeneration.** The tool-call gate passed degenerate text. The post-hoc degeneration rule failed its positive control, and its word criterion is confounded with length.
- **Control design.** NNLS kept only 2 of 44 `faratom` atoms and 15 of 44 `randatom` atoms, so the controls are not atom-matched.
- **Draws.** The analysis unit should be the direction, but most arms are a single draw.

## Proposed next step

A pre-registered confirmatory re-test (J24) at n ≥ 200, with several independent directions per construction class. Its gate needs a degeneration criterion that is validated on both sides, and the controls need to be atom-matched.
