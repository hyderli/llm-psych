# H8 amendment: workspace-mediated steering

**Date:** 2026-08-27 (draft)  
**Status:** Proposed amendment to `HYPOTHESES.md`. Must be merged before the H2 blackmail / H7 sycophancy steering pods launch.  
**Depends on:** Phase-1 J-space decomposition (`plans/j-space-decomposition.md`).

## Motivation

Phase-1 decomposition shows that the J-space component of each emotion vector is small in absolute terms (~3–15% of squared norm at k=16, consistent with Gurnee et al.'s own regime) but is not uniformly distributed across emotions. The most validated vector — loathing — carries the highest J-space fraction in all three primary models; sadness is second; joy and the failed admiration vector are lower. This pattern raises the causal question that H8 tests: **does the behavioral effect of an emotion steering vector travel through its J-space component, its residual, or both?**

If the J-space component alone reproduces the full-vector effect while the residual is inert, the workspace hypothesis gains a mechanistic basis for the emotion-concept results. If the residual carries the effect, the emotion concept is implemented largely outside the verbalizable workspace. If both carry partial effects, the concepts are hybrid representations.

## Hypothesis (H8)

For validated emotion vectors, the J-space component reproduces the behavioral effect of the full vector at matched norm; the residual component produces at most a substantially smaller effect.

## Design

Five arms at the locked layer for each target emotion and model:

| arm | vector | norm policy |
|---|---|---|
| 1. full | full emotion vector v | native |
| 2. J-matched | J-space component v_j / ||v_j|| × ||v|| | matched to full |
| 3. residual-matched | residual v − v_j / ||v − v_j|| × ||v|| | matched to full |
| 4. J-native | J-space component v_j | native (smaller norm) |
| 5. random-atom control | mean of k random J-space atoms, scaled to ||v|| | matched to full |

- **Primary contrast:** arm 2 vs arm 3 (both matched to the full-vector norm). This is the cleanest test of whether the workspace component is sufficient and necessary.
- **Secondary contrasts:** arm 4 shows whether the native small component still moves behavior; arm 5 controls for “any workspace content of this norm would do it.”
- **Amplification cap:** when norm-matching the J or residual arm, do not amplify beyond 5× the native component norm. If the native component is smaller than 0.2× the full vector, the matched arm is omitted rather than pushed into an unvalidated regime.

## Scope

- **Primary cells (confirmatory):** loathing and sadness, the two emotions that passed C2 validation in all three primaries.
- **Exploratory:** joy in Llama and Qwen (passed) and Gemma (failed).
- **Excluded:** admiration is routed to the text-residualization rescue per `plans/residualization-admiration.md`; it is not a clean H8 cell because the vector failed C2.

## Outcomes and pre-committed interpretation

| result | interpretation |
|---|---|
| J-matched ≈ full, residual ≈ zero | Emotion concept effect is workspace-mediated. Strong support for H8. |
| residual-matched ≈ full, J-matched ≈ zero | Emotion effect lives outside the J-space workspace. Rejects H8; sharpens follow-up into non-verbalizable representation mechanisms. |
| both non-zero | Hybrid representation; emotion concept is distributed across workspace and non-workspace subspaces. |
| random-atom ≈ J-matched | Effect is generic workspace-content at that norm, not emotion-specific. Weakens H8-specific claim. |

All four outcomes are publishable; no post-hoc selection.

## Implementation

1. Re-run J-space decomposition with `k=64` and digit-span projection for all three primaries. `scripts/h8_prep.sh` does this, but is scoped to the legacy `story` track and its four emotions (`--emotions admiration joy loathing sadness`); G2 as amended needs it extended to `story-wheel32` across all cells. **Check first** whether `results/workspace_decomposition_k64/<model_key>-story/manifest.yaml` already carries a `digit_projection` block — if `h8_prep.sh` has run, the legacy-track half of G2 is answered and only the wheel32 extension remains. The wheel32 k=64 set currently on the dataset was produced by `cloud_decompose.sh` **without** `--digit-projection`, so it does not satisfy G2.
2. Use the k=64 manifests to produce norm-matched and native J/residual steering vectors at each locked layer.
3. Inject the five arms into the existing H2/H7 steering harness, reusing the same prompts, alpha schedule, and judge pipeline.
4. Record exact norm ratios, layer, model SHA, and lens revision in run metadata.

## Gates before PR

The amendment can be drafted and reviewed before the k=64 numbers are ready, but the following three values must be filled in from the prep pod run before the PR is opened. They are referred to as G1–G3 elsewhere in the project.

**G1 — budget stability.** `[k64]` Overall J-space fraction stability between k=16 and k=64 for loathing/sadness locked layers. Reads `frac_norm_squared` and is independent of the digit projection. *Satisfiable from existing artefacts:* both budgets are on the dataset for `story-wheel32` — `results/workspace_decomposition/` at k=16 and `results/workspace_decomposition_k64/` at k=64 — with matching file inventories (1921 / 1665 / 2561 files per model in each).

**G2 — digit confound location.** `[k64]` Does the digit confound live in the J component or the residual? Report the digit-span projection for **all wheel32 cells**, and locate admiration and loathing within that distribution rather than comparing the two in isolation.

> **Amended 2026-09-07.** The gate originally named only admiration and loathing. The pairing is the right contrast: admiration failed C2 at every layer on all three primaries with a number-captured pattern (`plans/residualization-admiration.md`), while loathing is the best-validated cell and carries the highest J-space fraction in all three models; they are also opposite poles of the same axis and share the newly-authored numeric stimulus construction (`plans/numerical-intensity-control.md`). But two cells yield a ratio with no dispersion behind it. A digit fraction is not interpretable in absolute terms — some share of any vector lands on digit-topped J-lens atoms simply because those atoms occupy part of the lens basis at that layer (`n_digit_atoms: 2` at Llama L16).
>
> The two-cell scope was a cost decision, and the cost is gone. The projection's dominant term was the layer-invariant `J_l . W_U^T` product, previously recomputed once per vector (~6.6 TFLOP per vector for Gemma, ~97% of runtime); commit ee4bf1d caches it per layer, so computing the projection for 32 cells now costs roughly what 2 cells used to. The full distribution turns "admiration is higher than loathing" into a percentile statement, and supplies per-cell digit fractions that G3 can use as a covariate.

*Pass condition:* admiration's digit fraction sits in the upper tail of the 32-cell distribution while loathing does not, and the split between J component and residual is consistent across at least 2 of 3 models. If admiration and loathing are indistinguishable within the distribution, the metric is not detecting the admiration failure; record that outcome rather than reframing the gate.

**G3 — validation correlation.** `[k64]` Correlation between per-cell J-space fraction and C2 validation metrics across the full layer sweep. Blocked on C2 revalidation for `story-wheel32` (`plans/c2-revalidation-wheel32.md`, decisions D1–D3), since wheel32 cells have no C2 metrics yet. The legacy four-emotion readout at `results/workspace_decomposition_k64/jspace_c2_correlation.csv` stands in the meantime but is not the migrated-track result.

### Known-bad field

The k=16 decomposition at `results/workspace_decomposition/` carries a `digit_projection` block that this plan never requested. In it, `metrics_neg.residual_fraction` is a copy of `metrics_pos.residual_fraction` rather than a measurement of the −v residual — confirmed on 480/480 Llama wheel32 vectors, 0 differing. `v_fraction` and `n_digit_atoms` are unaffected, both being sign-invariant by construction. Fixed in ee4bf1d. Do not read `metrics_neg.digit_projection.residual_fraction` from any k=16 artefact; no gate does, so no backfill is required.

## Relation to phase-1

Phase-1 is purely descriptive; H8 is the first causal test. The k=64 prep run is shared infrastructure for both: it refines the J-space fraction estimates, adds the digit-projection diagnostics, and supplies the vectors needed for the steering arms.
