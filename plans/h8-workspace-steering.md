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

1. Re-run J-space decomposition with `k=64` and digit-span projection for all three primaries (`scripts/h8_prep.sh`).
2. Use the k=64 manifests to produce norm-matched and native J/residual steering vectors at each locked layer.
3. Inject the five arms into the existing H2/H7 steering harness, reusing the same prompts, alpha schedule, and judge pipeline.
4. Record exact norm ratios, layer, model SHA, and lens revision in run metadata.

## Gates before PR

The amendment can be drafted and reviewed before the k=64 numbers are ready, but the following three values must be filled in from the prep pod run before the PR is opened:

- `[k64]` Overall J-space fraction stability between k=16 and k=64 for loathing/sadness locked layers.
- `[k64]` Digit-span projection values for admiration and loathing residuals (does the digit confound live in the J component or residual?).
- `[k64]` Correlation between per-cell J-space fraction and C2 validation metrics across the full layer sweep.

## Relation to phase-1

Phase-1 is purely descriptive; H8 is the first causal test. The k=64 prep run is shared infrastructure for both: it refines the J-space fraction estimates, adds the digit-projection diagnostics, and supplies the vectors needed for the steering arms.
