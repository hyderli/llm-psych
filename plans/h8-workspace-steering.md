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
| 6. angle ladder (calibration) | v rotated away from its own direction by θ ∈ {10°, 20°, 40°, 60°, 80°} and by θ* = angle(v_j, v); each rotation uses a direction perpendicular to v **and** to the J-space span this vector used, averaged over ≥3 random draws per θ | matched to full |

- **Calibrate before interpreting anything (revised 2026-09-08).** Arm 6 is a *ladder*, not a single control. We do not currently know how the steering effect decays as v is rotated away from its own direction — that has never been measured. Without that curve no arm in this table is readable, because every arm sits at some angle from v: arm 3 at 15-20°, arm 2 at 68-77° (the component and residual are orthogonal, so cos(v, v_j) = √frac and cos(v, r) = √(1−frac)). The ladder measures effect-vs-angle directly and turns every other arm into a point comparable against a known baseline.
- **Primary contrast:** arm 2 vs the ladder point at θ* = angle(v_j, v). Same length, same angle to v, and the ladder's rotation is drawn from outside the J-space span this vector used — so the only remaining difference is whether the vector is built from workspace material. Arm 2 beating the ladder at its own angle supports H8; a tie means the effect tracks alignment with v rather than workspace membership.
- **Secondary contrasts:** arm 2 vs arm 5 controls for “any workspace content of this norm would do it”; arm 4 shows whether the native, unamplified component moves behavior at all. Arm 4 matters more than its label suggests: norm-matching a 5-13% component means amplifying it 2.8-4.5x, well outside any regime these vectors were validated in, so “arm 2 ≈ zero” could mean the workspace does not carry the effect *or* that the amplified vector left the linear regime. Arm 4 is the only arm free of that ambiguity.
- **Arm 3 is retained as a full-residual ablation, not as a co-primary.** Because the component and residual are orthogonal (`cos_component_residual` = 0 in every manifest), `cos(v, r) = √(1 − frac)`, which is 0.93 on Llama and 0.97 on Qwen and Gemma. Arm 3 is the full vector at a 15-20° angle, so “arm 3 ≈ arm 1” is close to guaranteed and carries almost no information. Report it; do not treat it as the counterpart to arm 2.

  The earlier framing (arm 2 vs arm 3 as the primary) is clean in one direction only. Arm 2 ≈ full with arm 3 ≈ zero would be unambiguous and strong — the small slice works while the large remainder does not. But the opposite result, arm 3 ≈ full with arm 2 ≈ zero, is predicted equally well by two different accounts: that the effect lives outside J-space, and that steering simply tolerates a 15-20° rotation. Arms 1-3 cannot separate those. The ladder can.
- **Amplification cap:** when norm-matching the J or residual arm, do not amplify beyond 5× the native component norm. If the native component is smaller than 0.2× the full vector, the matched arm is omitted rather than pushed into an unvalidated regime.

> **Arms 2 and 3 are not symmetric alternatives (added 2026-09-08).** Measured J-space fractions at k=64 are ~13% of squared norm on Llama, ~6% on Qwen and ~5% on Gemma, so the residual is 87-95% of the vector. Arm 3 is therefore very nearly arm 1 by construction, and "residual-matched ≈ full" is close to a guaranteed outcome carrying almost no information. The informative result is **arm 2 succeeding**: a 5-13% slice, amplified 2.8-4.5x to match norm, reproducing the full vector's effect — which is also the shape of the parent paper's finding that J-space accounts for ≤10% of activation variance yet mediates report. Read the outcome table with that asymmetry in mind; do not report arms 2 and 3 as a symmetric horse race. Note the amplification cap of 5x is close to binding on Gemma (4.5x needed at 5%).


## Scope

- **Primary cells (confirmatory):** loathing and sadness, the two emotions that passed C2 validation in all three primaries.
- **Exploratory:** joy in Llama and Qwen (passed) and Gemma (failed).
- **Excluded:** admiration is routed to the text-residualization rescue per `plans/residualization-admiration.md`; it is not a clean H8 cell because the vector failed C2.

## Outcomes and pre-committed interpretation

| result | interpretation |
|---|---|
| arm 2 ≈ full, ladder at θ* ≈ zero | The effect follows J-space *membership*, not mere alignment with v. Strong support for H8. |
| arm 2 ≈ ladder at θ*, both non-zero | The effect follows alignment with v at this norm; J-space membership adds nothing. Rejects workspace *mediation of emotion steering* — not Gurnee et al.'s report/reasoning results, which are a different task; and “outside the workspace” here means outside the specific atoms this vector used, not outside the lens as a whole. State it with both limits. |
| arm 2 ≈ arm 5 | The effect is generic workspace content at this norm, not this emotion's workspace content. Weakens the H8-specific claim independently of arm 6. |
| arm 2 ≈ zero, ladder at θ* ≈ zero, arm 1 non-zero | The effect needs the full vector; neither a workspace slice nor an angle-matched sham reproduces it. Hybrid or distributed representation. |
| arm 4 non-zero | The native, unamplified component moves behavior on its own — the strongest available form of support, since it needs no amplification. |
| arm 3 ≈ full | Expected by construction (cos(v, r) = 0.93-0.97). Reported for completeness; never read on its own as evidence for or against H8. |

All outcomes are publishable; no post-hoc selection. Note the outcomes are *not* symmetric in what they cost to obtain: arm 2 succeeding is informative precisely because the component is 5-13% of squared norm and must be amplified 2.8-4.5x to be tested at all.

## Implementation

1. Re-run J-space decomposition with `k=64` and digit-span projection for all three primaries. `scripts/h8_prep.sh` does this, but is scoped to the legacy `story` track and its four emotions (`--emotions admiration joy loathing sadness`); G2 as amended needs it extended to `story-wheel32` across all cells. **Check first** whether `results/workspace_decomposition_k64/<model_key>-story/manifest.yaml` already carries a `digit_projection` block — if `h8_prep.sh` has run, the legacy-track half of G2 is answered and only the wheel32 extension remains. The wheel32 k=64 set currently on the dataset was produced by `cloud_decompose.sh` **without** `--digit-projection`, so it does not satisfy G2.
2. Use the k=64 manifests to produce norm-matched and native J/residual steering vectors at each locked layer.
3. Run the staged sequence below. **The H2/H7 steering harness does not exist yet** (as of 2026-09-08 the repo has injection machinery in `src/llm_psych/hooks.py` and frozen blackmail / reward-hacking scenarios, but no runner and no steering results of any kind). Building it is a prerequisite of this amendment, not an assumption of it.
4. Record exact norm ratios, angles to v, layer, model SHA, and lens revision in run metadata.

### Staged sequence (added 2026-09-08)

The five arms all presuppose that the full vector reliably moves behavior on these tasks at 7-9B. That has never been shown in this project. Decomposing an effect before demonstrating it is the wrong order, and each stage below is a prerequisite for reading the next.

**Stage 0 — does the full vector steer at all?** Arm 1 only: full vector, dose-response across the alpha schedule, on the behavioral tasks, for the Tier A cells that pass C2 confirmation. This is a subset of the planned work, not extra. *Stop condition:* if the full-vector effect is absent or inconsistent at this scale, H8 has nothing to decompose and the amendment is withdrawn rather than run. Report that outcome — a null on emotion steering at 7-9B against Sofroniew et al.'s larger-model result is itself publishable.

**Stage 1 — how tolerant is steering to direction?** The arm 6 ladder. Without it no arm is interpretable, because every arm sits at some angle from v and we have no idea what a given angle costs. Note the prior from the literature is *not* reassuring: refusal vectors derived by different methods at pairwise cosine 0.10-0.42 engage >90%-overlapping circuits (arXiv 2604.08524), which suggests steering may tolerate very large angular differences. If the ladder is flat out to 70-80°, arms 2-6 cannot discriminate anything and the design must change before it is run. That paper's vectors were different estimates of the *same* concept direction rather than arbitrary directions, so it is a warning rather than a result — but it is the reason the ladder comes before the components, not after.

**Stage 2 — the component arms.** Arms 2-5 plus the ladder point at θ*, per the contrasts above, run only if stage 0 shows an effect and stage 1 shows enough angular selectivity for the comparison to mean something.

Prior work supports the norm-matching choice: concept information is carried by activation *direction*, not hidden-state magnitude, with norm governing generation stability instead (arXiv 2606.06735, tested on Llama-3.1-8B, Qwen2.5-7B and Gemma-2-9B — our three primaries). No published angular tolerance band exists for these models, so stage 1 is a contribution in its own right rather than only a control.

## Gates before PR

The amendment can be drafted and reviewed before the k=64 numbers are ready, but the following three values must be filled in from the prep pod run before the PR is opened. They are referred to as G1–G3 elsewhere in the project.

**G1 — budget stability.** `[RESOLVED 2026-09-08]` Overall J-space fraction stability between k=16 and k=64. Reads `frac_norm_squared`; independent of the digit projection. Measured by `scripts/jspace_gate_readout.py` over 3360 paired vectors (emotion × layer × sign, both tracks, three models):

| model | median rel. change | Spearman ρ | f64 < f16 | pursuit hit k=64 cap |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 25.6% | 0.958 | 0 | 96.0% |
| Qwen2.5-7B-Instruct | 3.0% | 0.993 | 0 | 24.4% |
| gemma-2-9b-it | 0.3% | 1.000 | 0 | 0.6% |

**Verdict: passes for orderings, fails for magnitudes.** Rank order is preserved under a 4× budget change on every model (ρ ≥ 0.906 including the add-on track), and `f64 < f16` is zero everywhere, confirming the pursuit is monotone in its budget. But the median fraction moves 25.6% on Llama, and the size of that movement is predicted almost exactly by whether the pursuit exhausted its atom budget.

The pursuit's only stopping condition is geometric exhaustion — it halts when no remaining candidate atom has positive correlation with the residual (`if corr[i] <= 0: break`), with no error tolerance. Llama therefore never converges at k=64 and its fraction is a lower bound tracking the cap; Gemma converges on 99.4% of vectors and its fraction is a measured value.

**This is not resolved by raising k.** J-space is *defined* as sparse nonnegative combinations of ≤k lens vectors, so k is part of the construct, not a nuisance parameter concealing a true value. Pushing k toward the 512-candidate pool would asymptotically measure "how much of v lies in the nonnegative cone of the lens," a different quantity from the one the parent paper's claims concern.

**Consequent rules, binding on all downstream use:**

1. Every reported J-space fraction states its k. No number is ever "the" J-space fraction.
2. k=64 is the pre-registered operating budget for H8; k=16 stands as the sensitivity check.
3. Absolute-magnitude claims are made only where the pursuit converged; comparative and rank claims are made anywhere, since orderings are stable.
4. Cross-model comparison of arm 2 results reports each model's k=64 cap-hit rate alongside, because Llama's J component is a truncation while Gemma's is converged — the two are not the same kind of object.

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
