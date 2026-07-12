# Pre-registered amendment: per-emotion layer selection for H1 concept validation

**Date:** 2026-07-12
**Status:** APPROVED 2026-07-12 with modifications — see the 2026-07-12
amendment block in HYPOTHESES.md, which supersedes this draft where they
differ. Modifications: (1) explicit vector-quality clause (the draft's rule
never actually fails admiration, whose implicit accuracy is ~1.00 at nearly
all layers — the expected-outcomes table below conflates intensity with
implicit numbers); (2) confirmation requires FRESH inverse families (the
existing intensity sweep is already observed, hence retrospective/descriptive
only — it is not held out); (3) selection on 3-layer moving average, exact-tie
break shallower (the ±0.02 window is vacuous at n=10 granularity 0.1).
**Triggered by:** C2 validation sweep results (2026-07-11/12) showing that the uniform ~2/3-depth layer convention fails for sadness, joy, and admiration on the three primaries, while a per-emotion layer grid recovers sadness and partially recovers joy.

---

## Problem the sweep revealed

The C2 validation suite (`scripts/validate_{implicit_scenarios,intensity_semantic}.py`) was run at a **uniform layer per model** (Llama L21, Qwen L19, Gemma L28) — the ~2/3-depth convention used in the paper and adopted in this project. The sweep across layers (15–20 layers per model) shows:

- **Sadness** recovers strongly at non-conventional layers on all three models (Llama +0.73 at L16, Qwen +0.94–0.96 at L14–18, Gemma +0.71–0.96 at L21–22 and L39–40). This is a **layer-selection problem**, not a vector-quality problem.
- **Joy** recovers at earlier layers on Llama (+0.71–0.77 at L16–18) and Qwen (+0.89 at L14, +0.62–0.83 at L24–26), but never reaches 0.6 on Gemma (max +0.56). Mixed: layer-selection on two families, possibly genuine on Gemma.
- **Admiration** fails at **every layer** on all three models (Gemma is anti-semantic at all 20 layers, Qwen never exceeds +0.16, Llama hovers +0.2–0.6). This is a **vector-quality problem** — the story-method construction for admiration is confounded by the digit itself (number-captured pattern), and no layer choice rescues it.
- **Loathing** passes everywhere; no change needed.

The uniform-layer convention therefore **over-rejects valid concept vectors** (sadness, partially joy) and **under-rejects confounded vectors** (admiration). Correcting this requires per-emotion layer selection, but doing so *after seeing the sweep results* is a forking-paths violation. The selection rule must be preregistered before any held-out confirmation is examined.

---

## Proposed selection rule (defended before data)

### Principle

For each emotion, the "concept layer" is defined as the layer that maximizes cross-context generalization — measured by **implicit-scenario accuracy** (the test with the strongest signal-to-noise ratio, n=50 per emotion, 10 scenarios + 40 emotion-laden scenarios). This is preferred over intensity-Spearman because the intensity test has a pervasive digit confound (mean |ρ(raw x)| ≈ 0.5–0.8 at every layer) and smaller per-cell n.

### Algorithm (to be run automatically, not by hand)

1. **For each model × emotion:** compute implicit-scenario accuracy at every layer in the model's sweep grid.
2. **Select the layer with the highest accuracy.** If there are ties (within ±0.02), prefer the shallower layer (closer to the input) — this is a weak regularization against overfitting to late-layer noise.
3. **If the best accuracy is < 0.60 (chance = 0.25):** mark the emotion as "no recoverable layer" and report it as a vector-quality failure. Do not force a layer choice.
4. **Report the selected layer and its accuracy as a pre-registration output** before examining held-out confirmation.

### Held-out confirmation (to prevent overfitting the sweep grid)

After the layer is selected by the rule above, the confirmation is:
- **Intensity semantic ρ on the *inverse families only*** (the confound-controlled subset) at the selected layer. Target: ≥ 0.6. This is a separate stimulus family from the implicit scenarios, so the layer was not selected to optimize it.
- **Logit-lens token congruence** at the selected layer. Target: top-5 tokens include ≥2 emotion-congruent English words (not code tokens or multilingual fragments). This is a weaker, more subjective test; reported as descriptive, not gating.

If the selected layer fails the intensity confirmation, the emotion is reported as **layer-sensitive but not robustly concept-validated** — the vector signal exists but is not stable across read-out positions.

### Why this is not p-hacking

- The implicit-scenario test uses **pre-registered, frozen stimuli** (not authored after seeing results). The layer grid is fixed by the model's architecture, not chosen post-hoc.
- The selection rule is **deterministic** (max accuracy, tie-break shallower), not a subjective "best-looking" choice.
- The **confirmation metric** (intensity inverse-family ρ) is on a **different stimulus set** with a **different confound structure** (digit vs. scenario content). Overfitting the layer to both would require a true cross-task generalization, which is the definition of a concept.
- The **digit confound** is stronger than any layer-tuning benefit: if the vector were truly surface-captured, it would track raw digits at *every* layer, not just one. The layer that maximizes scenario accuracy while also showing ρ(rank) > ρ(raw x) on inverse families is evidence of a genuine concept signal, not optimization artifact.

---

## Changes to HYPOTHESES.md

Add to the H1 section (after the 2026-06-14 C2 amendment):

- **Layer selection for C2 concept validation (2026-07-12 amendment):** The H1 concept-validation suite (logit-lens, implicit scenarios, numerical intensity) is evaluated at a **per-emotion layer** selected by the rule above, rather than the uniform ~2/3-depth convention. The rule is: (1) maximize implicit-scenario accuracy across the layer sweep, (2) tie-break shallower, (3) require ≥0.60 accuracy, (4) confirm on intensity inverse-family Spearman. If an emotion fails step 3 or 4, it is reported as "not validated as a concept vector" and is excluded from H2/H3/H7 steering claims unless a construction fix (e.g., residualization) recovers it.
- The **H1 probe metric** (best-layer test AUC) remains unchanged; the probe layer is still selected by held-out AUC on the story-activation training set, not by the C2 sweep. The C2 layer and the H1 probe layer may differ, and this is reported.
- **Loathing** is grandfathered: it passed the uniform-layer convention, so no sweep-based selection is needed. This is a descriptive convenience, not a hypothesis test.

---

## Expected outcomes under the new rule

| emotion | Llama | Qwen | Gemma | verdict |
|---|---|---|---|---|
| admiration | FAIL (all layers <0.60) | FAIL | FAIL | **Vector-quality failure** → residualization experiment |
| joy | L16–18 (+0.71–0.77) | L14 or L24–26 (+0.62–0.89) | L? (max +0.56) | **Layer-recoverable on 2/3; Gemma uncertain** |
| loathing | L21 (already passing) | L19 (already passing) | L28 (already passing) | **No change needed** |
| sadness | L16 (+0.73) | L14–18 (+0.94–0.96) | L21–22 or L39–40 (+0.71–0.96) | **Layer-recoverable on all 3** |

Note: these are the *expected* selections based on the already-run sweep; the actual amendment is about the *rule*, not these specific numbers. The rule will be applied to the sweep data once, the results locked, and then held-out confirmation run.

---

## Residualization fork for admiration

Since admiration fails at every layer on all three models, and the failure pattern is consistent with the digit-confound hypothesis (projection tracks raw number against semantic rank), the next step is to test whether **text residualization** removes the surface confound and recovers a concept vector. This is a separate pre-registered experiment, not a layer-selection issue. See `plans/residualization-admiration.md` (to be drafted if this amendment is approved).

---

## First action if approved

1. PI approves this amendment in HYPOTHESES.md (dated block at bottom).
2. Run the selection algorithm on the existing sweep outputs (no new GPU time — data already collected).
3. Lock the selected layers per model × emotion in a new file (e.g., `configs/vector_validation/layers.yaml`).
4. Re-run the intensity inverse-family test at the selected layers (GPU: ~30 min per model, can be batched).
5. Re-run logit-lens at selected layers (GPU: ~15 min per model).
6. Push updated C2 reports to HF.
7. If sadness/joy pass held-out confirmation, proceed with H2/H3/H7 using the per-emotion layer for steering vectors (with the C2-validated layer as the steering target layer).
8. If admiration still fails, launch the residualization experiment.
