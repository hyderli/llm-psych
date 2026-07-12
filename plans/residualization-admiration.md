# Plan: text residualization for admiration vector (concept-vs-surface rescue)

**Date:** 2026-07-12 (draft, pending layer-selection amendment approval)
**Status:** Proposed. This is a **follow-up experiment**, not an amendment to H1/H2/H3/H7 — it tests whether the story-method construction itself is the problem (surface confound baked into the derivation) or whether the confound is irreducible.
**Triggered by:** C2 sweep showing admiration fails at **every layer** on all three primaries, with a consistent "number-captured" pattern (projection tracks raw digit, not semantic rank). The other three emotions show layer-dependent recovery, suggesting admiration is uniquely confounded in the derivation corpus, not in the model.

---

## What residualization does and why it matters here

The story-method construction generates stories per emotion, then takes token-50-mean activations. If the model's generation process itself introduces a surface confound (e.g., admiration stories consistently describe *quantitatively impressive* feats, so the generated text contains numbers that correlate with the emotion label), the resulting vector captures "number + admiration" rather than "admiration."

**Text residualization** (Srujananjali's research-direction proposal, Meeting 2026-07-07) uses regression to remove the contribution of text embedding vectors from the activation space, leaving a "pure emotion" residual. The idea:

1. For each story activation, regress it against the story's text embedding (e.g., pooled sentence embedding, or mean token embedding).
2. Take the residual: `activation_resid = activation − β · text_embedding`.
3. The residual is orthogonal to the surface text representation by construction.
4. Re-derive the emotion vector from the residualized activations.

If the residualized admiration vector (a) passes the intensity inverse-family test at some layer, and (b) shows cleaner logit-lens tokens, then the original failure was a **derivation confound**, not a "models don't represent admiration" finding. If it still fails, then either:
- The confound is deeper than text (e.g., generation strategy, not just vocabulary), or
- The model's admiration representation is genuinely weaker / more distributed at 7-8B scale.

Either outcome is informative and publishable.

---

## Design

### 1. Residualization method (to be implemented)

Add a `--residualize` flag to `scripts/extract_story_activations.py` or create `scripts/extract_story_activations_residualized.py`:

- **Input:** `activations/<model>-story/<emotion>.npz` (the existing story activations, already extracted).
- **Text representation:** For each story, compute the mean pooled token embedding from the model's own embedding layer (or a lightweight sentence encoder like `all-MiniLM-L6-v2`). Using the model's own embeddings is preferred — it captures the exact vocabulary the model sees.
- **Regression:** Per-layer, per-position, fit `activation ~ text_embedding` via OLS. Store `β` and `R²` (the fraction of activation variance explained by text). The residual is `activation − β · text_embedding`.
- **Output:** `activations/<model>-story-residualized/<emotion>.npz` with the same shape as the input.

### 2. Re-derive vectors

Run `scripts/derive_story_steering_vectors.py` on the residualized activations (or add a `--source residualized` flag), producing `steering_vectors/<model>-story-residualized/`.

### 3. Re-validate

Run the full C2 suite (`logit_lens`, `implicit_scenarios`, `intensity_semantic`) on the residualized vectors, with the **same layer-selection rule** as the non-residualized vectors (max implicit-scenario accuracy, tie-break shallower, confirm on intensity). This keeps the comparison fair — the layer is selected by the same criterion on both constructions.

### 4. Comparison metrics

Report side-by-side:
- Implicit-scenario accuracy: original vs. residualized
- Intensity inverse-family Spearman: original vs. residualized
- Logit-lens top-5 token quality: original vs. residualized
- Text-regression R²: how much variance was "explained away" (descriptive)
- Cosine between original and residualized vector: if they're nearly identical, residualization did nothing; if orthogonal, it completely rebuilt the vector.

### 5. Control: residualize neutral

Also residualize the neutral stories and confirm the neutral vector stays near zero (it should — there's no emotion to remove). If the neutral vector becomes large after residualization, the method is overcorrecting.

---

## Success criterion

- **Rescue:** Residualized admiration passes the C2 suite (implicit accuracy ≥ 0.60, intensity inverse-family ρ ≥ 0.6) on ≥ 2 of the 3 primaries at the layer selected by the standard rule. This proves the original vector was surface-confounded and the concept is recoverable.
- **Partial rescue:** Residualization improves one or two metrics but not all three. Reported as "confound reduced but not eliminated" — suggests the derivation needs deeper changes (e.g., topic-matching, number-balancing in the story generation prompt).
- **No rescue:** Residualization does not improve admiration on any metric. Reported as "admiration representation may be genuinely weak at 7-8B scale" — a real negative result. This is still valuable: it constrains the generalization claim of H1/H2 to {joy, loathing, sadness} and frames admiration as an open question for larger models.

---

## Scope and cost

- **GPU time:** Extracting text embeddings from the story corpus is cheap (~5 min per model on CPU; the stories are already generated). The regression is OLS, trivial. Re-validation is the same as the original C2 suite (~30 min per model). Total: ~2 hours on one 4090 = ~$1.
- **No new stimuli:** Reuses the existing `data/public/story_topics.txt` and `intensity_templates.jsonl`.
- **No new model downloads:** Reuses the already-cached model weights.
- **Code:** One new script (`extract_story_activations_residualized.py`) plus a flag on `derive_story_steering_vectors.py`. ~50 lines.

---

## First action if approved

1. Implement `extract_story_activations_residualized.py` (or `--residualize` flag).
2. Run on one dev model first (e.g., Qwen 0.5B) to check the method doesn't destroy all vectors (including the ones that already pass).
3. If the dev run shows loathing/joy/sadness still pass and admiration improves, run on all three primaries.
4. Push residualized vectors and validation reports to HF under a new subfolder (`steering_vectors/<model>-story-residualized/` and `vector_validation_residualized/`).
5. Update the paper draft with the comparison table.

---

## Relation to the layer-selection amendment

This plan is **independent** of the layer-selection amendment. The layer-selection amendment is needed regardless (sadness and joy need it). Residualization is a **follow-up** for admiration only. They can be approved together or separately. If the layer-selection amendment is approved first, the residualization experiment uses the **same layer-selection rule** on the residualized vectors, so the comparison is apples-to-apples.

