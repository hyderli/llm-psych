# Plan: J-space decomposition of emotion vectors (global-workspace integration, phase 1)

**Date:** 2026-08-26 (draft)
**Status:** Proposed. This is an **analysis-only follow-up** on already-stored activations and vectors — no new pre-registration needed for the descriptive phase. The steering-arm extension (§6) *would* be an amendment (proposed H8) and must go through PR before the H2/H7 steering runs launch.
**Triggered by:** Gurnee, Sofroniew et al. (2026), *"Verbalizable Representations Form a Global Workspace in Language Models"* (Transformer Circuits, July 6 2026). The paper identifies a privileged "J-space" of verbalizable representations that mediates verbal report, directed modulation, and flexible internal reasoning — and its companion release makes the method directly usable on our primaries.

---

## Why this matters for llm-psych

Sofroniew et al. (April 2026) showed emotion concepts causally shift misalignment behaviors. Gurnee et al. (July 2026) showed a verbalizable workspace mediates flexible reasoning and report, and note in their open questions that J-space contents appear tied to "something like emotional reactions" without working out how. Neither paper connects the two. We hold exactly the assets needed to connect them on open weights: validated emotion vectors (loathing everywhere; sadness at locked layers; joy partial), three model families, and a not-yet-run steering pipeline on blackmail/sycophancy.

The phase-1 question: **how much of each emotion vector lives in the workspace, and does that explain our validation results?** Three concrete payoffs:

1. **Layer selection becomes theory-driven.** If our locked per-emotion layers (2026-07-12 amendment) fall inside each model's workspace band (the layer range where J-lens readouts are coherent — L38–92 reindexed in the paper), the ad-hoc selection rule becomes a workspace-band prediction. That is a result, not a patch.
2. **The digit confound may be a J-space object.** Digits are maximally verbalizable tokens, so the pervasive number confound (neutral |ρ| 0.5–0.8 at every layer) plausibly lives *inside* the J-space. Projecting out digit J-lens vectors is a principled variant of residualization — directly comparable to `plans/residualization-admiration.md`, and potentially a cleaner diagnosis of the admiration construction failure.
3. **Sets up the workspace-mediated steering arms (H8)** on the compute we are already committed to spending for H2/H7.

---

## Artifact availability (checked 2026-08-26)

- **Code:** `github.com/anthropics/jacobian-lens` — reference implementation, works on
  HuggingFace decoder transformers generally; `JacobianLens.from_pretrained()` loads
  pre-fitted lenses. Fitting from scratch is dominated by the model's backward pass
  (paper: 1000 sequences × 128 tokens; README notes ~100 prompts is usable).
- **Pre-fitted lenses:** `huggingface.co/neuronpedia/jacobian-lens` — pre-fitted
  J-lenses for 36+ models (~58 GB total, MIT). The folder listing includes
  **Llama 3.1 8B base + instruct, Qwen 2.5 7B instruct, and Gemma 2 9B (both
  variants)** — all three primaries covered with zero fitting cost.
- **Interactive check:** `neuronpedia.org/<modelId>/jlens` hosts a J-lens UI for the
  supported open models — useful for eyeballing readouts before committing code.

**Step 0 verifications (before any analysis):**
- [ ] Confirm which layers each pre-fitted lens covers for our three primaries (the paper reports 25 evenly spaced layers; the release layout may differ).
- [ ] Confirm whether **Qwen 2.5 7B base** is included (needed for the H4 base-vs-instruct extension; if absent, either fit locally with the reference implementation on a 4090 — budget the hours first — or drop Qwen from the base/instruct arm).
- [ ] Sanity-check one pre-fitted lens against the Neuronpedia UI on a known prompt (e.g. the multi-hop "spider legs" example) before trusting it in scripts.
- [ ] Record HF revision SHAs of the lens repo in `run_meta.json` per our reproducibility rule.

**Fallback:** the paper reports the plain logit lens captures much of the workspace structure in mid-to-late layers with lower reliability. We already run logit-lens in C2, so every analysis below has a zero-download fallback arm; report both where they disagree.

---

## Design

### 1. Build the emotion J-lens vocabulary

For each emotion, a small token set: the emotion word and inflections plus 5–10 near synonyms that are single tokens in each tokenizer (checked per model — "admiration" is multi-token in some vocabularies; use the paper's template-lens idea as the multi-token fallback, which is methodologically close to our story-method derivation anyway). Also build the **digit set** (0–9, number words) for §4. Freeze both lists in `data/public/jlens_vocab.yaml` before running.

### 2. J-lens vectors per model

Load the pre-fitted lens matrix J_ℓ per layer; J-lens vectors are the rows of W_U·J_ℓ for the chosen tokens, using each model's own unembedding. Cache to `steering_vectors/<model>-jlens/`.

### 3. Decompose each stored emotion vector

At each swept layer (reusing the C2 sweep grid): solve for a sparse nonnegative combination of k J-lens vectors approximating the emotion vector (gradient pursuit, k ≤ 25, as in the paper §2.3); the fit is the **J-space component**, the rest the **residual**. Report per emotion × model × layer: fraction of vector norm/variance in the J-space, and which tokens carry it (face-valid emotion tokens vs junk).

### 4. Diagnostics

- **Workspace band vs locked layers:** overlay each model's coherent-readout band on the C2 layer sweeps. Prediction: intensity-test recovery layers (sadness L14–L30 etc.) sit inside the band; the old uniform 2/3-depth convention sat at its edge for some models.
- **Digit capture:** projection of each emotion vector onto the digit J-lens set. Prediction: admiration ≫ others; on Gemma, tracks the anti-semantic behavior.
- **J-space residualization arm:** re-run the C2 suite on (a) the J-space component alone, (b) the vector with digit J-lens directions projected out. (b) is the direct comparison to text residualization — same success metrics as `residualization-admiration.md` so the two rescue methods are apples-to-apples.

### 5. Comparison metrics

Side-by-side per emotion × model: C2 implicit accuracy and intensity ρ for {original, J-component, digit-projected, text-residualized} vectors; J-space fraction; top-10 contributing J-lens tokens. One figure: J-space fraction vs C2 pass/fail across the 12 emotion × model cells — if validated vectors are systematically more workspace-loaded, that is the headline descriptive result.

### 6. Extension (H8, requires amendment + PR): workspace-mediated steering

When H2 blackmail / H7 sycophancy run, add two arms at matched norms: J-space component only, and residual only, alongside full-vector and existing controls. Prediction from the workspace account: the J-component carries the behavioral effect; the residual affects at most fluency/style. Either outcome publishable. Amendment must be merged **before** the steering pods launch. (Same infrastructure later serves the report/introspection experiment — revived H5 — and the base-vs-instruct workspace point-of-view test under H4; separate design notes when phase 1 is done.)

---

## Success criteria (phase 1, descriptive)

- **Strong:** J-space fraction and/or digit-capture cleanly separates validated from failed cells (loathing/sadness high emotion-token loading; admiration digit-captured), and locked layers fall inside workspace bands on ≥ 2 of 3 models.
- **Informative negative:** emotion vectors are mostly *outside* the J-space everywhere. This contradicts the naive workspace reading of Sofroniew et al. and sharpens H8 into a genuinely open question — worth reporting either way.
- **Method check:** logit-lens fallback agrees with pre-fitted J-lens on the sign of every headline comparison; disagreements are flagged, not averaged.

---

## Scope and cost

- **GPU: ~$0 for the core.** Lens matrices are downloaded; decomposition is linear algebra on stored activations (Mac M5, CPU/MPS). Only §4's C2 re-validation arm needs a pod (~30 min/model, same shape as residualization re-validation, ~$1–2 total).
- **Disk:** budget ~2–5 GB per model of lens matrices; pull only our layers/models, not the full 58 GB.
- **Code:** `scripts/decompose_jspace.py` (gradient pursuit ~40 lines + plumbing), a loader for the HF lens repo, and a `--source jspace-*` flag on the C2 runner.
- **Ordering:** does not block, and is not blocked by, the confirmation-stimuli pass or text residualization. Natural slot: run in parallel with confirmation; feed results into the residualization comparison table.

---

## First action if approved

1. Step 0 verifications above; record lens repo revision.
2. Freeze `jlens_vocab.yaml`.
3. Dev-model dry run (Qwen 0.5B — fit a lens locally if no pre-fit exists; small model, cheap) to validate the decomposition code end-to-end.
4. Run decomposition + diagnostics on all three primaries from cached activations.
5. Draft the H8 amendment only after seeing phase-1 J-space fractions (if emotion vectors are 2% J-space, the steering-arm design needs rethinking first).
