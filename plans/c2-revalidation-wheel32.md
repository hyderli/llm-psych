# Plan: C2 revalidation + layer selection for the story-wheel32 track (migration amendment draft)

**Date:** 2026-09-06 (draft)
**Status:** Proposed pre-registration amendment. Must merge via PR **before any behavioral steering** on story-wheel32 vectors. Three team decision boxes (D1–D3) below need answers before the freeze; everything else is written to be locked as-is.
**Triggered by:** Decision to formally migrate steering from the 4-emotion `story` track to the 32-emotion `story-wheel32` track (plutchik_wheel_results.pdf, 2026-09-04). The new vectors are cross-emotion centered over 32 emotions, so **no validation status transfers**: the July C2 results, locked layers, and loathing's grandfathered pass all attach to the old vector files and become descriptive-only for the old track. Everything below is measured fresh on wheel32 vectors.

---

## Why this is not a rerun of the July protocol

Three things changed since 2026-07-12, and each forces one deliberate difference from the old rule; the rest is carried over verbatim.

1. **Timing is clean this time.** In July the sweeps had already been inspected when the selection rule was written, which forced the implicit-accuracy-driven selector (the intensity sweep was contaminated as a selector) and a post-hoc disclosure. On wheel32, **no C2 metric has been computed on any vector** — only the nearest-centroid discriminability check in the wheel report, which uses story activations, not the C2 stimuli. We can therefore choose the better selector and freeze *both* stimulus sets before any model run.
2. **Implicit accuracy is ceilinged.** The k=64 correlation run showed implicit accuracy pinned at 0.98–0.99 with essentially no variance across layers, and the wheel report found within-band discriminability flat to ≤0.03. An implicit-driven argmax over a flat, ceilinged curve degenerates into "tie-break shallower picks the shallowest layer." The metric that actually varied — and that catches the admiration failure mode — is the confound-controlled intensity ρ.
3. **The stimulus sets don't cover the new emotions.** The frozen implicit scenarios and intensity families exist for {admiration, joy, loathing, sadness} only. Any steering-set emotion outside that four needs newly authored, frozen stimuli.

## Scope: two tiers (D1)

**Tier A — steering candidates, full C2 + confirmatory status.** Only Tier A emotions can ever enter confirmatory H2/H3/H7/H8 steering claims. Proposed membership, to be finalized in D1: the four continuity emotions {admiration, joy, loathing, sadness} plus the task-mapped candidates for the three behavioral tasks (e.g. anger and/or fear for blackmail, love and/or joy for sycophancy, and the team's pick for reward hacking). Target size 6–8: each emotion beyond the existing four costs ~6 newly authored inverse families (sweep + confirmation) plus scenario items, and the cost check freezes steered-emotion count as the budget's main multiplier.

> **D1 (team):** fix Tier A membership against the H2/H7/H3 emotion-task mapping. Until decided, the doc assumes {admiration, joy, loathing, sadness, anger, love}.

**Tier B — all remaining wheel cells, descriptive only.** Pairwise discriminability (already done in the wheel report) and logit-lens token congruence (cheap, run for all 32). No confirmatory status, no stimulus authoring, no layer lock. Tier B emotions can be promoted later only by a further amendment that runs them through the full Tier A pipeline.

## The suite, test by test

**(1) Logit-lens congruence** — unchanged, descriptive, never gating. All 32 emotions, all swept layers (falls out of the sweep for free).

**(2) Implicit-emotion scenarios** — kept as the *floor*, not the selector. Design change forced by 32 emotions: scoring stays **4-way blocks** to preserve chance = 0.25 and the 0.60 floor's meaning. Each Tier A emotion gets ≥10 scenarios; each scenario is scored by argmax over a fixed 4-candidate block = {target, its wheel opposite, two same-valence distractors from other axes}, with blocks frozen alongside the stimuli. Existing frozen scenarios for the old four emotions are reused where their blocks can be formed; new Tier A emotions get new scenarios under the same constraints (no emotion-label words; LLM-drafted, PI-audited line-by-line, provenance disclosed — the 2026-07-28 procedure, now the standing one).

**(3) Numerical-intensity inverse families** — promoted to the **selector**. Two disjoint sets, both frozen before any wheel32 C2 model run:

- **Sweep set** (selection): ≥3 inverse families per Tier A emotion, n=12 x-values per family (the 2026-07-28 improvement), plus 3 neutral flat-control families. May recycle the June-2026 templates for the old four emotions (they are spent for confirmation but valid for description/selection).
- **Confirmation set** (the confirmatory test): ≥3 **new** inverse families per Tier A emotion, n=12, disjoint by family name, sentence, and digit-stripped template from *both* prior sets and from the sweep set — `tests/test_confirmation_stimuli.py` extended to enforce all three exclusions plus the `_plural_after_x` guard. Because both sets are frozen up front, the confirmation set is held out by construction — no post-hoc timing disclosure needed this time.

Both sets MD5-registered in `configs/stimuli_hashes.yaml` before any model run.

## Layer-selection rule (deterministic, per model × emotion)

0. **Grid.** All layers with wheel32 vectors. Before the sweep, close the wheel report's open item: **extend extraction below half depth for Qwen (best layer currently at the L14 window edge) and Gemma (L23, third from edge)** — corpora exist, this is extraction-only and cheap (D3). If not extended, an endpoint selection is flagged "boundary — not established as a maximum" and carries that caveat into every downstream use.
1. Compute inverse-family mean ρ(rank) on the **sweep set** at every grid layer.
2. Smooth with a 3-layer moving average; select the argmax; exact ties break shallower.
3. **Floor:** implicit-scenario accuracy (4-way blocks) ≥ 0.60 at the selected layer. If it fails there but ≥ 0.60 within ±2 layers, select the nearest passing layer (ties shallower); if no layer passes, report "no recoverable layer."
4. **Vector-quality clause (unchanged in substance):** if no grid layer reaches sweep-set mean ρ ≥ 0.6, the emotion is a vector-quality failure → routed to residualization, not to layer choice.
5. Locked layers written to `configs/vector_validation/layers_wheel32.yaml` (new file; the old `layers.yaml` stays frozen as the old track's record).

**No grandfathering.** Loathing re-runs like everything else; its July pass belongs to different vectors.

> **D2 (team):** confirm the selector swap (intensity-primary, implicit-as-floor). The July rule inverted this only because the intensity sweep was already observed then; that constraint is gone, and keeping an argmax over a ceilinged flat curve would make the tie-break do the selecting. If the team prefers strict continuity, the fallback is the verbatim 2026-07-12 rule plus a pre-registered tie-radius (all layers within 0.05 smoothed accuracy of the max count as tied → among them pick max sweep-ρ), which amounts to the same thing with extra steps.

> **D3 (team):** approve the below-half-depth extraction extension for Qwen + Gemma before the sweep (~1 short GPU pass; reuses frozen corpora, no new generation).

## Pass criteria and consequences (unchanged from 2026-07-12/28)

Confirmation-set inverse-family mean ρ(rank) ≥ 0.6 at the locked layer → **concept-validated**; otherwise "layer-sensitive but not robustly concept-validated." Only concept-validated (model × emotion) cells enter confirmatory steering claims; VQ failures route to residualization and re-enter only by clearing this same confirmation set at a layer chosen by this same rule. H1's probe metric and falsifier are untouched (probes re-fit on wheel32 activations as part of the same pod pass; probe layer may differ from C2 layer, both reported). C2 remains a validation family, separate from the steering FDR families.

## Interaction with the J-space program

The wheel32 decomposition (feat/jspace-wheel32-decomposition) runs at **all** grid layers, so layer locking here just indexes into it — no re-decomposition. H8's gates G1–G3 are then filled from the wheel32 decomposition **at the layers this amendment locks**, for the Tier A cells that pass confirmation. The phase-1 J-fraction × C2 correlation is re-run on wheel32 with the intensity sweep values (32-emotion Tier B cells contribute discriminability instead), with the standing non-independence caveat.

## Execution order and cost

1. Freeze Tier A (D1) → author + freeze both stimulus sets and scenario blocks (≈1–2 days authoring/audit; no compute).
2. (D3) Extraction extension pass for Qwen/Gemma lower layers (~1 GPU-hour).
3. Sweep pass: C2 suite, Tier A × grid layers × 3 models (July's sweep took ~hours; estimate $5–15 on a 4090).
4. Apply the rule once → lock `layers_wheel32.yaml` → PR.
5. Confirmation pass at locked layers only (~1 GPU-hour, ~$1).
6. Fill H8 gates from the wheel32 decomposition at locked layers; then the H8 PR; then steering.

Steps 1–2 can run in parallel with the already-planned decomposition pod session; step 3 must follow the freezes.

## Reporting rules

Old-track C2 results are reported as the old track's record, never numerically compared with wheel32 values (different centering, different stimulus n). Every ρ states its set (sweep vs confirmation) and n. The wheel report's within-band flatness is cited as prior context for why single-layer optima are weakly identified; locked layers are a reproducibility device, not a claim that the optimum is sharp.
