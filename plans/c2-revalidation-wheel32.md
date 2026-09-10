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

**Tier A — steering candidates, full C2 + confirmatory status.** Only Tier A emotions can ever enter confirmatory H2/H3/H7/H8 steering claims. Membership (fixed in D1, below): the four continuity emotions {admiration, joy, loathing, sadness} plus the task-mapped candidates for the three behavioral tasks (e.g. anger and/or fear for blackmail, love and/or joy for sycophancy, and the team's pick for reward hacking). Target size 6–8: each emotion beyond the existing four costs ~6 newly authored inverse families (sweep + confirmation) plus scenario items, and the cost check freezes steered-emotion count as the budget's main multiplier.

> **D1 — RESOLVED 2026-09-08.** Tier A = **{admiration, joy, loathing, sadness, anger, love, desperation, nervousness}** (eight cells). The four continuity emotions, plus anger and love, plus two of the three Sofroniew-paper cells.
>
> **Calm was dropped after a redundancy check on the derived vectors.** Median-over-layers cosine to its nearest wheel cell: calm~serenity **0.969 / 0.955 / 0.955** (Llama / Qwen / Gemma). For reference, the nearest-neighbour cosine *among the 32 wheel cells themselves* has median 0.80-0.83, and the tightest wheel pair on Llama is sadness~grief at 0.941 — so calm is closer to serenity than any two wheel cells are to each other. Use serenity; record that the paper's "calm" maps onto it.
>
> **Nervousness was kept.** nervousness~apprehension is 0.902 / 0.918 / 0.918 — high, at the 78th-88th percentile of the wheel's own nearest-neighbour distribution, but *below* rage~anger (0.93-0.97) and terror~fear (0.95-0.97), which the project already treats as separate cells. Dropping nervousness for redundancy while keeping rage and anger would be inconsistent.
>
> **Desperation was kept and is the most distinct of the three.** Nearest wheel cell 0.740 / 0.697 / 0.601 — the 19th-28th percentile, i.e. more distinct than roughly three-quarters of the wheel cells are from their own nearest neighbour. Its nearest neighbour is not even stable across models.
>
> **Collinearity note, reportable in its own right.** A median nearest-neighbour cosine of 0.83 across 32 supposedly distinct emotions — *after* cross-emotion centering, which should push them apart — indicates the wheel is not carving 32 directions out of these models. This is consistent with the wheel report's within-band discriminability being flat to ≤0.03, and it is the reason the 32-way confusion matrix above is worth recording.

**Tier B — all remaining wheel cells, descriptive only.** Pairwise discriminability (already done in the wheel report) and logit-lens token congruence (cheap, run for all 32). No confirmatory status, no stimulus authoring, no layer lock. Tier B emotions can be promoted later only by a further amendment that runs them through the full Tier A pipeline.

## The suite, test by test

**(1) Logit-lens congruence** — unchanged, descriptive, never gating. All 32 emotions, all swept layers (falls out of the sweep for free).

**(2) Implicit-emotion scenarios** — kept as the *floor*, not the selector. Design change forced by 32 emotions: scoring stays **4-way blocks** to preserve chance = 0.25 and the 0.60 floor's meaning. Each Tier A emotion gets ≥10 scenarios; each scenario is scored by argmax over a fixed 4-candidate block = {target, its wheel opposite, two same-valence distractors from other axes}, with blocks frozen alongside the stimuli. Existing frozen scenarios for the old four emotions are reused where their blocks can be formed; new Tier A emotions get new scenarios under the same constraints (no emotion-label words; LLM-drafted, PI-audited line-by-line, provenance disclosed — the 2026-07-28 procedure, now the standing one).

> **Gate on the block, report the full ranking (added 2026-09-08).** The 4-way block exists to preserve the pre-registered 0.60 floor's meaning — argmax over all 32 would put chance at 0.031 and silently turn 0.60 into a far harder bar. That is a procedural reason, not a measurement one, and the block discards information the same run already produces. So: the **gate** stays the 4-way block at ≥0.60, and *additionally* every scenario's activation is projected onto **all 32 cell vectors**, with three descriptive outputs recorded — the rank of the correct cell out of 32, the mean reciprocal rank per cell, and the full 32×32 confusion matrix. This is free: the projections are dot products against vectors already on disk, no extra generation, no extra authoring. The confusion matrix is the scientifically interesting half, since it shows whether errors are near-twin confusions along a wheel axis (see the collinearity note under D1). Where the two disagree — a cell clearing 0.60 in its block but ranking poorly out of 32 — the discrepancy is reported and the cell is flagged before it enters any steering claim.

> **Blocks are constructed from wheel geometry only.** Candidates are fixed by the rule above (target, wheel opposite, two same-valence distractors *from other axes*) and never by inspecting the vectors. Selecting distractors by vector similarity would choose the distractors the vectors already separate well and guarantee a pass — the test would be built from its own answer key. Block difficulty (mean candidate-to-candidate cosine) is computed **after** blocks are frozen and reported alongside accuracy as a covariate, so a reader can see that one cell passed on an easy block and another on a hard one, without either having been engineered. Note the "other axes" qualifier already prevents same-axis collisions structurally: anger can never be a distractor for rage, nor terror for fear.

> **Axis assignment for add-on cells.** Desperation and nervousness are not wheel cells and so have no geometric opposite or axis. Assigned here on theory, before any scenario is scored: **nervousness → fear axis** (which excludes apprehension, terror and fear as distractors, resolving its near-twin structurally), **desperation → own axis, opposite optimism** (no wheel cell negates it as directly, and its measured nearest neighbour is unstable across models — terror on Llama and Gemma, anger on Qwen).

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

---

## Amendment log

One entry per change to pre-registered material, dated, with the reason. All
four below were written before any wheel32 C2 result existed; the stimulus set
they govern had not been built, hashed, or run at the time of writing. Working
drafts and derivations are in `plans/c2-stimuli-drafts.md`.

### A1 — 2026-09-09 — inverse-only invariant replaced by a pairing invariant

`tests/test_confirmation_stimuli.py::test_all_emotion_families_are_inverse`
asserts that no emotion family is `increasing`. That is retired.

Reason: the July rationale ("for increasing families rho(rank) == rho(x)
identically, so they cannot separate meaning from digit") holds for a family in
isolation and fails for the set. If every family decreases with x, a probe
reading numeral magnitude with a negative weight scores rho(rank) = +1 on all of
them with no emotional content, and only three impersonal neutral families
stand between that reading and the emotional one. Increasing families are now
admitted **as complements only**, and the diagnostic unit is the pair.

Replaced by `test_every_emotion_family_has_a_complement`, plus checks that
paired families share scenario and denominator, differ in predicate only, and
match in polarity. The invariant is not removed, it is exchanged for the
property that now carries the inference.

### A2 — 2026-09-09 — grid construction and `N_PER_FAMILY`

Grids are per-family and closed under the complement map: decreasing arms
x in {1…12}, increasing arms x in {8…19}, neutrals over the union x in {1…19}.

Reason: with arm A counting k and arm B counting twenty minus k, a shared grid
makes the two arms sample different regions of the same underlying variable —
one on the steep part of the intensity curve, one on the flat part. The
resulting unequal attenuation lands in the symmetric half and is
indistinguishable from magnitude contamination. Closing the grid under the
complement map puts both arms on k in 1…12.

x = 0 is excluded: it is a categorical state (nobody left to ask, nothing in the
folder) rather than the bottom of a continuum, the same objection that governs
the denominator-above-max rule. `N_PER_FAMILY = 12` is unchanged at twelve
values per arm; neutrals carry nineteen.

### A3 — 2026-09-09 — pair statistic computed on `rho_x`, not the gate's `rho_rank`

The pair decomposition is `(rho_A - rho_B) / 2` as the primary statistic and
`(rho_A + rho_B) / 2` as the artifact estimate, with **A always the decreasing
arm**, computed on `rho_x = Spearman(proj, x)`.

Reason: `intensity_rank` decreases with x in an inverse family and increases
with x in its complement, so `rho_rank = -rho_x` on one side of the pair and
`+rho_x` on the other. Computed on `rho_rank` the two halves swap roles exactly
— the emotion signal lands in the symmetric half and the magnitude artifact in
the antisymmetric half — and both remain plausible-looking. The C2 gate itself
continues to use `rho_rank` and is unchanged; the pair analysis reaches past it.

Guard added: for every family, assert `rho_rank == -rho_x` when direction is
decreasing and `+rho_x` when increasing. This catches a flipped `direction`
field or a complement registered with the wrong one, which is the failure most
likely to survive review because the numbers stay plausible either way.

Point-wise companion test: A(x) and B(20 - x) describe the same world state, so
`A(x) - B(20 - x)` is regressed on x with the intercept free (reported as the
frame offset); the slope estimates 2w for a magnitude weight w. A
scenario-matched neutral estimates w directly, so the two should stand at 2:1 —
read as a diagnostic with a tolerance band, not a gate, and not read at all
where the neutral slope is below a stated floor.

### A4 — 2026-09-09 — layer and model selection for the pair statistic

The layer is locked per (emotion, model) on the **sweep** families by the
existing D2 rule. The pair statistic is evaluated on the **confirmation**
families at that locked layer only, and that instance is the result. The full
layer profile is reported descriptively and is never the source of the headline
number.

Reason: three models, all layers and six pairs is several hundred instances of
the statistic. Without a rule fixed in advance, "the antisymmetric half is
large" is unfalsifiable, since some layer will deliver it. This rule adds no new
selection surface, keeping selection and evaluation on the disjoint stimulus
sets the July amendment already separates.

Three conditions without which the rule does not hold in practice.

**The split is at the level of scenario, and the pair is atomic.** Complement
arms describe the same world state, so a layer chosen to maximise one arm is
chosen on the other arm's data. Both arms of a pair must sit on the same side of
the sweep/confirmation split, and no scenario may appear on both sides.

The July split does not cover this and cannot be extended by analogy. Its
guards — verified 2026-09-10 — are family name, sentence and digit-stripped
template skeleton
(`tests/test_confirmation_stimuli.py::test_no_family_name_reuse`,
`::test_no_sentence_reuse`, `::test_no_template_skeleton_reuse`). There is no
scenario-level notion in it, because it was drawn over six single-arm families
before complements existed, and the natural way to add twelve families to such a
split — alternating — is exactly the leak. Disjointness is therefore upgraded to
scenario level, and the confirmation set requires **fresh scenarios**, not fresh
sentences within the sweep scenarios. New guard:
`test_scenarios_disjoint_across_split`.

**The normalisation is locked with the layer.** The point-wise slope and the
ratio diagnostic are absolute-scale statistics, and raw projections are not
comparable across layers or models. Projections are recorded as
`proj / (||h|| * ||v||)` — the cosine between the residual stream at the readout
position and the unit emotion vector — so the quantity is dimensionless and
bounded, and slopes are per-unit-x change in cosine. Identical for the pair and
its neutral, since the ratio compares them directly. The normalisation is part
of what the lock fixes and is recorded with it.

**The ratio floor is a precision condition, not a threshold on w.** The failure
at small w is that the ratio's variance explodes, not that w is small, so the
diagnostic is read only where the **neutral slope's confidence interval excludes
zero** at the pre-specified level. On a clean probe both slopes approach zero,
the neutral interval covers zero, and the diagnostic reports "not applicable, no
detectable leak" rather than a number. This avoids setting an arbitrary
threshold on a quantity whose scale is not known in advance.

### A5 — 2026-09-10 — neutral families are minimum-coupling, not zero-coupling

The six neutral families reached their current form through three rounds of
rejection, each on the same test: a neutral predicate must be arbitrary **with
respect to the outcome**, not merely unrelated to the count.

| round | family | why it was rejected |
|---|---|---|
| 1 | `places_distance` | distance from her flat bears on how bad the application outcome is |
| 1 | (2 neutrals only) | two scenarios cannot certify a null in the other four |
| 2 | `questions_seen` | "seen in a past paper" is preparedness — the construct family 4 measures |
| 2 | `shelters_called_before` | prior calls to the same shelters imply prior episodes of the same need |
| 2 | `contacts_added` | when a contact was added bears on willingness to lend |
| 3 | `contacts_photo` | a contact saved with a photograph is one she knows well |
| 3 | `shelters_boundary` | inside the city boundary bears on whether she can get there tonight |
| 3 | `questions_heading` | many questions under one heading describes a narrow, easier exam |

Reason for recording it: a reviewer seeing only the final six has no way to
know that inertness was tested and iterated rather than assumed, and the
framing below would otherwise appear as an assertion with no history.

**The framing.** Zero coupling is not reachable by wording. Any predicate that
mentions the same twenty objects can bear on the outcome by some path, because
those objects are individuated by their role in the scenario — this is a
property of the design, not of the sentences, and a fourth round would not fix
it. What the rewrites achieved is longer inferential paths: a shared charity
registration number reaches the outcome in several steps where the city boundary
reached it in one.

The neutrals are therefore declared **minimum-coupling, not zero-coupling**, and
the residual is measured by the ratio diagnostic in A4 rather than assumed away.

Stated as an assumption rather than a result: that leakage decreases with
inferential path length is a modelling assumption about how the probe reads the
sentence, plausible but not established here, and it is not evidence for the
neutrals' inertness. The ratio diagnostic is what carries that burden.
