# Design note: scaling C2 concept validation to an expanded emotion set (~20)

**Date:** 2026-07-12
**Status:** Placeholder / design intent. NOT a pre-registration. To be
converted into a full pre-registered amendment before any expanded-set
data is collected. The locked four-emotion set ({admiration, joy,
loathing, sadness} + neutral, 2026-06-12 amendment) and the 2026-07-12
per-emotion layer-selection rule are unchanged by this note.

**Context:** The team intends to expand the emotion set for the larger
models (Meeting 2026-07-07). The 2026-07-12 layer-selection rule was
designed for 4 emotions and does not scale as-is. This note records the
agreed design direction so it isn't re-derived (or forgotten) later.

---

## Why the 4-emotion rule breaks at ~20

1. **Stimulus authoring wall.** Per emotion: 10 implicit scenarios +
   3 sweep inverse families + 3 fresh confirmation families, all
   hand-authored. ×20 ≈ 120 inverse families + 200 scenarios.
2. **Argmax implicit test degrades.** Chance drops to 0.05 but
   near-neighbor emotions (admiration/awe/gratitude,
   sadness/grief/disappointment) confuse each other; accuracy stops
   meaning "concept present."
3. **Selection multiplicity.** 20 emotions × 3 models × ~20 layers of
   small-n estimates with a max-selection step → spurious peaks.
   Project conventions require multiple-comparisons correction for any
   exploratory probe across > 5 prompts.
4. **Cross-emotion geometry breaks.** 60 per-(model, emotion) layers
   make vector-angle / opposite-pair / values-correlation analyses
   incomparable without a shared space.

## Design for the expanded set

### Stimuli
- LLM-generate candidate scenarios and inverse families against the
  frozen constraints (no emotion-label words; deterministic build);
  human-audit a random sample per emotion; MD5-freeze before any model
  run (as per `configs/stimuli_hashes.yaml` practice).
- Increase intensity-family n from 6 → ≥12 rows per family. The n=6
  Spearman noise is the weakest link already at 4 emotions.
- Generation/audit provenance logged (generator model + prompt + SHA).

### Selection metric
- Replace argmax accuracy with **one-vs-rest AUC** per emotion
  (concept present vs. all else) — scales with class count, does not
  punish emotions for having close relatives.
- Report the full confusion / similarity structure separately as a
  **result** (emotion geometry), not as validation noise.

### Layer search
- **Band-then-layer:** pre-specify depth bands (early / mid / late
  thirds); select the best band by mean smoothed one-vs-rest AUC, then
  the best layer within the band (3-layer moving average, exact ties
  → shallower). No free search over the full grid.
- Keep the 2026-07-12 vector-quality clause: no layer reaching the
  intensity target anywhere in the sweep → vector-quality failure →
  construction fix (residualization), not layer choice.

### Confirmation & statistics
- Fresh MD5-frozen confirmation stimuli at locked layers only (as in
  the 2026-07-12 rule).
- **BH-FDR across the ~20 confirmation tests per model** (per project
  stats conventions: BH-FDR for many contrasts).
- Report effect size + 95% bootstrap CI (n=10_000) per emotion, not
  just pass/fail.

### Geometry
- **Two layer reports per model:** (a) per-emotion validated layer —
  used for validation and steering; (b) one shared reference layer —
  used for all cross-emotion geometry analyses (angles, opposite
  pairs, values-emotions correlations). Every geometry claim states
  which space it lives in.

### Depth profile as a hypothesis (not a nuisance)
- At ~20 emotions, the layer-depth profile is itself a finding.
  Pre-register: *concept-validation depth correlates with emotion
  complexity / appraisal structure* (e.g., social emotions such as
  admiration and gratitude validate deeper than valence primitives
  such as joy and sadness).
- The four-emotion sweep data (2026-07-11/12) is the motivating pilot
  and must be cited as such — it cannot double as confirmatory data
  for this hypothesis.

## Preconditions before converting this into a pre-registration

1. Four-emotion program complete: layer confirmation run, admiration
   residualization resolved (`plans/residualization-admiration.md`).
2. Emotion list chosen and justified (Plutchik coverage? opposite
   pairs? appraisal-theory sampling?) — a decision block of its own.
3. Compute + authoring budget estimated (sweeps at 20 emotions are
   ~5× the story-generation cost of the current run).
4. Dated amendment block in HYPOTHESES.md; this note is superseded by
   that amendment where they differ.
