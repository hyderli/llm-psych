# Plan: Plutchik-wheel emotion-set expansion (24 cells + 8 dyads)

**Date:** 2026-08-28
**Status:** Proposed. Supersedes `plans/emotion-set-expansion-design.md`
(2026-07-12 placeholder) where the two differ. Contains a draft
HYPOTHESES.md amendment block (§13) — **not yet merged, no wheel data may
be collected until it is.**
**Decision inputs:** PI decision 2026-08-28 — expand to the full wheel,
24 derived cells plus a dyad arm, and proceed without waiting on the
four-emotion confirmation run or the admiration residualization result.
§11 records the risk that decision accepts and the mitigations that make
it survivable.

---

## 1. Scope: 33 corpora

**24 derived cells** = 8 Plutchik axes x 3 intensity rings.

| axis | high ring | middle ring | low ring | opposite axis |
|---|---|---|---|---|
| joy | ecstasy | joy | serenity | sadness |
| trust | admiration | trust | acceptance | disgust |
| fear | terror | fear | apprehension | anger |
| surprise | amazement | surprise | distraction | anticipation |
| sadness | grief | sadness | pensiveness | joy |
| disgust | loathing | disgust | boredom | trust |
| anger | rage | anger | annoyance | fear |
| anticipation | vigilance | anticipation | interest | surprise |

**8 primary dyads**, derived independently and then tested against their
predicted compositions (§9, H11):

| dyad | predicted composition |
|---|---|
| love | joy + trust |
| submission | trust + fear |
| awe | fear + surprise |
| disapproval | surprise + sadness |
| remorse | sadness + disgust |
| contempt | disgust + anger |
| aggressiveness | anger + anticipation |
| optimism | anticipation + joy |

Plus **neutral** as the reference corpus. Total: **33 story corpora per
model**, 32 emotion vectors per model.

The dyads are derived from their own story corpora on equal footing with
the 24 — otherwise H11 is circular. They are *not* used to define any
axis or ring.

### Why 24 and not 8

Eight axis vectors would just be a bigger emotion set. The ring structure
is what makes the wheel a *falsifiable* object: Plutchik's claim is that
ecstasy/joy/serenity differ in degree along one dimension, not in kind.
That is a testable geometric prediction (H9) and it doubles as a
validation signal — a set of vectors that fails it is telling you the
derivation is picking up lexical identity rather than affective content.

### A ring inconsistency this fixes

The locked four are not on one ring: **admiration and loathing are
high-ring cells; joy and sadness are middle-ring.** So the existing
opposite-pair comparison confounds axis with intensity ring. The wheel
design removes that confound, and the four reappear as 4 of the 24 cells
(with different vectors — see §5).

---

## 2. What this study is, relative to the locked pre-registration

The wheel run is an **additional derivation track**, not a replacement.
The confirmatory four-emotion analyses (H1-H3, H7) stay bound to the
frozen four-emotion derivation in `steering_vectors/<model>-story/`.
Nothing in this plan re-runs, re-fits, or overwrites them.

---

## 3. The centering consequence (read before writing any code)

`configs/derivation/story.yaml` defines

```
v_e = mean_e(pooled_story_acts) - mean_over_emotions(mean_e(pooled_story_acts))
```

**The grand mean is taken over the emotion set.** Going from 4 to 32
emotions changes it, so *every* vector changes — including joy, sadness,
admiration and loathing. Three consequences:

1. The wheel vectors go to a **new namespace**,
   `steering_vectors/<model>-story-wheel32/`, and a new HF dataset path.
   `<model>-story/` is never touched (project rule: primary-model
   steering vectors are never regenerated — stochastic story generation
   would invalidate C2 validation and the locked layers).
2. Locked layers from the 2026-07-12 rule **do not transfer**. The wheel
   track selects its own layers under the rule in §7.
3. `cos(v_wheel_joy, v_four_joy)` and the same for sadness / admiration /
   loathing is a free **construction-stability check**. Report it. If the
   four-emotion and 32-emotion versions of joy are near-orthogonal, the
   derivation is dominated by the contrast set rather than the emotion,
   and that is a finding that changes how every earlier number is read.

Neutral-PC projection is refit on the expanded neutral corpus
(`project_out.var_threshold: 0.5` unchanged).

---

## 4. Label scheme and configs

Integer labels 1-20 are taken (`configs/emotion/EMOTION_LABELS.md`), and
nine wheel cell names collide with existing config files (`joy`,
`sadness`, `admiration`, `loathing`, `fear`, `anger`, `disgust`,
`surprise`, `contempt`). The legacy configs are a mixed Ekman / Wilcox /
primary-9 set with different stimuli — reusing them would silently mix
frameworks.

**Therefore:**

- New directory `configs/emotion_wheel/`, one YAML per cell, **generated**
  from a single `configs/wheel.yaml` spec by
  `scripts/build_wheel_configs.py`. Do not hand-author 33 files.
- New label namespace, no collisions with 1-20:
  `label = 100 + 10*axis_index + ring` where `axis_index` 1-8 in the §1
  table order and `ring` 1=high, 2=middle, 3=low.
  So 111=ecstasy, 112=joy, 113=serenity, ..., 183=interest.
  Dyads 191-198 in the §1 table order.
- Each cell config carries `axis`, `ring`, `opposite`, and (dyads only)
  `components`, so the geometry analyses read structure from config
  rather than a hard-coded table.
- `EMOTION_LABELS.md` gains a wheel section; the legacy tables stay as
  they are and are marked "not part of the wheel track".

Note `contempt`: Ekman legacy label 16 **and** Plutchik dyad 196. Two
different constructs with two different stimulus sets. The generated
config must not inherit the legacy file.

---

## 5. Stimuli

Per cell (32 cells + neutral):

| set | per cell | total | current |
|---|---|---|---|
| implicit scenarios | 10 | 330 | 50 |
| intensity templates (exploratory) | 3 inverse families x 12 rows = 36 | 1,188 | 156 |
| intensity confirmation (held out) | 3 fresh inverse families x 12 rows = 36 | 1,188 | 144 |

Rules, carried from the 2026-07-12 design note and unchanged:

- **LLM-generated against the frozen constraints** in
  `docs/methods.md` (no emotion-label words, deterministic build),
  **human-audited** on a random sample per cell, **MD5-frozen** in
  `configs/stimuli_hashes.yaml` before any model run.
- Generation provenance logged: generator model + SHA, prompt verbatim,
  temperature, seed.
- Intensity families are **inverse only** for the headline metric — the
  increasing families cannot separate meaning from the digit
  (2026-06-14 finding). n raised 6 -> 12 rows per family; the n=6
  Spearman was already the weakest link at four emotions.
- Near-neighbour authoring hazard: at 24 cells the generator will happily
  write near-identical scenarios for joy/serenity or fear/apprehension.
  The audit sample must explicitly check **ring discriminability**, and
  the audit rubric records, per cell, whether a human rater can assign
  the intended ring. Cells that fail human ring-assignment are reported
  as such rather than quietly dropped.
- Story corpora reuse the existing frozen `story_topics.txt` (56 topics,
  7 stories/topic = 392 stories/cell) so topic distribution is identical
  across all 33 corpora and cannot separate them.

**Free gate closure:** run the already-frozen 144-row four-emotion
confirmation set (`intensity_confirmation.jsonl`, MD5
`36be3dc0...`) on the four-emotion vectors in the same pod session. It
costs minutes, and it closes the outstanding 2026-07-28 confirmatory
test that the wheel decision otherwise leaves dangling.

---

## 6. Derivation arms

Two arms from the **same** stored activations — residualization is a
linear post-hoc operation, so the second arm costs no extra generation:

- `<model>-story-wheel32/` — standard construction.
- `<model>-story-wheel32-resid/` — text-residualized per
  `plans/residualization-admiration.md` (per-layer OLS against pooled
  story text embedding; store beta and R^2).

This is the mitigation for proceeding before the admiration
residualization verdict is in: whatever that verdict turns out to be, the
wheel run already contains both arms and the comparison is available
without a second GPU campaign. Pre-specify that the **standard arm is
primary**; the residualized arm is a pre-registered secondary reported
side-by-side.

---

## 7. Layer selection at 33 cells

The 2026-07-12 per-emotion rule does not scale (free max-selection over
~20 layers x 32 emotions x 3 models). Replace with:

- **Selection metric:** one-vs-rest AUC per cell (concept present vs all
  other cells + neutral), not argmax accuracy. Argmax chance drops to
  ~0.03 and near-neighbour confusion makes it stop meaning "concept
  present".
- **Band-then-layer:** pre-specify depth bands (early / middle / late
  thirds). Choose the band by mean smoothed one-vs-rest AUC across cells,
  then the layer within the band by a 3-layer moving average; exact ties
  resolve shallower. No free search over the full grid.
- **Two layer reports per model, stated on every claim:**
  (a) per-cell validated layer — used for validation and any steering;
  (b) **one shared reference layer** — used for *all* cross-emotion
  geometry (§9). Angles between vectors read at different layers are not
  comparable, and the entire wheel-geometry story lives in (b).
- Vector-quality clause retained: a cell reaching the intensity target at
  no layer in the sweep is a **construction failure**, reported as such,
  not rescued by layer choice.

---

## 8. Statistics

- Confirmation family: 32 cells x 3 models = 96 tests. **BH-FDR across
  the 32 tests within each model**, models reported separately (project
  convention: BH-FDR for many contrasts).
- Effect size + 95% bootstrap CI (n=10,000) per cell, not pass/fail.
- Geometry tests (§9) are their own FDR family, declared separately.
- Pilot-vs-scale discipline unchanged: no steering claim from the wheel
  track at n < 200.

---

## 9. What the wheel actually buys — the pre-registered geometry claims

These are the reason to do 24 rather than 8, and they are the publishable
core. All are evaluated at the **shared reference layer**.

**H9 — Ring structure is magnitude, not direction.**
Within an axis, the three ring vectors are near-collinear and ordered by
norm: `cos(v_high, v_low)` is high (pre-register >= 0.7 as the
"supported" threshold), and `||v_high|| > ||v_middle|| > ||v_low||`.
Falsified if ring cells are near-orthogonal — which would say the model
represents serenity and ecstasy as different concepts, not different
intensities, and is an equally interesting result.

**H10 — Opposite axes are antipodal.**
For the four opposite pairs (joy/sadness, trust/disgust, fear/anger,
surprise/anticipation), `cos(v_axis, v_opposite)` is more negative than
for matched non-opposite pairs at the same ring. Tested at the middle
ring to avoid the ring confound noted in §1.

**H11 — Dyads compose.**
For each of the 8 dyads, the independently derived `v_dyad` is closer to
`normalize(v_a + v_b)` than to `normalize(v_a + v_c)` for the 7
non-component axes c. This is the strongest available claim: the wheel
makes a compositional prediction and it has not been tested in an LLM.

Supporting, non-confirmatory: cross-model Procrustes alignment of the
32-vector configuration (is the wheel geometry model-invariant?), and the
depth profile as a hypothesis in its own right — social/appraisal-complex
cells validating deeper than valence primitives, with the four-emotion
sweep cited as motivating pilot only.

---

## 10. Controls (non-optional)

**The lexical-geometry null.** The wheel is a structure over *words*. The
first reviewer objection will be that any recovered geometry is the
geometry of the emotion vocabulary, not of the model's representations.
Pre-register the null and report every §9 result as a residual over it:

- (a) angle matrix from static embeddings of the 32 labels;
- (b) angle matrix from pooled text embeddings of the generated story
  corpora themselves.

If the ring collinearity and the opposite-pair antipodality are predicted
by (a) or (b), the study measured a dictionary. Same machinery as the
residualization arm (§6), so it is cheap to add.

**Discriminability matrix.** At 32 near-neighbour cells, distinctness is
a result, not an assumption. Report the full 32x32 pairwise one-vs-one
probe AUC. Expect collapse in some cells (serenity/joy,
apprehension/fear, boredom/neutral). Collapsed cells are a finding about
the model's affective granularity, and they must be reported before any
geometry claim that uses those cells.

**Random-vector and shuffled-label controls** as in the four-emotion
track. Non-negotiable.

---

## 11. Risk accepted by going straight to the wheel

The PI decision (2026-08-28) is to proceed without the four-emotion
confirmation run and without the admiration residualization verdict.
The concrete risk: **admiration is a construction failure on all three
primaries** (2026-07-14 status; "number-captured" at conceptual layers).
Admiration is a high-ring cell, and the wheel adds seven more high-ring
cells built the same way. If the failure is a property of high-ring
construction rather than of admiration specifically, up to 8 of 24 cells
inherit it.

Mitigations, all cheap, all built into this plan:

1. **Both derivation arms from the start** (§6) — the residualization
   comparison arrives with the wheel data instead of requiring a second
   campaign.
2. **Stop-check after model 1.** Run Llama 3.1 8B first, complete. If
   more than 8 of 24 cells fail the intensity criterion at every layer,
   **halt** before models 2 and 3 and reconvene on construction. Written
   into `scripts/run_wheel.sh` as an explicit gate with a printed
   verdict, not a judgement call at 2am.
3. **Ring-stratified failure report** — report failures broken down by
   ring, so "high-ring cells fail" is visible immediately rather than
   averaged away.
4. The free four-emotion confirmation run (§5) rides along in the same
   session, so that gate closes anyway.

---

## 12. Cost and authoring estimate

**Compute is not the constraint.** Per model: 33 corpora x 392 stories x
200 max_new_tokens ~ 2.6M generated tokens, plus extraction forward
passes and the C2 sweeps (which scale with *stimulus* count, ~2,700 rows
x ~20 layers, not with vector count).

| item | estimate |
|---|---|
| story generation + extraction, per model | ~3-5 GPU-h |
| C2 suite + layer sweeps, per model | ~1-2 GPU-h |
| 3 primaries | ~12-21 GPU-h |
| at $0.34/h Community Cloud | **~$4-7** |
| with 3x contingency | **~$20** |

Against the $150 cap this is comfortable. **Caveat: the per-emotion
story-pipeline wall-clock was never recorded** (the runbook's cost table
covers the old CAA prompt path, ~700 prompts / 5 min). Instrument the
first cell of the Llama run and revise this table before booking the full
campaign.

**Storage is the real infrastructure constraint.** ~8x the current
activation volume per model. Pod disk >= 100 GB, and keep the existing
free-the-HF-cache-between-models step from `run_primaries.sh` (the
disk-full lesson of 2026-06-14).

**The actual costs:**

| item | estimate |
|---|---|
| stimulus generation (API, judge/generator models) | ~2,700 rows, low $ but log provenance |
| **human audit** | ~33 cells x sample; the binding cost, days not dollars |
| statistical multiplicity | 96 confirmation tests + geometry family |

---

## 13. Draft HYPOTHESES.md amendment block

To be reviewed, dated on merge, and appended to the Amendments section.
Not authoritative until merged.

```markdown
### 2026-08-28 — Plutchik-wheel emotion-set expansion (H9-H11, exploratory track)

**Decision.** Add a second, parallel emotion-vector derivation track
covering the full Plutchik wheel: 24 cells (8 axes x 3 intensity rings)
plus 8 primary dyads, 32 emotion corpora plus neutral, on the three
primary models. The dyads are derived independently, on equal footing
with the 24, and are used only as the test set for H11.

**Relation to the locked pre-registration.** This track is ADDITIONAL.
The confirmatory four-emotion analyses (H1, H2, H3, H7) remain bound to
the frozen four-emotion derivation in `steering_vectors/<model>-story/`,
which is not re-run, re-fit, or overwritten. Because the story-method
construction centres on the cross-emotion grand mean, the wheel vectors
are numerically different objects from the four-emotion vectors,
including for joy, sadness, admiration and loathing; they live in
`steering_vectors/<model>-story-wheel32/` and carry their own layer
selection. No claim in this track substitutes for a four-emotion claim.

**Emotion set.** Axes joy, trust, fear, surprise, sadness, disgust,
anger, anticipation; rings high/middle/low per Plutchik; dyads love,
submission, awe, disapproval, remorse, contempt, aggressiveness,
optimism. Labels in a new 111-198 namespace (see EMOTION_LABELS.md);
configs generated from `configs/wheel.yaml`. The Plutchik `contempt`
dyad is a distinct construct from the legacy Ekman `contempt` config
(label 16) and does not inherit its stimuli.

**Stimuli.** 10 implicit scenarios and 3 inverse intensity families
(12 rows each) per cell, plus 3 fresh held-out inverse families per cell
for confirmation. LLM-generated against the frozen constraints in
docs/methods.md, human-audited on a per-cell random sample with an
explicit ring-discriminability check, and MD5-frozen in
configs/stimuli_hashes.yaml before any model run. Story corpora use the
existing frozen topic list, 7 stories per topic per cell.

**Layer selection.** Argmax implicit accuracy is replaced, for this
track only, by one-vs-rest AUC. Layers are chosen band-then-layer:
pre-specified early/middle/late depth bands, band chosen by mean
smoothed one-vs-rest AUC, layer chosen within band by 3-layer moving
average with ties resolved shallower. Two layers are reported per model
— a per-cell validated layer for validation and steering, and one shared
reference layer used for all cross-emotion geometry. Every geometry
claim states which space it is in. The vector-quality clause of the
2026-07-12 amendment is retained: a cell that reaches the intensity
target at no layer is a construction failure, not a layer-choice
problem.

**Derivation arms.** Two arms are derived from the same stored
activations: the standard construction (PRIMARY) and a text-residualized
construction (SECONDARY, per plans/residualization-admiration.md),
reported side by side.

**H9 (ring structure).** Within an axis, ring vectors are near-collinear
and norm-ordered: cos(v_high, v_low) >= 0.7 and
||v_high|| > ||v_middle|| > ||v_low||, at the shared reference layer.

**H10 (opposite antipodality).** For the four opposite axis pairs,
cos(v_axis, v_opposite) at the middle ring is more negative than for
matched non-opposite pairs at the same ring.

**H11 (dyad composition).** For each of the 8 dyads, the independently
derived v_dyad has higher cosine to normalize(v_a + v_b), its Plutchik
components, than to normalize(v_a + v_c) for any of the 7 non-component
axes c.

**Controls.** All three hypotheses are reported as residuals over a
lexical-geometry null computed two ways: static embeddings of the 32
labels, and pooled text embeddings of the generated story corpora. A
full 32x32 one-vs-one probe-AUC discriminability matrix is reported
before any geometry claim; cells that fail human ring-assignment in the
stimulus audit or collapse in the discriminability matrix are reported
as such and flagged wherever they enter a geometry test. Random-vector
and shuffled-label controls as in the four-emotion track.

**Statistics.** Confirmation tests: BH-FDR across the 32 cells within
each model, models reported separately. Geometry tests H9-H11 form a
separate declared FDR family. Effect size and 95% bootstrap CI
(n = 10,000) for every reported cell. No steering claim from this track
at n < 200.

**Stopping rule.** Llama 3.1 8B runs first and complete. If more than 8
of the 24 ring cells fail the intensity criterion at every layer in the
sweep, the campaign halts before the remaining two models and the
construction is reconsidered.

**Status.** Exploratory-confirmatory: H9-H11 are pre-registered before
data collection, but the track as a whole is secondary to the
four-emotion confirmatory program and is reported as such.
```

---

## 14. Execution order

0. Merge this plan; merge the §13 amendment. **No wheel data before
   both.** (Per CLAUDE.md, start a fresh session on step 1 reading only
   this file.)
1. `configs/wheel.yaml` + `scripts/build_wheel_configs.py` -> 33 configs;
   EMOTION_LABELS.md wheel section; tests for label uniqueness and for
   no-inheritance from legacy configs.
2. Stimulus generation + human audit + MD5 freeze. Longest pole.
3. `scripts/run_wheel.sh` — story pipeline -> extraction -> both
   derivation arms -> one-vs-rest AUC layer selection -> C2 suite +
   sweeps -> push to HF under the wheel namespace. Includes the
   model-1 stop-check gate and the ride-along four-emotion confirmation
   run.
4. Llama 3.1 8B, complete, stop-check, verdict written to
   `results/wheel32/model1_gate.md`.
5. Qwen 2.5 7B, Gemma 2 9B.
6. Discriminability matrix, lexical-geometry null, then H9-H11.
7. Only afterwards: extend the J-space decomposition to the wheel — and
   note that the null-calibration problem recorded in
   `jspace-measurement-validity.md` applies to every new J-fraction
   computed here. Do not report wheel J-fractions without an empirical
   null.

---

## 15. Open questions for the team

1. **Ring naming in outputs.** Report Plutchik lexemes (ecstasy,
   serenity) or axis+ring (`joy_high`, `joy_low`)? Lexemes are readable
   and invite the lexical-geometry objection; axis+ring is honest about
   what is manipulated. Recommendation: axis+ring in all tables, lexeme
   in prose on first mention.
2. **Does the wheel track get steering at all**, or is it geometry-only?
   Steering 32 cells x n>=200 on two behavioral tasks is a different
   budget conversation. Recommendation: geometry-only for now; steering
   restricted to cells that pass both validation and discriminability.
3. **Third model.** Gemma 2 9B admiration already fails hard
   (intensity_rho -0.66). If Gemma fails broadly at high ring, is the
   wheel track reported on two models?
4. **H8 interaction.** The workspace amendment must merge before any
   steering run; if the wheel stays geometry-only there is no conflict,
   but a wheel J-space analysis inherits gates G1/G2.
