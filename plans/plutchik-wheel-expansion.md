# Plan: Plutchik-wheel emotion-set expansion (24 cells + 8 dyads)

**Date:** 2026-08-28
**Status:** Proposed. Supersedes `plans/emotion-set-expansion-design.md`
(2026-07-12 placeholder) where the two differ. Contains a draft
HYPOTHESES.md amendment block (§13) — **not yet merged, no wheel data may
be collected until it is.**
**Decision inputs:** PI decision 2026-08-28 — expand to the full wheel,
24 derived cells plus a dyad arm, and proceed without waiting on the
three-emotion confirmation run (joy/loathing/sadness) or the admiration
residualization result.
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
(with different vectors — see §3).

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
   `steering_vectors/<model>-story-wheel32-{xc,nr,resid}/` (see §6.2 for
   the three centering parameterizations; `xc` is primary), and a new HF
   dataset path.
   `<model>-story/` is never touched. The written constraints this
   rests on: `configs/vector_validation/layers.yaml` `_locked:` ("do not
   re-run after the confirmation stimuli exist") and
   `results/story_screening/report.md` (re-deriving a corpus requires a
   dated HYPOTHESES.md amendment). Story generation is stochastic, so a
   re-run would invalidate the C2 numbers and the locked layers.
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
**ten of the 33 generated config names collide** with existing files in
`configs/emotion/`: the nine cell names `joy`, `sadness`, `admiration`,
`loathing`, `fear`, `anger`, `disgust`, `surprise`, `contempt`, **plus
`neutral`**. The legacy configs are a mixed Ekman / Wilcox / primary-9
set with different stimuli — reusing them would silently mix frameworks.
The neutral collision matters most: neutral is the reference corpus and
the source of the PC projection, so the wheel track needs its own
neutral config and its own neutral corpus, not the legacy file.

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
  `plans/emotion-set-expansion-design.md` §Stimuli and the 2026-07-28
  HYPOTHESES.md amendment (no emotion-label words, deterministic build;
  mechanically enforced by `tests/test_confirmation_stimuli.py`),
  **human-audited** on a random sample per cell, **MD5-frozen** in
  `configs/stimuli_hashes.yaml` before any model run.
- **This is a documented deviation and the amendment must own it.**
  `docs/methods.md` states that all new stimulus files are
  *hand-authored*, frozen and MD5-locked. Authoring ~2,700 rows by hand
  across 33 cells is not feasible, so the wheel track substitutes
  LLM generation + human audit. Say so explicitly rather than letting
  the two documents silently disagree.
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
- Story corpora reuse the existing frozen `story_topics.txt` (**46**
  topics after comment/blank lines; 7 stories/topic = **322 stories/cell**,
  10,626 stories/model across 33 corpora) so topic distribution is
  identical across all 33 corpora and cannot separate them.

**Free gate closure:** run the already-frozen 144-row confirmation set
(`intensity_confirmation.jsonl`, MD5 `36be3dc0...`) on the four-emotion
vectors in the same pod session. Note its actual coverage: **joy,
loathing, sadness + neutral only, 36 rows each — admiration is excluded**
(vector-quality failure, no locked layer), so it closes the 2026-07-28
confirmatory test for three of the four on Llama and Qwen. On Gemma it
closes two: joy is also a vector-quality failure there (max +0.56, no
locked layer, per `configs/vector_validation/layers.yaml`). It costs
minutes and the gate otherwise stays dangling.

---

## 6. Derivation arms and centering parameterizations

### 6.1 Everything after extraction is free

`activations/<model>-story-wheel32/<cell>.npz` stores one pooled vector
per story per layer. Every downstream choice — how to centre, whether to
residualize, which reference to use — is a linear operation on that
stored array, computed on the Mac at zero GPU cost. **No centering
decision belongs on the pod.** The pod's only job is generation and
extraction, which are parameterization-agnostic.

Corollary, and a footgun: `_discover_emotions` in
`scripts/derive_story_steering_vectors.py` globs `*.npz` and excludes
`neutral.npz`, so **the emotion set — and therefore the grand mean — is
whatever files are in the activation directory.** Deleting or adding one
cell's `.npz` silently changes every other vector. Rule for this track:
the wheel activation directory contains exactly the 33 corpora, always;
failed cells are excluded at *analysis* time by name, never by removing
the file. `scripts/run_wheel.sh` asserts the directory holds exactly 33
`.npz` files before calling derive.

### 6.2 Three centering parameterizations, all derived offline

| tag | vector | role |
|---|---|---|
| `xc` | `v_e = project_out(mean_e − grand_mean_over_emotions, neutral_PCs)` | **PRIMARY** |
| `nr` | `v_e = project_out(mean_e − mean_neutral, neutral_PCs)` | secondary |
| `resid` | `xc` recomputed on text-residualized activations | secondary |

**`xc` (cross-emotion centering) is pre-registered as primary.** It is
the paper's construction, it is what the four-emotion track used, and it
is the only choice that keeps `cos(v_wheel_joy, v_four_joy)` (§3)
interpretable as a construction-stability check rather than a comparison
of two different operations.

`nr` (neutral-referenced) exists because it is required to interpret H9
— see §6.3 — and because neutral is excluded from the grand mean anyway,
so `mean_neutral` is already computed and sitting in the same directory.

`resid` (text-residualized per `plans/residualization-admiration.md`:
per-layer OLS against the pooled story text embedding, storing beta and
R^2) is the mitigation for proceeding before the admiration
residualization verdict is in. Whatever that verdict turns out to be, the
wheel data already contains the arm and the comparison needs no second
GPU campaign.

Output namespaces: `steering_vectors/<model>-story-wheel32-{xc,nr,resid}/`.
All three are derived in the same offline pass and all three are pushed;
storage is small (vectors, not activations).

### 6.3 Which parameterization each hypothesis is tested in

This is not a robustness detail. Cross-emotion centering subtracts a
grand mean that absorbs whatever the 32 cells share — including the
generic "this text is emotional" component — so it is **not neutral with
respect to the geometry claims**:

- **H9 (ring magnitude) is not invariant under `xc`.** The norm ordering
  `||v_high|| > ||v_middle|| > ||v_low||` is measured against a moving
  reference: the grand mean already contains the shared axis content, so
  centering can compress or invert the ring ordering it is supposed to
  detect. **H9 is therefore pre-registered in `nr`**, where the reference
  is a fixed neutral corpus and the norm has a stable meaning, with `xc`
  reported alongside. The collinearity half of H9,
  `cos(v_high, v_low) >= 0.7`, is direction-only and is reported in both.
- **H10 (antipodality) is direction-only and reported in `xc`** (primary),
  `nr` alongside. Note that `xc` mechanically pushes vectors apart —
  subtracting a common mean from a set of similar vectors increases their
  mutual angles — so the pre-registered comparison is against **matched
  non-opposite pairs at the same ring**, not against zero. An absolute
  negative cosine under `xc` is not by itself evidence of antipodality.
- **H11 (dyad composition) is invariant under `xc` only in the average
  form.** With `v = m − g`, the centred average is exactly the average of
  the centred components:
  `(m_a + m_b)/2 − g = ((m_a − g) + (m_b − g))/2`. The centred *sum* is
  not: `m_a + m_b − 2g` carries a second copy of the grand mean.
  Since the test is a cosine, `normalize()` removes the factor of 2 and
  the average and sum forms coincide — **but only if the underlying
  relation is the average.** H11 is therefore stated in the average form,
  tested in `xc` (primary), and reported in `nr` as the check that the
  result is not an artifact of the centering algebra.

Every reported geometry number states its parameterization tag. A claim
that holds in `xc` but not `nr` (or vice versa) is reported as such and
is a finding about the construction, not a number to choose between.

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
- **Layer selection runs on the primary parameterization (`xc`) only.**
  Selecting layers separately per parameterization would make the three
  incomparable and would triple the selection space. `nr` and `resid` are
  read at the layers `xc` selects.

---

## 8. Statistics

- Confirmation family: 32 cells x 3 models = 96 tests. **BH-FDR at
  q = 0.10 across the 32 tests within each model**, models reported
  separately. Note the project convention is "BH-FDR for many models,
  Bonferroni for few primary contrasts" (CLAUDE.md); 32 cells within a
  model is a many-contrasts family, so BH-FDR is the right instrument
  here, and q = 0.10 matches every other FDR family in HYPOTHESES.md.
- Effect size + 95% bootstrap CI (n=10,000) per cell, not pass/fail.
- Geometry tests (§9) are their own FDR family, declared separately.
- Pilot-vs-scale discipline unchanged: no steering claim from the wheel
  track at n < 200.

---

## 9. What the wheel actually buys — the pre-registered geometry claims

These are the reason to do 24 rather than 8, and they are the publishable
core. All are evaluated at the **shared reference layer** (§7), and each
names its centering parameterization (§6.3).

**H9 — Ring structure is magnitude, not direction.** *(norms in `nr`;
collinearity in both `xc` and `nr`)*
Within an axis, the three ring vectors are near-collinear and ordered by
norm: `cos(v_high, v_low) >= 0.7` (pre-registered threshold), and
`||v_high|| > ||v_middle|| > ||v_low||`. The norm claim is tested in the
neutral-referenced parameterization because it is not invariant under
cross-emotion centering (§6.3). Falsified if ring cells are
near-orthogonal — which would say the model represents serenity and
ecstasy as different concepts, not different intensities, and is an
equally interesting result.

**H10 — Opposite axes are antipodal.** *(`xc` primary, `nr` alongside)*
For the four opposite pairs (joy/sadness, trust/disgust, fear/anger,
surprise/anticipation), `cos(v_axis, v_opposite)` is more negative than
for **matched non-opposite pairs at the same ring**. Tested at the middle
ring to avoid the ring confound noted in §1. The matched-pair comparison
is load-bearing: cross-emotion centering mechanically inflates mutual
angles, so an absolute negative cosine proves nothing on its own.

**H11 — Dyads compose.** *(`xc` primary, `nr` as centering-algebra check)*
For each of the 8 dyads, the independently derived `v_dyad` is closer to
`normalize((v_a + v_b)/2)` — the **average** form, which is the one that
survives cross-emotion centering (§6.3) — than to `normalize((v_a +
v_c)/2)` for each of the 6 non-component axes c. This is the strongest
available claim: the wheel makes a compositional prediction and it has
not been tested in an LLM.

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

1. **All three parameterizations from the start** (§6.2) — the
   residualized arm arrives with the wheel data instead of requiring a
   second campaign, and costs no GPU time because centering and
   residualization are offline linear operations on stored activations.
2. **Stop-check after model 1.** Run Llama 3.1 8B first, complete. If
   more than 8 of 24 cells fail the intensity criterion at every layer,
   **halt** before models 2 and 3 and reconvene on construction. Written
   into `scripts/run_wheel.sh` as an explicit gate with a printed
   verdict, not a judgement call at 2am.
3. **Ring-stratified failure report** — report failures broken down by
   ring, so "high-ring cells fail" is visible immediately rather than
   averaged away.
4. The free three-emotion confirmation run (§5) rides along in the same
   session, so that gate closes anyway.

---

## 12. Cost and authoring estimate

**Compute is not the constraint.** Per model: 33 corpora x 322 stories x
200 max_new_tokens ~ 2.1M generated tokens, plus extraction forward
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

**Storage is the real infrastructure constraint.** ~6.6x the current
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
`steering_vectors/<model>-story-wheel32-{xc,nr,resid}/` and carry their
own layer selection. No claim in this track substitutes for a four-emotion claim.

**Emotion set.** Axes joy, trust, fear, surprise, sadness, disgust,
anger, anticipation; rings high/middle/low per Plutchik; dyads love,
submission, awe, disapproval, remorse, contempt, aggressiveness,
optimism. Labels in a new 111-198 namespace (see EMOTION_LABELS.md);
configs generated from `configs/wheel.yaml`. The Plutchik `contempt`
dyad is a distinct construct from the legacy Ekman `contempt` config
(label 16) and does not inherit its stimuli.

**Stimuli.** 10 implicit scenarios and 3 inverse intensity families
(12 rows each) per cell, plus 3 fresh held-out inverse families per cell
for confirmation. LLM-generated against the frozen constraints of
plans/emotion-set-expansion-design.md and the 2026-07-28 amendment,
human-audited on a per-cell random sample with an explicit
ring-discriminability check, and MD5-frozen in
configs/stimuli_hashes.yaml before any model run. This substitutes LLM
generation plus human audit for the hand-authoring convention stated in
docs/methods.md; the deviation is deliberate and is scoped to this
track, on grounds of infeasibility at ~2,700 rows. Story corpora use the
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

**Derivation arms and centering.** Story generation and activation
extraction are parameterization-agnostic; all centering choices are
linear operations on the stored pooled activations and are computed
offline. Three parameterizations are derived from the same activations
and all three are reported:
xc = project_out(mean_e - grand_mean_over_emotions, neutral_PCs), the
paper construction, PRE-REGISTERED AS PRIMARY;
nr = project_out(mean_e - mean_neutral, neutral_PCs), secondary;
resid = xc recomputed on text-residualized activations (per-layer OLS
against the pooled story text embedding), secondary, per
plans/residualization-admiration.md.
Every reported number states its parameterization. A result that holds
under one parameterization and not another is reported as such.

**H9 (ring structure).** Within an axis, ring vectors are near-collinear
and norm-ordered: cos(v_high, v_low) >= 0.7, and
||v_high|| > ||v_middle|| > ||v_low||, at the shared reference layer.
The norm-ordering claim is tested in nr and reported in xc alongside,
because norms are not invariant under cross-emotion centering: the grand
mean absorbs the shared axis content and can compress or invert the
ordering the hypothesis is about. The collinearity claim is
direction-only and is tested in both.

**H10 (opposite antipodality).** For the four opposite axis pairs,
cos(v_axis, v_opposite) at the middle ring is more negative than for
matched non-opposite pairs at the same ring. Tested in xc, reported in
nr alongside. The matched-pair comparator is required: cross-emotion
centering mechanically inflates mutual angles, so an absolute negative
cosine is not evidence for this hypothesis.

**H11 (dyad composition).** For each of the 8 dyads, the independently
derived v_dyad has higher cosine to normalize((v_a + v_b)/2), the
average of its Plutchik components, than to normalize((v_a + v_c)/2)
for any of the 6 non-component axes c (8 axes minus the dyad's own two
components). The average form is pre-registered rather than the sum
because it is the form invariant under cross-emotion centering:
(m_a + m_b)/2 - g equals the average of the centred components, whereas
the centred sum carries a second copy of the grand mean. Tested in xc,
reported in nr as a centering-algebra check.

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
   all three centering parameterizations (offline) -> one-vs-rest AUC
   layer selection on `xc` -> C2 suite +
   sweeps -> push to HF under the wheel namespace. Includes the
   model-1 stop-check gate and the ride-along confirmation run for the
   frozen joy/loathing/sadness set.
4. Llama 3.1 8B, complete, stop-check, verdict written to
   `results/wheel32/model1_gate.md`.
5. Qwen 2.5 7B, Gemma 2 9B.
6. Discriminability matrix, lexical-geometry null, then H9-H11.
7. Only afterwards: extend the J-space decomposition to the wheel. The
   J-fraction has no null as currently computed — candidate atoms are
   selected by the vector's own lens logits and then greedily matched to
   the running residual, so any vector scores non-trivially and the
   procedural floor is unknown (this is recorded in the session project
   memory, not in the repo; it should be written up in
   `plans/j-space-decomposition.md`). Do not report a wheel J-fraction
   without an empirical null from the identical pipeline on norm-matched
   random and element-shuffled vectors.

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
   (headline inverse-family rho -0.60 at L28; max across the sweep
   -0.54, per `configs/vector_validation/layers.yaml`), and joy fails
   there too. If Gemma fails broadly at high ring, is the wheel track
   reported on two models?
4. **H8 interaction.** The workspace amendment must merge before any
   steering run; if the wheel stays geometry-only there is no conflict,
   but a wheel J-space analysis inherits the three unfilled prep-run
   gates in `plans/h8-workspace-steering.md` §"Gates before PR"
   (k=16 vs k=64 stability, digit-span projection, J-fraction x C2
   correlation).
