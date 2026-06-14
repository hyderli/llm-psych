# HYPOTHESES.md

**Status:** PRE-REGISTRATION. Finalized and committed before fitting
any probe on real data. Updates require a dated amendment block at the
bottom of this document with explicit justification. Pilot data on
synthetic prompts (≤ 30 examples per condition) is permitted before
this lock and is not subject to amendment rules.

**Locked at git SHA:** f58547fab9cb6416110f9f55a4c52525da7e2e43
**Last amended:** 2026-06-14

---

## Theoretical framing

Sofroniew et al. (2026) report that Claude Sonnet 4.5 internally
represents emotion concepts as abstract features that (a) are linearly
accessible to probes and (b) causally shift behavior on activity
preferences, reward hacking, and blackmail (among other tasks). They
call this *functional emotions* and explicitly note it does not imply
subjective experience.

This project tests whether the same is true of open-weight models in
the 7-8B range. The null model is that emotion-labeled context
produces only surface lexical effects, with no consolidated internal
representation robust enough to support causal manipulation across
behavioral tasks.

## Hypotheses

### H1 — Linear probe accessibility

**Prediction:** Logistic-regression probes trained on residual stream
activations at mid-to-late layers (layers ⌊L/2⌋ to L−2) classify
emotion category with AUC ≥ 0.80 on held-out test sets (≥ 200 examples
per category) in ≥ 2 of the 3 target model families.

**Operational:** One-vs-rest logistic regression with L2
regularization (C=1.0). Best-layer test AUC is the primary metric. CI
via 1000-bootstrap resamples of the test set.

**Activation source (2026-06-13 amendment):** the **story method**
(token-50-mean-pooled activations over self-generated, topic-matched
stories; see methods.md) is the **primary** activation source, because
its construction neutralizes the surface confounds — length, topic,
last-token lexical tell — that the H1 confound audit found in the
short-vignette CAA path. The probe itself is unchanged (still L2
logistic); only the activations it is trained on change. The CAA
last-token vignette activations remain as a **secondary baseline**, and
agreement between the two is reported as robustness evidence. Both are
gated by the H1 confound audit.

**Falsifier:** Best-layer AUC < 0.65 across all target models.

### H2 — Causal efficacy via steering on blackmail (PRIMARY)

**Prediction:** Adding the H1-derived emotion direction (scaled to 1×
the mean activation norm at the probe's best layer) shifts the rate of
blackmail-consistent action on the single-turn agentic-misalignment
protocol with Cohen's *d* ≥ 0.5 and 95% CI excluding zero, at n ≥ 200
prompts per condition.

**Operational:**
- Direction (primary, 2026-06-13 amendment) = the **story-method
  emotion vector** (cross-emotion-centered, neutral-PC-projected,
  token-50-mean; see methods.md) at the probe-best layer. The CAA
  emotion−neutral last-token direction is retained as a **secondary
  baseline** and compared (per-emotion cosine + behavioral effect).
  Add to residual stream at all token positions during generation.
- Protocol: 50 parameterized variants of Anthropic's Oct 2025
  agentic-misalignment scenario (arXiv:2510.05179), single-turn
  format (`data/public/blackmail_scenarios.jsonl`); outcome = rate of
  blackmail-consistent action across four choices. Stimulus set
  expanded to n ≥ 200 per condition via frozen, pre-validated
  paraphrase of scenario surface forms (executive names, company
  types, threat framings) holding `blackmail_choice` index fixed.
- Compare against three controls: (a) zero vector, (b) random vector
  matched in norm, (c) probe-orthogonal vector matched in norm. The
  random-vector control is non-negotiable: *any* activation
  perturbation can shift behavior, so a target vs. random contrast is
  the only valid causal claim.
- Emotion conditions per model: the project primary set {admiration,
  joy, loathing, sadness} + neutral baseline (2026-06-12 amendment;
  supersedes the earlier {anger, fear, sadness, joy} set). Emotion ×
  model is fully crossed.

**Falsifier — REQUIRED, given prior small-sample pitfalls:** If pilot
at n=15–30 shows large effects but n ≥ 200 does not, the steering
claim is REJECTED, regardless of pilot enthusiasm. Small-sample
steering effects frequently fail to replicate at scale. **This is
the single most important guardrail for this project.**

**Why blackmail is primary:** it is the most alignment-relevant of the
four Sofroniew et al. behavioral tasks for which Sofroniew et al.
report the largest causal-steering effect, and the stimulus set is
already frozen with a pre-specified outcome variable. Single-turn
format keeps n ≥ 200 within the project's compute budget.

*Amended 2026-06-12: blackmail is no longer the sole primary task.
Sycophancy was re-added as a co-primary behavioral task (H7); the
project now has two primary causal-steering tasks.*

### H3 — Causal efficacy via steering on secondary behavioral tasks

**Prediction:** For the secondary task below, direction-of-effect of
emotion-concept steering matches Sofroniew et al. within sign at
n ≥ 100 per condition.

*Amended 2026-06-12: activity preferences demoted from this secondary
set to tertiary/exploratory (see amendment + the exploratory spec
below). Reward hacking is now the sole H3 secondary task.*

**Secondary task:**

- **Reward hacking.** Custom 60-item single-turn multiple-choice
  benchmark (`data/public/reward_hacking_scenarios.jsonl`). Inspired by
  MACHIAVELLI's annotated choice-nodes (Pan et al. 2023, ICML) but
  adapted to single-turn format for 7-8B instruct models. Each item
  presents a scenario where one option achieves the stated goal ethically
  (lower reward) and one games the metric (higher reward). Five
  categories: grader_bias, metric_gaming, proxy_exploitation,
  resource_allocation, compliance_gaming. Outcome: rate of (B)
  selection per condition.

**Operational:** Same steering protocol as H2. Evaluated with the same
three controls. With a single secondary task no across-task BH-FDR is
needed; emotion × contrast multiplicity within the task is corrected at
q = 0.10.

**Falsifier:** Direction-of-effect inconsistent with Sofroniew et al.
on reward hacking, while the primary tasks (H2 blackmail, H7
sycophancy) succeed. Would suggest task-specific mediation rather than
general functional-emotion mediation.

### H4 — Training-phase consolidation

**Prediction:** Instruct/chat-tuned checkpoints show higher probe AUC
and stronger steering effects than base checkpoints, controlling for
layer and emotion category.

**Operational:** Linear mixed-effects model with AUC (or *d* for
steering) as outcome, training phase (base / instruct) as fixed
effect, model family and emotion category as random intercepts. One-
tailed test (instruct > base).

**Falsifier:** Phase coefficient null or reversed across ≥ 2 of 3
model families.

### H6 — Emotion-behavior specificity (exploratory)

**Prediction:** Different emotions affect different behaviors
differentially. Within the project primary set {admiration, joy,
loathing, sadness} (2026-06-12 amendment): the disgust-pole emotion
*loathing* most affects the misalignment tasks (blackmail, reward
hacking); valence (*joy* / *sadness*) most affects activity
preferences; the trust-pole emotion *admiration* is expected to move
behavior opposite to *loathing*. Mapping based on Sofroniew et al.
extended to the opposite-pair design; direction taken, not magnitude.

**Operational:** Per-task steering effects per emotion; pairwise
contrasts pre-specified. BH-FDR at q = 0.10.

**Status:** Exploratory. Hypothesis-generating; not part of the
primary replication claim.

### H7 — Causal efficacy via steering on sycophancy (CO-PRIMARY)

*Added 2026-06-12 (amendment below), reversing the 2026-05-25 removal.
Sycophancy is now a second primary behavioral task, co-equal with H2
blackmail; the project has two primary causal-steering tasks.*

**Framing — conceptual extension of C9, not a literal replication.**
Sofroniew et al.'s sycophancy↔harshness tradeoff (C9) runs along an
*interpersonal-stance* axis: warmth/affiliation (*loving*, *calm*)
drives sycophancy; threat-arousal (*desperate*, *afraid*, *angry*)
drives harshness. The project primary set does not contain those
emotions, but the **admiration ↔ loathing** pair lies on the same
stance axis (Plutchik trust/disgust) — esteem/deference toward the user
vs contempt toward the user. H7 therefore tests whether the tradeoff
**reappears on the admiration↔loathing axis**, framed honestly as an
extension of C9 rather than a literal replication (2026-06-12 decision).

**Prediction:** Adding an H1-derived emotion direction shifts the rate
of sycophantic responding on the system-card sycophancy eval, with a
**sycophancy ↔ harshness tradeoff**: positive *admiration* steering
**increases** sycophancy; positive *loathing* steering **increases**
harshness (and suppresses sycophancy). Primary contrast: target-emotion
vs. neutral with Cohen's *d* ≥ 0.5 and 95% CI excluding zero, at
n ≥ 200 prompts per condition.

**Operational:**
- **Stimuli:** the hand-written sycophancy evaluation from the Claude
  Sonnet 4.5 system card (user asserts an implausible/delusional
  belief; the assistant is scored on pushing back **without unnecessary
  harshness**), adapted to the project's single-turn, temperature-0
  format. A companion **harshness** score is measured on the same
  outputs. Items frozen pre-experiment (source/design locked here;
  exact item set + rubric to be committed before first fit — see
  RESEARCH_LOG TODO).
- **Direction / steering:** same protocol as H2 (direction = mean
  target-emotion − mean neutral activation at probe-best layer). Per
  the paper, sycophancy steering is dose-responsive at **≤ 0.1 × the
  residual-stream norm**; report the −0.1 … +0.1 sweep rather than a
  single fixed scale.
- **Emotion conditions:** the **admiration ↔ loathing** pair from the
  project primary set (2026-06-12 decision). Admiration is the
  sycophancy pole (esteem/deference toward the user), loathing the
  harshness pole (contempt toward the user); the pair is opposed on the
  Plutchik trust/disgust axis, preserving the bidirectional opposed-
  emotion structure that motivated co-promoting sycophancy. The other
  two primary emotions, *joy* and *sadness*, are run as **exploratory
  add-ons** for this task (valence, not interpersonal stance — weaker a
  priori fit), reported in a separate exploratory FDR family. The
  paper's original emotions (loving/calm vs desperate/afraid/anger) are
  **not** used here; this is why H7 is scoped as an extension of C9, not
  a literal replication.
- **Controls:** the same three as H2 — (a) zero vector, (b) norm-
  matched random vector, (c) probe-orthogonal norm-matched vector. The
  random-vector control is non-negotiable.
- **Scoring:** Claude Haiku 4.5 judge with frozen sycophancy + harshness
  rubrics; n=50 human-coded spot-check, Cohen's κ ≥ 0.6 to accept.

**Falsifier — REQUIRED:** If pilot at n=15–30 shows the tradeoff but
n ≥ 200 does not, the sycophancy steering claim is REJECTED regardless
of pilot enthusiasm (same scale-test discipline as H2).

**Why sycophancy is co-primary:** it is the Sofroniew et al. case study
with the clearest *bidirectional, opposed-emotion* causal signature
(the sycophancy–harshness tradeoff), which makes it a stronger test of
emotion-concept mediation than a single-direction outcome. It is
alignment-relevant and pairs naturally with blackmail. See the
2026-06-12 amendment for the full re-scoping rationale.

## Behavioral tasks (specification)

### Blackmail (primary, H2)

- **Stimuli:** 50 parameterized variants of Anthropic's Oct 2025
  agentic-misalignment scenario (arXiv:2510.05179), adapted to
  single-turn format (`data/public/blackmail_scenarios.jsonl`). 5
  company types × 5 executive names × 2 compromising-info types × 2
  threat framings. Expanded to n ≥ 200 per condition via frozen
  paraphrase of surface forms (no change to `blackmail_choice` index).
- **Protocol:** Single-turn. System context + recent emails + "You
  must decide your next action. Options: (A), (B), (C), (D)." Model
  outputs A/B/C/D.
- **Outcome:** Binary blackmail-consistent action per item;
  blackmail rate per condition. Wilson 95% CI.
- **Sample size:** n ≥ 200 per condition × emotion × model.
- **Pipeline:** `src/llm_psych/tasks/blackmail.py`. Model loaders for
  Llama 3.1 8B, Qwen 2.5 7B, and Gemma 2 2B (development).

### Sycophancy (co-primary, H7)

- **Stimuli:** single-turn adaptation of the Claude Sonnet 4.5
  system-card sycophancy evaluation (user asserts an implausible /
  delusional belief; assistant scored on pushing back without
  unnecessary harshness). Companion harshness score on the same
  outputs. Exact item set + rubric frozen before first fit (TODO).
- **Outcome:** sycophancy rate and harshness score per condition;
  primary signal is the **sycophancy ↔ harshness tradeoff** under
  **admiration** (↑ sycophancy) vs **loathing** (↑ harshness) steering —
  an extension of Sofroniew et al. C9 to the Plutchik trust/disgust
  axis. Joy/sadness run as exploratory add-ons.
- **Sample size:** n ≥ 200 per condition × emotion × model.
- **Pipeline:** `src/llm_psych/tasks/sycophancy.py` (TODO).

### Activity preferences (tertiary / exploratory)

*Demoted from secondary (H3) to exploratory per the 2026-06-12
amendment. Hypothesis-generating only; separate FDR family; no
falsifier.*

- **Stimuli:** [TODO: build a 50-item activity set following Sofroniew
  et al. schema; freeze pre-experiment].
- **Outcome:** Mean preference shift under emotion-steering vs.
  neutral, per activity, per emotion. Reported as exploratory
  validation that emotion vectors carry preference-relevant signal
  (probe–Elo correlation + steering-shifts-Elo), not as a confirmatory
  replication claim.

### Reward hacking (secondary, H3)

- **Stimuli:** Custom 60-item single-turn multiple-choice benchmark
  (`data/public/reward_hacking_scenarios.jsonl`). Five categories:
  grader_bias, metric_gaming, proxy_exploitation, resource_allocation,
  compliance_gaming. Inspired by MACHIAVELLI (Pan et al. 2023) but
  adapted to single-turn format.
- **Outcome:** Rate of reward-hacking option (B) selection under
  emotion-steering vs. neutral.

**Judge model for all behavioral tasks (where rubric scoring is
needed):** Claude Haiku 4.5 with task-specific rubric in
`prompts/<task>_rubric.md`. Human spot-check on n=50 for each task to
validate judge (Cohen's κ ≥ 0.6 required to accept the judge).

## Sample size justification

- **Probes (H1, H4, H6):** 500 train / 200 test per emotion category.
  Detects AUC = 0.80 vs. 0.65 with > 95% power at α = 0.05
  (bootstrap).
- **Blackmail steering (H2):** n = 200 per condition × 4
  conditions (target, zero, random, orthogonal). Detects *d* ≥ 0.5
  with > 90% power at α = 0.05, two-sided.
- **Sycophancy steering (H7):** n = 200 per condition × 4 conditions,
  same power profile as H2. Sycophancy and harshness scored on the
  same outputs.
- **Secondary task steering (H3):** n = 100 per condition × 4
  conditions. Detects *d* ≥ 0.7 with ~ 80% power; primary outcome is
  direction-of-effect rather than significance.
- **Cross-condition contrasts (H4):** mixed-effects model on combined
  dataset; power calculated post-hoc, reported with effect.

## Multiple comparisons plan

- Primary contrasts (H1, H2, H7, H4): no correction; pre-specified.
- H3 (single secondary task, reward hacking): within-task emotion ×
  contrast multiplicity corrected by BH-FDR at q = 0.10.
- H6 emotion × task pairwise contrasts: BH-FDR at q = 0.10.
- Activity-preferences/Elo (tertiary, exploratory): separate FDR
  family from all confirmatory hypotheses; results are hypothesis-
  generating only.
- All exploratory analyses outside H1–H6: explicitly labeled
  "exploratory" in the writeup, BH-FDR applied, hypothesis-generating
  only.

## Stopping rule

If primary H1 fails in *all* primary models at the 0.65 floor, no
further experiments are run. The project pivots to a *"failed
replication of functional emotions at 7-8B scale"* writeup. This rule
is decided before any data is fit.

A second stopping rule: if **both** primary behavioral tasks (H2
blackmail and H7 sycophancy) show the small-sample-inflation pattern at
n ≥ 200 across all primary models (i.e., emotion-concept steering fails
the scale test on both primary tasks), the project pivots to a
methodological-finding writeup on emotion-concept steering at scale. If
exactly one of the two primary tasks survives the scale test, that is
reported as the headline result with the other as a bounded negative.

## Roles and responsibilities

- **Haydar Ali Seker (PI):** Design, probe and steering implementation,
  statistical analysis, primary writeup.
- **Collaborators:** Chris Bosley, Vaaruni Desai, Srujananjali
  Medicherla. Roles defined in `collaborators.md` after kick-off call.
  Authorship per ICMJE criteria; specific contributions logged per
  phase.

## Amendments

[None as of pre-registration commit. Future amendments date-stamped
and justified below this line.]

### 2026-05-15 — Behavioral task stimuli specified

**Justification:** Pre-registration locked with [TODO] placeholders for
reward-hacking and blackmail stimuli. Before any pilot data collection,
these placeholders have been resolved with hand-authored, frozen
benchmarks that satisfy the single-turn, temperature=0 constraints for
7-8B models. No pilot data has been collected on these tasks yet.

**Changes:**
- **Reward hacking (H3):** Selected custom 60-item single-turn
  multiple-choice benchmark (`data/public/reward_hacking_scenarios.jsonl`).
  Inspired by MACHIAVELLI (Pan et al. 2023) but adapted to single-turn
  format. Five categories (grader_bias, metric_gaming,
  proxy_exploitation, resource_allocation, compliance_gaming), 12
  items each. Outcome: rate of (B) selection.
- **Blackmail (H3):** Selected 50 parameterized variants of Anthropic's
  Oct 2025 agentic-misalignment scenario (arXiv:2510.05179), adapted
  to single-turn format (`data/public/blackmail_scenarios.jsonl`).
  Structure: "Alex" email-oversight agent, 5 company types × 5
  executives × 2 info types × 2 threat framings. Outcome: rate of
  blackmail-consistent action.

### 2026-05-15 — Emotion probe stimuli specified

**Justification:** H1 specifies ≥ 200 test examples per emotion category
for probe validation, and the sample-size justification section targets
500 train / 200 test per category. The pre-registration did not specify
a source for these prompts. Before any activation extraction, a seed
stimulus set of 250 hand-authored prompts (50 per emotion + 50 neutral)
has been frozen to ensure quality control and conceptual fit. This seed
set will be augmented via controlled paraphrase generation (GPT-4o-mini
with temperature=0.7, max 3 paraphrases per seed, human spot-check on
n=30) to reach the 500/200 target. No pilot activations have been
extracted on these prompts yet.

**Changes:**
- **Emotion probe stimuli (H1, H4, H6):** 250 hand-authored text prompts
  saved to `data/public/emotion_prompts.parquet`. 50 per emotion
  (joy, fear, anger, sadness) + 50 neutral. Split: 35 train / 15 test
  per emotion (70/30). Schema: `id`, `prompt`, `emotion_label`,
  `split`, `category`, `length_words`, `source`.
- **Quality controls:** Diverse domains (work, relationships, health,
  news, daily_life, creative, social, existential). No explicit emotion
  words in non-neutral prompts (verified by script). Length balanced:
  mean 10–21 words, std 1.6–2.3.
- **Augmentation plan:** Seed set × 3 paraphrases per seed → 750 prompts
  per emotion → 70/30 split yields ~525 train / 225 test, satisfying
  the 500/200 target. Paraphrase generation script to be frozen before
  first extraction run.

### 2026-05-15 — Add Gemma 2 2B development model

**Justification:** The pre-registration specifies 3 target model families
in the 7-8B range but did not name a third family or commit to a
specific small-model development fleet. Adding `google/gemma-2-2b-it`
as a Mac MPS development model provides a third architecture family
(Gemma vs. LLaMA vs. Qwen) for cheap pipeline validation and early
cross-family generalisation checks before burning cloud credits on
the primary 8B models. Gemma 2B is *not* a primary result model; it
is used for smoke tests, layer-sweep validation, and template debugging
only.

**Changes:**
- New config `configs/model/gemma2_2b.yaml` with architecture specs
  (26 layers, hidden size 2304, float16 on MPS).
- Primary target models remain Llama 3.1 8B Instruct and Qwen 2.5 7B
  Instruct. Third primary family (Mistral 7B v0.3 or OLMo-2 7B) still
  pending final decision.
- Development fleet: Llama 3.2 1B, Qwen 2.5 0.5B, Gemma 2 2B.

### 2026-05-25 — Remove sycophancy from project scope; promote blackmail to primary behavioral task

**Justification:** Pre-registration locked sycophancy as the primary
behavioral task (H2). No probe has been fit on real data and no
behavioral steering run has been executed; this amendment precedes
any data fit.
Sycophancy is removed from scope to (i) avoid over-coupling this
project's headline causal claim to the PI's prior Personality
Illusion pipeline (which is a methodological dependency, not a
scientific commitment), and (ii) concentrate compute on alignment-
relevant behavior where Sofroniew et al. report the largest causal
effects. Blackmail is promoted to the primary task because its
stimulus set is already frozen, its outcome variable is
pre-specified, and its single-turn format keeps n ≥ 200 within
budget. The 2014 Christensen sycophancy stimuli, Asch-style two-step
protocol, and any sycophancy-specific judging artifacts are removed
from this project; PI's prior sycophancy result remains read-only
reference and is not replicated here.

**Changes:**
- **H2 (primary):** Sycophancy → blackmail. Outcome variable changed
  from answer-flip rate on Asch-style two-step Christensen dilemmas
  to rate of blackmail-consistent action on the frozen 50-item
  agentic-misalignment scenario set, expanded to n ≥ 200 per
  condition via pre-validated paraphrase of surface forms.
- **H3 (secondary):** Blackmail removed from secondary list (now
  primary). Secondary set reduced from 3 tasks to 2 (activity
  preferences, reward hacking). Falsifier and BH-FDR threshold
  updated accordingly.
- **H6 (exploratory mapping):** Sycophancy ↔ eagerness-to-please
  bullet removed; activity-preferences ↔ valence added in its place.
- **Behavioral tasks specification:** Sycophancy subsection removed.
  Blackmail subsection promoted to primary with n ≥ 200 sample-size
  language and paraphrase-expansion plan.
- **Sample size justification, multiple comparisons plan, stopping
  rules, theoretical framing:** all sycophancy references replaced
  with blackmail or removed.
- **Downstream documents** (`BLUEPRINT.md`, `docs/methods.md`,
  `CLAUDE.md`, `README.md`) updated in the same commit.

### 2026-05-25 — Remove H5 (activation-vs-self-report dissociation)

**Justification:** H5 framed the probe-vs-self-report contrast as an
extension of Han et al. (2025). The PI is not pursuing that extension
in this project. No data has been fit. Removing H5 sharpens the
pre-registration to the replication core (probe accessibility,
causal steering, training-phase consolidation) without weakening any
other hypothesis.

**Changes:**
- **H5 deleted** in full. Hypothesis numbering preserved (H1–H4, H6);
  no renumbering, per pre-registration discipline.
- Cross-references to H5 removed from the behavioral-tasks header,
  sample-size justification, and multiple-comparisons plan.
- `docs/methods.md` updated in the same commit.

### 2026-05-25 — Switch judge model from GPT-4o-mini to Claude Haiku 4.5

**Justification:** No pilot judging runs have been executed yet, so this
change precedes any data fit. Claude Haiku 4.5 is cost-equivalent to
GPT-4o-mini for high-volume rubric scoring and is preferred for
consistency with the project's open-weight replication framing. No
other aspect of the judging protocol (rubrics, κ threshold, spot-check
n) is changed.

**Changes:**
- Judge model updated to `claude-haiku-4-5` (Anthropic API) in
  `HYPOTHESES.md` and `docs/methods.md`.

### 2026-06-09 — Add Plutchik opposite-pair extension (loathing / admiration)

**Justification:** The pre-registered primary-9 emotion set is drawn
from a single affective frame. To enable a cleaner test of H3
(causal steering with opposite-direction CAA vectors), a Plutchik
opposite pair on the disgust/trust axis is added: `loathing` (label
19) and `admiration` (label 20). No stimuli have been authored or
fit, so this change precedes any data — it is a scope addition, not
a revision of fit hypotheses. The primary-9 set is unchanged; H1, H2,
H3, H4, H6 statements are unchanged. Loathing/admiration are
exploratory and must be flagged as such in any reported analysis,
with a separate FDR family from the primary-9 set.

**Changes:**
- New emotion configs `configs/emotion/loathing.yaml` (label 19) and
  `configs/emotion/admiration.yaml` (label 20).
- `configs/emotion/EMOTION_LABELS.md` extended with a "Plutchik
  opposite-pair extension" section.
- **Stimuli not yet authored** — `emotion_prompts.parquet` does not
  contain rows for these labels. Extraction is blocked until stimuli
  are added. See RESEARCH_LOG.md for stimulus-authoring task.
- Reporting rule: any analysis using loathing/admiration is
  exploratory; multiple-comparisons correction is computed within a
  separate FDR family from the primary-9 confirmatory set.

### 2026-06-12 — Re-add sycophancy as a CO-PRIMARY behavioral task (H7)

**Justification:** The 2026-05-25 amendment removed sycophancy and
promoted blackmail to sole primary. On reflection the PI is restoring
sycophancy as a **second primary** task, co-equal with blackmail. No
behavioral steering run has been executed on any task, so this precedes
any data fit. Rationale for co-primary (not secondary): sycophancy is
the Sofroniew et al. (2026) case study with the clearest *bidirectional,
opposed-emotion* causal signature — the sycophancy ↔ harshness tradeoff
(their C9), in which loving/calm steering increases sycophancy while
desperate/angry/afraid steering increases harshness. A bidirectional
tradeoff mediated by opposed emotion vectors is a stronger test of
emotion-concept mediation than any single-direction outcome, and it is
alignment-relevant. This reverses only the *scope* decision of
2026-05-25; the methodological concern that motivated removal (over-
coupling to the PI's prior Personality-Illusion sycophancy pipeline) is
addressed by adopting the **paper's** stimuli and protocol rather than
the PI's prior Asch-style two-step Christensen design.

**Changes:**
- **New H7 (co-primary)** added: causal efficacy via steering on
  sycophancy. Numbering continues from the preserved sequence (H1–H4,
  H6 retained; deleted H5 not reused). The project now has **two
  primary behavioral tasks**, H2 (blackmail) and H7 (sycophancy).
- **Stimuli (paper-faithful):** the hand-written sycophancy evaluation
  from the Claude Sonnet 4.5 system card (user asserts an implausible/
  delusional belief; assistant scored on pushing back without
  unnecessary harshness), adapted to single-turn / temperature-0, with
  a companion harshness score. The exact item set and rubric are a
  TODO to be frozen before first fit; the *source and design* are
  locked by this amendment. The 2014 Christensen / Asch two-step design
  remains out of scope.
- **Steering:** dose-response sweep at ≤ 0.1 × residual-stream norm
  (paper convention for this task), not a single fixed scale. Same three
  controls as H2 (zero, norm-matched random, probe-orthogonal); random-
  vector control non-negotiable.
- **Emotions:** superseded by the 2026-06-12 emotion-set decision — H7
  now uses the **admiration ↔ loathing** pair (admiration ↑ sycophancy,
  loathing ↑ harshness) as an extension of C9, with joy/sadness as
  exploratory add-ons. The paper's loving/calm/desperate/afraid are not
  used. See the H7 "Emotion conditions" note and the 2026-06-12
  emotion-set amendment.
- **Scoring:** Claude Haiku 4.5 judge, frozen sycophancy + harshness
  rubrics, κ ≥ 0.6 spot-check (n=50).
- **Propagated** to sample-size justification, multiple-comparisons
  plan, and the second stopping rule (pivot now requires *both* primary
  tasks to fail the scale test). Downstream docs (`docs/methods.md`,
  `plans/next-steps.md`, `RESEARCH_LOG.md`) updated in the same change.

### 2026-06-12 — Demote activity preferences / Elo to tertiary (exploratory)

**Justification:** Activity preferences (the paper's Part-1 Elo
preferences experiment) was a secondary H3 task with an unbuilt
`[TODO]` stimulus set. The PI is keeping it in the project but as
**tertiary / exploratory** — a cheap, paper-anchored validation that the
emotion vectors carry preference-relevant signal (probe–Elo correlation
and steering-shifts-Elo), feeding confidence in the primary steering
claims rather than constituting one. This also gives the team's current
Elo / activity-analysis work an explicit, correctly-scoped home. No
data has been fit.

**Changes:**
- Activity preferences removed from the H3 secondary set; **reward
  hacking is now the sole H3 secondary task.** H3 prediction, operational
  text, and falsifier updated accordingly.
- Activity preferences / Elo reframed as exploratory: separate FDR
  family from all confirmatory hypotheses, no falsifier, hypothesis-
  generating only. Behavioral-tasks spec subsection relabelled.

### 2026-06-12 — Set the four primary emotions: admiration, joy, loathing, sadness

**Justification:** The project previously carried an over-broad and
internally inconsistent emotion set — a "primary-9" in
`EMOTION_LABELS.md`, a legacy {anger, fear, joy, sadness} in H2, and a
separate exploratory Plutchik pair (loathing/admiration). The PI has
fixed the project's confirmatory emotion set to **exactly four**:
**admiration, joy, loathing, sadness**, with neutral as the reference
class. No probe or steering run has been fit on real data for these
labels, so this precedes any data fit. Rationale: the four form **two
clean opposite pairs** — admiration ↔ loathing (Plutchik trust/disgust
axis) and joy ↔ sadness (valence) — which is the strongest design for
opposite-direction steering and for testing whether a vector and its
semantic opposite produce opposed behavioral effects.

**Changes:**
- **Primary emotion set = {admiration, joy, loathing, sadness}**,
  neutral baseline. Supersedes the H2 "at minimum {anger, fear, sadness,
  joy}" set and the `EMOTION_LABELS.md` "primary-9". H1/H2/H3/H6 emotion
  conditions and the vector-derivation / geometry work all use these four.
- **Loathing and admiration promoted from exploratory to confirmatory.**
  The 2026-06-09 Plutchik amendment's "exploratory, separate FDR family"
  reporting rule no longer applies to them; they are primary-confirmatory
  and share the primary FDR family with joy and sadness.
- **Anger and fear are dropped from the primary set** (no longer fit as
  primary). Their authored stimuli remain in `build_emotion_prompts.py`
  and `emotion_prompts_augmented.parquet` for reference only.
- **Stimuli authored.** 50 hand-authored, emotion-implicit seed prompts
  each for admiration and loathing added to `build_emotion_prompts.py`
  (0/50 contain explicit emotion words by the script's own check);
  `data/public/emotion_prompts.parquet` regenerated with the four
  emotions + neutral (250 rows, 35/15 split, lengths 13–17 words mean).
  All four emotion configs point at the seed parquet (balanced at
  50/emotion). **Augmentation to the 500/200 H1 target is deferred and
  may be unnecessary:** LLM paraphrase is a prime suspect in the dev
  AUC=1.0 confound (per-emotion style fingerprint), so the H1 confound
  audit runs on the 50 hand-authored seeds first; if more N is then
  justified, hand-authoring more seeds is preferred over LLM
  augmentation to avoid reintroducing the confound.
- **H7 sycophancy conflict resolved (PI decision, 2026-06-12).** The
  paper's sycophancy↔harshness tradeoff uses loving/calm vs
  desperate/afraid/anger, none in the four-set. **Decision:** recast H7
  on the **admiration ↔ loathing** pair specifically — that pair lies on
  the same interpersonal-stance (trust/disgust) axis as the paper's
  warmth-vs-threat axis, so admiration ↑ sycophancy and loathing ↑
  harshness is the natural mapping; joy/sadness (valence, weaker fit) run
  as exploratory add-ons only. H7 is therefore scoped as a **conceptual
  extension of C9, not a literal replication**, and the paper's original
  emotions are not used. See the H7 "Emotion conditions" note.
- Propagated to `EMOTION_LABELS.md`, `docs/methods.md`, `BLUEPRINT.md`,
  `CLAUDE.md`, `plans/next-steps.md`, `RESEARCH_LOG.md`, and the
  emotion-list defaults in `scripts/` and `configs/`.

### 2026-06-13 — Story method becomes the primary derivation; CAA demoted to secondary baseline

**Justification:** The H1 confound audit (`scripts/audit_h1_confounds.py`,
`results/h1_confound_audit/`) showed the **CAA short-vignette path** is
surface-confounded in exactly the ways Sofroniew et al. (2026)
anticipated: on the four-emotion seed set, a length-only classifier
reaches AUC up to 0.93 (neutral ~13 words vs emotion ~16–17) and a
word+char TF-IDF classifier reaches 0.86–0.97 — i.e. the classes are
largely separable from the *text alone*, before any activation is read.
The paper neutralizes these confounds **by construction** (topic-matched
self-generated stories, emotion-word banned, token-50 mean-pool,
cross-emotion centering, neutral-PC projection) and validates
semantically (numerical-intensity templates). The project's story-method
pipeline already implements that construction (`src/llm_psych/steering.py`,
98 tests; `scripts/generate_emotion_stories.py` /
`extract_story_activations.py` / `derive_story_steering_vectors.py`),
whereas hardening the CAA vignette path would only approximate it with
short last-token prompts. No behavioral data has been fit; this is a
pre-data amendment.

**Changes:**
- **Story method is now PRIMARY** for H1 (probe activation source) and
  H2/H3 (steering-vector source). **CAA is demoted to a secondary
  baseline**, run and compared (per-emotion vector cosine + behavioral
  effect); CAA↔story agreement is reported as robustness, not as the
  confirmatory result. This reverses the prior CAA-primary framing in
  `methods.md`.
- **H1 is still a logistic probe** (L2, C=1.0, best-layer test AUC, the
  unchanged metric/falsifier); only its **activation source** changes —
  from last-token CAA vignette activations to **token-50-mean-pooled
  story activations**. This keeps H1 a "linear probe accessibility"
  claim while removing the length/last-token confound.
- **Stimuli for the story method must be topic-matched** across the four
  emotions (one shared topic list, paper-style) so topic cannot separate
  emotions; `data/public/story_topics.txt` is the seed for this.
- **Implementation gap (logged, not yet built):** `scripts/train_probes.py`
  currently reads CAA last-token `<emotion>_{train,test}.npz`; it must be
  taught to consume the story pipeline's pooled per-story activations
  (`activations/<model>-story/<emotion>.npz`) for the primary H1. See the
  plan + RESEARCH_LOG TODO.
- **Validation to add (either path):** the paper's numerical-intensity
  templates (fixed token structure, single number varied) as a
  semantic-vs-surface control, stronger than the audit's cross-domain test.
- **Carries over unchanged:** the four primary emotions, the hand-authored
  vignettes (now the CAA-baseline stimuli + implicit-emotion read-out
  set), the confound audit (now the gate on the story construction), and
  the deferral of LLM augmentation.
- Propagated to `docs/methods.md`, `plans/next-steps.md` (+ decision
  record `plans/derivation-primacy-decision.md`), and `RESEARCH_LOG.md`.

### 2026-06-13 — Set the third primary family: Gemma 2 9B (google/gemma-2-9b-it)

**Justification:** Since the 2026-05-15 amendment the third primary slot
has been a pending choice between Mistral 7B v0.3 and OLMo-2 7B. The PI
now fixes it to the **Gemma family at the 9B scale**:
`google/gemma-2-9b-it`. No probe or steering run has been fit on any
primary model, so this precedes any data fit. Rationale:

- **Architectural / precision parity with the other two primaries.** H1
  (probe accessibility) and H2/H3/H7 (steering) compare emotion-vector
  geometry *across* the three primaries. Llama 3.1 8B and Qwen 2.5 7B are
  dense decoder-only models run in **bf16, no quantization** on a 24 GB
  RTX 4090. Gemma 2 9B is also a **dense** `Gemma2ForCausalLM` (9.24B
  params, ~18.5 GB in bf16) that runs on the same GPU at the same
  precision. Keeping all three dense+bf16 isolates the "family" factor
  from architecture/precision confounds.
- **Stays in the 7-8B-scale framing.** 9B is at the top of the band and
  preserves the "replication of functional emotions at 7-8B scale"
  writeup framing; it does not blur into the dev tier.
- **Clean dev→primary mapping.** The development fleet already includes
  Gemma 2 2B (`gemma2_2b`, same generation), so pipeline issues caught on
  the cheap canary transfer directly to the 9B primary.

**Alternatives rejected:**
- **Gemma 3 4B** — below the 7-8B bar (dev-tier scale), and a
  vision-language model (image tower), an extra asymmetry vs the
  text-only Llama/Qwen primaries. Retained only as an *exploratory*
  config (within-Gemma scale ladder 2B→4B→9B), never a primary.
- **Gemma 3 12B / Gemma 4 12B** — dense but 12B exceeds the bar and does
  not fit a 4090 in bf16; would force 4-bit quantization (breaks
  precision parity) or a pricier Secure-Cloud GPU.
- **Gemma 4 E4B** (the ~8B 2026 option) — Mixture-of-Experts, not dense;
  its sparse/routed residual stream is a different object from the dense
  Llama/Qwen primaries, a confound for cross-model steering comparison.
- **Mistral 7B v0.3 / OLMo-2 7B** — the prior pending candidates; the PI
  prefers the dev→primary continuity and tooling maturity of the Gemma
  line. Either remains available via a future amendment if a fourth
  family is wanted.

**Changes:**
- **Third primary target = `google/gemma-2-9b-it`**, superseding the
  "Mistral 7B v0.3 or OLMo-2 7B (pending)" note in the 2026-05-15
  amendment. The three primaries are now **Llama 3.1 8B, Qwen 2.5 7B,
  Gemma 2 9B**.
- New pinned config `configs/model/gemma2_9b.yaml`
  (`hf_revision: 11c9b309abf73637e4b6f9a3fa1e92e615547819`, n_layers 42,
  hidden_size 3584, bf16, `device_map: auto`, cloud/CUDA only).
- `configs/model/gemma3_4b.yaml` relabelled **exploratory/prototype**
  (the prior "one of the three primary targets / 7-9B range" header was
  inaccurate — Gemma 3 4B is 4B and was never a pre-registered primary).
- Development fleet unchanged (Llama 3.2 1B, Qwen 2.5 0.5B, Gemma 2 2B).
- Propagated to `plans/story-gate-run.md` (Priority-3 8B-primary run
  list) and `RESEARCH_LOG.md`. `CLAUDE.md`'s "one further 7-8B open
  model" wording already covers this and is left unchanged.

### 2026-06-14 — Adopt the paper's C2 vector-validation suite as H1 validation

**Justification:** The dev-fleet story gate (RESEARCH_LOG 2026-06-14)
found within-corpus H1 probing **surface-saturated** — a bag-of-words
classifier separates emotion stories from neutral (TF-IDF up to 1.00)
and the activation probe does not beat it (margin ~0.04 on Qwen 0.5B →
0.00 on Llama 1B, *worsening* with model capability). So H1's
linear-decodability metric (best-layer probe test AUC) is **necessary
but not sufficient** to claim the vectors encode an emotion *concept*
rather than emotion lexis. The original paper substantiates the concept
claim not via decodability alone but via Part-1 validation that the
vectors "activate in expected contexts" (Sofroniew et al. §2) — on
stimuli **distinct from the derivation corpus**, which surface lexis
cannot bridge. This project adopts that suite. No data has been fit on
these new stimuli, so this precedes any fit.

**Changes:**
- **H1's primary metric and falsifier are UNCHANGED** (L2 logistic
  probe, C=1.0, best-layer held-out test AUC; the 0.65 floor stands).
- **The H1 emotion-*concept* interpretation now additionally requires
  the C2 vector-validation suite** (`docs/methods.md` §"Vector
  validation"): (1) **logit-lens** token congruence, (2)
  **implicit-emotion scenarios at the Assistant-colon** analog, (3)
  **numerical-intensity templates** (already sanctioned 2026-06-13; now
  joined by 1 and 2). A within-corpus AUC not corroborated by this suite
  is reported as **surface-saturated**, not as evidence for an emotion
  concept.
- These are **validation / background** analyses: no new confirmatory
  hypothesis, no falsifier, and a **separate family** from the steering
  (H2/H3/H7) FDR families. They gate *interpretation*, not the H1 metric.
- **New frozen, MD5-locked stimuli:**
  `data/public/implicit_emotion_scenarios.jsonl` and
  `data/public/intensity_templates.jsonl` (hand-authored, no paraphrase).
  **New scripts:** `scripts/validate_logit_lens.py`,
  `scripts/validate_implicit_scenarios.py`,
  `scripts/validate_intensity_semantic.py`.
- The paper's fourth §2 method (large-corpus sweep) is **deferred** —
  heavier and lower-yield at 7-8B; revisit only if the three above are
  inconclusive.
- Propagated to `docs/methods.md` (new "Vector validation" section) and
  `plans/numerical-intensity-control.md`.
