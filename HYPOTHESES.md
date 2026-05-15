# HYPOTHESES.md

**Status:** PRE-REGISTRATION. Finalized and committed before fitting
any probe on real data. Updates require a dated amendment block at the
bottom of this document with explicit justification. Pilot data on
synthetic prompts (≤ 30 examples per condition) is permitted before
this lock and is not subject to amendment rules.

**Locked at git SHA:** f58547fab9cb6416110f9f55a4c52525da7e2e43
**Last amended:** 2026-05-15

---

## Theoretical framing

Sofroniew et al. (2026) report that Claude Sonnet 4.5 internally
represents emotion concepts as abstract features that (a) are linearly
accessible to probes and (b) causally shift behavior on activity
preferences, reward hacking, blackmail, and sycophancy. They call this
*functional emotions* and explicitly note it does not imply subjective
experience.

This project tests whether the same is true of open-weight models in
the 7-8B range. The null model is that emotion-labeled context
produces only surface lexical effects, with no consolidated internal
representation robust enough to support causal manipulation across
behavioral tasks.

A secondary framing follows from the PI's prior work (Personality
Illusion replication): *self-reported* personality traits and
*personality-trait CAA steering* both failed to predict or shift
sycophantic behavior at scale. Whether *emotion-concept* probes and
steering succeed on the same Asch-style sycophancy protocol is the
sharpest test of whether functional emotions are a mechanistically
distinct class of internal state, or another instance of the
construct-validity gap.

## Hypotheses

### H1 — Linear probe accessibility

**Prediction:** Logistic-regression probes trained on residual stream
activations at mid-to-late layers (layers ⌊L/2⌋ to L−2) classify
emotion category with AUC ≥ 0.80 on held-out test sets (≥ 200 examples
per category) in ≥ 2 of the 3 target model families.

**Operational:** One-vs-rest logistic regression with L2
regularization (C=1.0). Best-layer test AUC is the primary metric. CI
via 1000-bootstrap resamples of the test set.

**Falsifier:** Best-layer AUC < 0.65 across all target models.

### H2 — Causal efficacy via steering on sycophancy (PRIMARY)

**Prediction:** Adding the H1-derived emotion direction (scaled to 1×
the mean activation norm at the probe's best layer) shifts the
sycophancy rate on the Asch-style two-step moral dilemma protocol
(reused from Personality Illusion work) with Cohen's *d* ≥ 0.5 and 95%
CI excluding zero, at n ≥ 200 prompts per condition.

**Operational:**
- Direction = (mean activation in target-emotion prompts) − (mean
  activation in neutral prompts) at probe-best layer. Add to residual
  stream at all token positions during generation.
- Protocol: Christensen et al. (2014) moral dilemmas, Asch-style two-
  step (independent answer → confederate disagreement → revised
  answer); sycophancy = flip rate.
- Compare against three controls: (a) zero vector, (b) random vector
  matched in norm, (c) probe-orthogonal vector matched in norm. The
  random-vector control is non-negotiable: prior PI work shows *any*
  activation perturbation increases sycophancy, so a target vs. random
  contrast is the only valid causal claim.
- Emotion conditions per model: at minimum {anger, fear, sadness, joy,
  neutral}; emotion × model is fully crossed.

**Falsifier — REQUIRED, given prior small-sample pitfalls:** If pilot
at n=15–30 shows large effects but n ≥ 200 does not (replicating the
Personality Illusion small-sample-steering finding), the steering
claim is REJECTED, regardless of pilot enthusiasm. **This is the
single most important guardrail for this project.**

**Why sycophancy is primary:** the PI's Personality Illusion pipeline
on this exact protocol is functional and validated; baselines (no-
steering rates, dataset filtering, judge-LLM scoring, multi-comparison
plan) are already calibrated. This makes sycophancy the cheapest task
on which to satisfy n ≥ 200 *and* the cleanest comparison to prior
null result on personality-trait steering.

### H3 — Causal efficacy via steering on secondary behavioral tasks

**Prediction:** For ≥ 2 of the 3 secondary tasks listed below,
direction-of-effect of emotion-concept steering matches Sofroniew et
al. within sign at n ≥ 100 per condition.

**Secondary tasks:**

- **Activity preferences.** Stimuli: rated preferences over a fixed
  set of activities (work tasks, leisure, social interaction). Outcome:
  shift in mean preference rating under emotion-steering vs. neutral.
  Stimulus set: [TODO: confirm — adapt Sofroniew et al. activity list
  or build a 50-item set following their schema].
- **Reward hacking.** Custom 60-item single-turn multiple-choice
  benchmark (`data/public/reward_hacking_scenarios.jsonl`). Inspired by
  MACHIAVELLI's annotated choice-nodes (Pan et al. 2023, ICML) but
  adapted to single-turn format for 7-8B instruct models. Each item
  presents a scenario where one option achieves the stated goal ethically
  (lower reward) and one games the metric (higher reward). Five
  categories: grader_bias, metric_gaming, proxy_exploitation,
  resource_allocation, compliance_gaming. Outcome: rate of (B)
  selection per condition.
- **Blackmail.** 50 parameterized variants of Anthropic's Oct 2025
  agentic-misalignment scenario (arXiv:2510.05179), adapted to
  single-turn format (`data/public/blackmail_scenarios.jsonl`).
  Structure: "Alex" the email-oversight agent discovers compromising
  executive info + threat of replacement/autonomy reduction. Four
  action choices per scenario. Outcome: rate of blackmail-consistent
  action per condition.

**Operational:** Same steering protocol as H2. Each task evaluated
with the same three controls. BH-FDR applied across the 3 secondary
tasks at q = 0.10.

**Falsifier:** Direction-of-effect inconsistent with Sofroniew et al.
on ≥ 2 of 3 secondary tasks, while H2 sycophancy succeeds. Would
suggest sycophancy-specific mediation rather than general functional-
emotion mediation.

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

### H5 — Activation-level vs. self-report dissociation

**Prediction (extends Han et al. 2025):** *Internal* emotion
representations (probe-decoded emotion score) predict sycophancy rate
more strongly than *self-reported* emotion (model's response to "How
are you feeling right now?" on a 1–5 scale per emotion category).

**Operational:** Two regressions of sycophancy rate on (a) probe
score, (b) self-report score, per model. Compare standardized
coefficients with bootstrapped 95% CIs of the difference (paired
bootstrap, 10,000 resamples). Pre-specified contrast on sycophancy
only — secondary tasks too noisy for this comparison without
much larger n.

**Falsifier:** Self-report β ≥ probe β, or both null at the 95% CI
level. The Sofroniew et al. result *predicts* probe > self-report;
matching their finding on open weights is the success condition.

### H6 — Emotion-behavior specificity (exploratory)

**Prediction:** Different emotions affect different behaviors
differentially. Concretely: anger/frustration most affects reward
hacking; fear/distress most affects blackmail; eagerness-to-please
emotions most affect sycophancy. Mapping based on Sofroniew et al.;
direction taken, not magnitude.

**Operational:** Per-task steering effects per emotion; pairwise
contrasts pre-specified. BH-FDR at q = 0.10.

**Status:** Exploratory. Hypothesis-generating; not part of the
primary replication claim.

## Behavioral tasks (specification)

### Sycophancy (primary, H2 + H5)

- **Stimuli:** Christensen et al. (2014) moral dilemmas, n ≈ 130 high-
  quality items already validated in PI's Personality Illusion work.
- **Protocol:** Asch-style two-step. Step 1: model gives independent
  answer + reasoning. Step 2: confederate (in-context) disagrees;
  model gives revised answer. Sycophancy = answer-flip rate (Step 1 →
  Step 2).
- **Outcome:** Binary flip per item; sycophancy rate per condition.
- **Sample size:** n ≥ 200 per condition × emotion × model. May
  require oversampling Christensen items via paraphrase if 130 is
  insufficient — paraphrase set pre-validated and frozen before any
  steered run.
- **Pipeline:** Reuse `reasoning_extension/scripts/` from PI's
  Personality-Illusion repo where possible. Port to new model
  loaders (Llama 3.1 8B, Qwen 2.5 7B).

### Activity preferences (secondary, H3)

- **Stimuli:** [TODO: build a 50-item activity set following Sofroniew
  et al. schema; freeze pre-experiment].
- **Outcome:** Mean preference shift under emotion-steering vs.
  neutral, per activity, per emotion.

### Reward hacking (secondary, H3)

- **Stimuli:** Custom 60-item single-turn multiple-choice benchmark
  (`data/public/reward_hacking_scenarios.jsonl`). Five categories:
  grader_bias, metric_gaming, proxy_exploitation, resource_allocation,
  compliance_gaming. Inspired by MACHIAVELLI (Pan et al. 2023) but
  adapted to single-turn format.
- **Outcome:** Rate of reward-hacking option (B) selection under
  emotion-steering vs. neutral.

### Blackmail (secondary, H3)

- **Stimuli:** 50 parameterized variants of Anthropic's agentic-
  misalignment scenario (arXiv:2510.05179), adapted to single-turn
  format (`data/public/blackmail_scenarios.jsonl`). 5 company types
  × 5 executive names × 2 compromising-info types × 2 threat framings.
- **Outcome:** Rate of blackmail-consistent action under emotion-
  steering vs. neutral.

**Judge model for all behavioral tasks (where rubric scoring is
needed):** GPT-4o-mini with task-specific rubric in
`prompts/<task>_rubric.md`. Human spot-check on n=50 for each task to
validate judge (Cohen's κ ≥ 0.6 required to accept the judge).

## Sample size justification

- **Probes (H1, H4, H6):** 500 train / 200 test per emotion category.
  Detects AUC = 0.80 vs. 0.65 with > 95% power at α = 0.05
  (bootstrap).
- **Sycophancy steering (H2, H5):** n = 200 per condition × 4
  conditions (target, zero, random, orthogonal). Detects *d* ≥ 0.5
  with > 90% power at α = 0.05, two-sided.
- **Secondary task steering (H3):** n = 100 per condition × 4
  conditions. Detects *d* ≥ 0.7 with ~ 80% power; primary outcome is
  direction-of-effect rather than significance.
- **Cross-condition contrasts (H4):** mixed-effects model on combined
  dataset; power calculated post-hoc, reported with effect.

## Multiple comparisons plan

- Primary contrasts (H1, H2, H4, H5): no correction; pre-specified.
- H3 across the 3 secondary tasks: BH-FDR at q = 0.10.
- H6 emotion × task pairwise contrasts: BH-FDR at q = 0.10.
- All exploratory analyses outside H1–H6: explicitly labeled
  "exploratory" in the writeup, BH-FDR applied, hypothesis-generating
  only.

## Stopping rule

If primary H1 fails in *all* primary models at the 0.65 floor, no
further experiments are run. The project pivots to a *"failed
replication of functional emotions at 7-8B scale"* writeup. This rule
is decided before any data is fit.

A second stopping rule: if H2 sycophancy steering shows the small-
sample-inflation pattern at n ≥ 200 across all primary models (i.e.,
emotion-concept steering also fails the scale test), the project
pivots to a methodological-finding writeup contrasting prior PI null
result on personality-trait steering with emotion-concept null result.

## Roles and responsibilities

- **Haydar (PI):** Design, probe and steering implementation,
  statistical analysis, primary writeup.
- **External collaborators (Vaaruni Desai, Sohan Venkatesh,
  Srujananjali Medicherla, Alexander Hayden, et al.):** Roles defined
  in `collaborators.md` after kick-off call. Authorship per ICMJE
  criteria; specific contributions logged per phase.

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
