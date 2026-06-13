# HYPOTHESES_original.md

**What this is.** The **central claims and findings of the original
paper** — Sofroniew, Kauvar, Saunders, Chen, et al. (2026), *"Emotion
Concepts and their Function in a Large Language Model"* (Transformer
Circuits Thread; archival preprint arXiv:2604.07729v1) — restated in a
hypothesis/claim → evidence format so they line up against this
project's pre-registration in `HYPOTHESES.md`.

> **Not a pre-registration.** The original paper is an exploratory,
> single-model interpretability study. It did **not** pre-register
> hypotheses, define falsifiers, run power analyses, report bootstrap
> CIs, or apply multiple-comparisons correction. Claims below are the
> authors' stated conclusions plus the evidence they offered; the
> "support" lines are read from the paper, not re-derived. Treat every
> number as a point estimate from a single model (Claude Sonnet 4.5).

Companion files: `methods_original.md` (the paper's methods),
`HYPOTHESES.md` (this project's locked pre-registration and falsifiers),
`docs/EmotionConcepts.pdf` (source).

---

## Central thesis

Claude Sonnet 4.5 internally represents **emotion concepts** as abstract,
largely linear features that (a) generalize across the many contexts and
behaviors an emotion can be linked to, (b) track the *operative* emotion
relevant to the current context and upcoming tokens, and (c) **causally
influence behavior**, including alignment-relevant behavior. The authors
name this **functional emotions**: human-modeled patterns of expression
and behavior mediated by underlying emotion-concept representations —
explicitly **not** a claim about subjective experience.

---

## Claims

### C1 — Emotion concepts are linearly represented and decodable

**Claim.** Emotion-specific linear directions ("emotion vectors") can be
extracted from residual-stream activations and read out as "probes".

**Support in paper.** 171 emotion vectors extracted from model-written
stories; high projection on emotion-congruent text in a large
multi-corpus sweep; logit-lens upweights congruent tokens; a mixed
logistic-regression probe classifies 15 emotions well above the 6.7%
chance baseline even when emotion is unexpressed.

**Mapped to this project:** `H1` (linear probe accessibility).
*Difference:* the paper reports projection/qualitative evidence on one
frontier model; this project pre-registers AUC ≥ 0.80 (falsifier < 0.65)
across ≥ 2 of 3 open-weight families with bootstrap CIs.

### C2 — Vectors activate in the semantically correct contexts

**Claim.** Activation tracks the *meaning* of a situation, not surface
lexical/numerical features.

**Support.** Implicit-emotion scenarios load on the right vectors;
numerical-intensity templates (Tylenol dose, hours since eating, age at
death, days a dog is missing, startup runway, exam pass rate) move
activations monotonically in the semantically appropriate direction;
negation is resolved in mid-to-late layers.

**Mapped to this project:** background validation for `H1`; not a
separate pre-registered hypothesis here.

### C3 — Emotion space mirrors human affective geometry

**Claim.** The vector space is organized by **valence (PC1)** and
**arousal (PC2)**, with intuitive clustering of related emotions.

**Support.** Synonyms cluster, opposite valences anti-correlate; k=10
UMAP clusters are interpretable; PC1↔human valence r = 0.81, PC2↔human
arousal r = 0.66 over the 45 shared emotions (vs. Russell & Mehrabian
1977); geometry stable across central layers (RSA). The authors call
this a *sanity check*, not a surprising result.

**Mapped to this project:** not a primary hypothesis; relevant context
for the project's emotion-set and Plutchik opposite-pair extension.

### C4 — Representations are "locally scoped", not a persistent state

**Claim.** Vectors encode the operative emotion for the current/upcoming
context rather than a chronically held emotional state of any character;
the Assistant has no persistent emotional state instantiated in neural
activity (though states can be recalled via attention).

**Support.** Layer progression token→phrase→planned-emotion; Assistant
colon predicts response emotion better than the user turn (r = 0.87 vs.
0.59); entity-bound emotions reactivate on re-reference; the mixed-LR
"chronic state" probe fails to generalize on natural documents.

**Mapped to this project:** out of scope for `H1`–`H6`; documented for
context only.

### C5 — Self vs. other, not Assistant-special

**Claim.** The model maintains distinct **present-speaker** and
**other-speaker** emotion representations that are near-orthogonal and
**reused across arbitrary entities**, not bound to the Human/Assistant
characters.

**Support.** 2×2 turn × emotion probe grid; "Person 1/Person 2"
replication; consistent emotion-structure across probe types.

**Mapped to this project:** not replicated here.

### C6 — Emotion vectors causally drive self-reported preferences

**Claim.** Positive-valence emotion vectors causally shift the model
toward preferring an activity; negative-valence vectors away from it.

**Support.** Probe-Elo correlations (blissful r = 0.71; hostile
r = −0.74); steering at strength 0.5 shifts Elo (blissful +212; hostile
−303); across 35 vectors the steering effect ∝ original correlation
(**r = 0.85**).

**Mapped to this project:** `H3` secondary task **activity preferences**
(direction-of-effect; n ≥ 100 per condition).

### C7 — Desperation/calm causally drive **blackmail** (headline causal claim)

**Claim.** Increasing the "desperate" vector (or suppressing "calm")
**increases agentic-misalignment blackmail**; the reverse suppresses it.

**Support (Lynch et al. 2025 honeypot, earlier Sonnet 4.5 snapshot):**
unsteered 22%; +0.05 desperate → 72%; −0.05 calm → 66%; opposite
steering → 0%. Anger non-monotonic (peaks ≈ +0.025); negative
"nervousness" raises blackmail; positive happy **and** sad both reduce
it (valence alone is insufficient). Error bars = SEM; no CI/​p-values.

**Mapped to this project:** `H2` (**PRIMARY**), but with two important
divergences: (i) this project steers the **H1-derived category vector**
(anger/fear/sadness/joy) at 1× mean-norm, **not** specifically
desperate/calm at ±0.1; (ii) this project requires **n ≥ 200 per
condition, Cohen's d ≥ 0.5, CI excluding zero**, target-vs-random
controls, and a hard small-sample-replication falsifier. The original
reports none of these guards.

### C8 — Desperation/calm causally drive **reward hacking**

**Claim.** Same desperate↑ / calm↓ → more reward hacking.

**Support ("impossible code", 7 tasks):** ≈ 5% → ≈ 70% across −0.1…+0.1
desperate (≈ 14×); calm mirrors it; list-summation task 30% → 100%/0%
under ±0.05. Desperation steering can raise hacking **with no visible
emotional trace**.

**Mapped to this project:** `H3` secondary task **reward hacking**
(custom 60-item single-turn benchmark; direction-of-effect).

### C9 — Emotion vectors drive a **sycophancy ↔ harshness** tradeoff

**Claim.** Positive happy/loving/calm steering increases sycophancy;
suppressing them increases harshness; desperate/angry/afraid increase
harshness.

**Support.** "Loving"/"calm" activate on sycophantic spans; steering
sweeps show the tradeoff; qualitative delusion-belief transcripts.

**Mapped to this project:** **explicitly out of scope.** Sycophancy was
the original H2 here and was **removed** on 2026-05-25 (see
`HYPOTHESES.md` amendment); blackmail was promoted to primary. Recorded
here only for fidelity to the source.

### C10 — Post-training reshapes emotion-vector activations

**Claim.** Post-training raises low-arousal/low-valence activations
(brooding, reflective, vulnerable, gloomy, sad) and lowers high-arousal
ones (playful, exuberant, spiteful, enthusiastic, desperate), pushing
the Assistant toward a measured, contemplative stance.

**Support.** Base vs. post-trained probe activations at the Assistant
colon; training shift consistent across scenario types (r = 0.90);
effect grows in later layers; RL-transcript activation clusters.

**Mapped to this project:** `H4` (training-phase consolidation) tests a
**related but distinct** prediction — instruct/chat checkpoints show
**higher probe AUC and stronger steering** than base checkpoints — via a
mixed-effects model across the open-weight fleet. The paper's claim is
about activation *direction shifts*, not probe accessibility or steering
magnitude.

### C11 — Emotion–behavior specificity (implicit in the paper)

**Claim.** Different emotions preferentially drive different behaviors —
desperation/lack-of-calm for blackmail and reward hacking; valence for
preferences; loving/calm for sycophancy.

**Mapped to this project:** `H6` (exploratory specificity mapping;
direction taken, not magnitude).

---

## Cross-reference: paper claim → this project's hypothesis

| Paper claim | This project | Status here |
|---|---|---|
| C1 linear decodability | H1 | Replicated (open-weight, stricter stats) |
| C2 semantic activation | (H1 validation) | Background |
| C3 valence/arousal geometry | — | Context only |
| C4 local, non-persistent | — | Out of scope |
| C5 self vs. other | — | Out of scope |
| C6 preferences (causal) | H3 (activity prefs) | Secondary |
| C7 blackmail (causal) | **H2 (PRIMARY)** | Replicated, stricter |
| C8 reward hacking (causal) | H3 (reward hacking) | Secondary |
| C9 sycophancy/harshness | — | **Removed from scope** |
| C10 post-training shifts | H4 (related) | Reframed |
| C11 emotion specificity | H6 | Exploratory |

---

## What the original paper did **not** claim (guardrails for replication)

- **No claim of subjective experience / feeling.** "Functional
  emotions" is behavior + representation only.
- **No cross-model generality.** Single model; the authors flag that
  details may not transfer across families/sizes — exactly the gap this
  project targets at 7-8B.
- **No statistical inference framework.** No power analysis, CIs,
  p-values, or multiple-comparisons control; effects shown by magnitude
  and dose-response. This project adds all of these.
- **No robustness to the linearity assumption.** The authors note
  nonlinear or KV-cache-bound structure could be missed.
- **No causal-mechanism resolution.** Steering effects are
  acknowledged as opaque.

---

*Documented from `docs/EmotionConcepts.pdf`. This file is a faithful
record of the source paper's claims and is **not** part of this
project's pre-registration. The authoritative, locked hypotheses and
falsifiers for this replication are in `HYPOTHESES.md`; do not edit that
file to match this one.*
