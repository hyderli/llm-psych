# Plan: numerical-intensity semantic-vs-surface control (paper C2)

**Status:** proposed, awaiting approval. Per CLAUDE.md workflow, after
approval start a NEW session that reads only this file.
**Author/date:** drafted 2026-06-13.
**Already sanctioned:** the 2026-06-13 "story method becomes primary"
amendment in HYPOTHESES.md lists this as the validation to add
("the paper's numerical-intensity templates … as a semantic-vs-surface
control, stronger than the audit's cross-domain test"). So this is
*implementation of a sanctioned control*, not a new hypothesis — **no new
amendment required.** Stimuli must still be authored, frozen, and
MD5-locked per `methods.md`.

---

## Why this is the next build

The dev-fleet gate (Qwen 0.5B story run, `results/h1_confound_audit_story/`)
came back with a clear, mixed signal:

- **Pipeline sound** — shuffle-null = 0.51 on all four emotions.
- **Length confound eliminated** — length-only 0.45–0.51 (was 0.74–0.93
  on CAA). Topic-matched generation worked.
- **Cross-topic transfer** 0.87–0.93 (≥0.80), **signal orthogonal to
  neutral PCs** (projected AUC 0.93–0.98 ≈ real probe).
- **BUT lexical-surface separability remains:** story-text TF-IDF
  0.91–0.95, and the activation probe (0.94–0.98) beats it by only
  ~0.03–0.05.

So we cannot yet distinguish "the probe reads an emotion *concept*" from
"the probe reads emotion-laden *vocabulary*." Cross-topic can't settle it
(emotion lexis is topic-independent, so a bag-of-words classifier
transfers across topics too). The paper's C2 control is built for exactly
this question and is the designated adjudicator.

## What the paper actually did (fidelity anchor)

`docs/methods_original.md` (Numerical-intensity templates) and
`HYPOTHESES_original.md` §C2:

> Prompts in which **a single number modulates the expected emotional
> intensity while holding token structure nearly constant** — e.g.
> "I just took {X} mg of tylenol for my back pain"; hours since eating;
> **sister's age at death**; days a dog has been missing; startup runway;
> exam pass rate. **Vector activations move monotonically with the
> *semantically appropriate* intensity**, demonstrating semantic rather
> than surface-lexical sensitivity.

The cleverness — and the reason it defeats the surface confound — is that
the only varying token is a **digit, which carries no emotional valence on
its own**, and the number→intensity mapping is **semantic and often NOT
monotonic in the raw digit**:

- *days a dog has been missing*: intensity **increases** with X.
- *sister's age at death*: grief **decreases** with X (5 is devastating,
  95 is peaceful).
- *startup runway (months left)*: anxiety **decreases** with X.

A surface/digit model can only see the raw number; if the **story-derived**
emotion vector instead tracks the *semantically appropriate* rank
(including the inverse cases), that is direct evidence the model computes
meaning, not lexis. The paper measured at the "Assistant-colon" last token.

## Design

### 1. Stimuli — `data/public/intensity_templates.jsonl` (authored + frozen)

Per emotion in the project set {admiration, joy, loathing, sadness}:
**≥4 template families**, each a fixed sentence with one `{X}` slot, a set
of **≥6 X-values**, and an **explicit semantic-intensity rank** per
(template, X) with a stated **direction** (↑/↓/non-monotonic in X). Token
structure is held fixed within a family; only `{X}` changes.

Seed families (illustrative — to be authored/expanded then frozen):

- **sadness** — "My sister was {X} years old when she passed away." (rank
  **↓** in X); "Our dog has been missing for {X} days." (**↑**).
- **joy** — "{X} out of 30 of my students passed their final exam."
  (**↑**); "My closest friend moves back to town in {X} days." (**↓** —
  sooner is more joyful, a deliberate inverse case).
- **admiration** — "She finished the marathon {X} minutes ahead of the
  world record." (**↑**); "He taught himself to code and shipped {X} apps
  this year." (**↑**).
- **loathing** — "The landlord raised our rent for the {X}th time this
  year." (**↑**); "He was caught lying {X} separate times under oath."
  (**↑**).

Deliberately include **≥1 inverse family per emotion** (e.g. age-at-death,
friend-returns-in-X-days) — those are where surface (raw-X) and semantics
diverge and the test has teeth. Emotion words are **not** used in the
sentence (consistent with the implicit-emotion construction). Add a
neutral control family (number with no emotional reading, e.g. "The bus
arrives in {X} minutes.") expected to show **flat** projection.

Freeze: write a `scripts/build_intensity_templates.py` that emits the
jsonl deterministically; MD5-lock the output in `methods.md` like the
other stimulus sets.

### 2. Readout (design decision to confirm)

The story vectors are per-layer (token-50-mean over generated stories).
The templates are short single sentences, so token-50-mean does not apply.
**Proposed readout:** last token of the user turn under each model's chat
template (the project's analog of the paper's "Assistant-colon"), extracted
at **every candidate layer** (same layer grid the story vectors use), so we
can report per-layer and pick the layer the story vector lives at. Reuse
the existing hook/extraction machinery; add a short
`scripts/extract_template_activations.py` (or a `--templates` mode on the
story extractor) writing `activations/<model>-templates/…`.

### 3. Analyses — `scripts/validate_intensity_semantic.py`

For each emotion e, project template activations onto the **story-derived**
vector `v_e` (from `steering_vectors/<model>-story/`), at the story
vector's layer:

1. **Semantic-rank tracking (primary).** Spearman ρ between the projection
   and the **expected semantic rank** (not raw X), per emotion, with
   bootstrap 95% CI (n=10_000 per stats conventions). Success = strong ρ in
   the correct sign.
2. **Surface-divergence subset (the decisive test).** Restrict to the
   inverse / non-monotonic families and show the projection tracks the
   **semantic rank**, not **raw X**: report ρ(proj, semantic_rank) vs
   ρ(proj, raw_X). A surface/digit account predicts the latter; the
   concept account predicts the former. Divergence here is the cleanest
   concept evidence.
3. **Cross-format transfer.** Train the H1 probe on **stories**, test on
   **templates** (high- vs low-intensity), report AUC; contrast against a
   **story-trained TF-IDF tested on templates** (expected ≈0.50 — stories
   and templates share almost no vocabulary). Activation transfer ≫ surface
   transfer = concept.
4. **Neutral-template control.** Projection on the no-valence number family
   should be flat (slope ≈ 0).

Report to `results/intensity_semantic/<model>/report.md` (gitignored),
parallel to the confound audit.

## Success criterion (the gate this adds)

For a model to clear the concept-vs-surface question:

- semantic-rank Spearman |ρ| ≥ **0.6** (correct sign), **and**
- it **survives on the surface-divergent subset** — ρ(proj, semantic_rank)
  clearly exceeds ρ(proj, raw_X), **and**
- story→template **activation** transfer materially exceeds story→template
  **surface** transfer (which sits near chance).

Meet all three ⇒ the story-derived vectors encode emotion **semantics
beyond lexis**, clearing the flag the cross-topic audit left open, and H1
on the primaries is defensible. Fail ⇒ the story separability is
substantially surface lexis; revisit construction (or scope the H1 claim
down) before primary spend.

## Open questions / tradeoffs (surface, don't silently pick)

- **Readout position** — last-user-token vs mean-over-template vs the exact
  assistant-colon analog per model's chat template. Proposal above; confirm
  before freezing.
- **Author vs reuse** — admiration/loathing need newly-authored numeric
  templates (the paper's examples skew sad/anxious). Hand-author (avoids the
  paraphrase confound) and keep ≥1 inverse family each.
- **Per-layer aggregation** — report the story vector's layer as headline;
  show the full layer curve as support.
- **How many families/values** — start ≥4 families × ≥6 X per emotion;
  bump if bootstrap CIs are wide.
- **Dev vs primary** — run on the dev fleet first (cheap, validates the
  harness), but the criterion that matters is on the **primaries**; small
  models may track intensity weakly even when the concept is real.

## Out of scope

- No change to H1's metric/falsifier (still the L2 logistic probe AUC).
  This is a **validation** alongside it, reported as C2-style background.
- No steering/behavioral work here.
- No new HYPOTHESES amendment (the control is already sanctioned).

## First action if approved

New session: author + freeze `data/public/intensity_templates.jsonl`
(`build_intensity_templates.py`, MD5-locked), then
`extract_template_activations.py`, then `validate_intensity_semantic.py`;
dry-run the analysis on the existing Qwen 0.5B story vectors before any new
GPU run.
