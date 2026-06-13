# Decision record: story method becomes the primary emotion-vector derivation

**Date:** 2026-06-13
**Status:** Decided (PI). Recorded as a HYPOTHESES.md amendment dated 2026-06-13.
**Supersedes:** the CAA-primary framing in `docs/methods.md` and the
"CAA vs story-method primacy — decision, not yet an amendment" item in
this plan (Priority 2.3).

## Context

H1 claims a linear probe reads emotion identity off residual-stream
activations. Whether a high AUC means anything depends on the probe
tracking the emotion *concept* rather than a surface artifact. The
project had two derivations:

- **CAA (was primary):** hand-authored short vignettes (13–17 words),
  **last-token** activation, vector = mean(emotion) − mean(neutral).
- **Story method (was parallel/validation):** model-generated stories,
  **token-50 mean-pool**, cross-emotion centering, neutral-PC
  projection. Implemented in `src/llm_psych/steering.py` (98 tests).

## What forced the decision

`scripts/audit_h1_confounds.py` on the four-emotion seed set
(`results/h1_confound_audit/report.md`):

- **Length tell:** neutral ~13 words vs emotion ~16–17; length-only
  classifier AUC up to **0.93** (admiration worst).
- **Lexical separability:** word+char TF-IDF AUC **0.86–0.97** — the
  classes are largely separable from the text alone.

These are the surface confounds Sofroniew et al. (2026) explicitly
designed against. Their controls — topic-matched self-generated stories,
emotion-word banned, token-50 mean-pool, cross-emotion centering,
neutral-PC projection, plus numerical-intensity semantic validation —
are precisely the story-method construction. The CAA last-token
short-vignette path diverges from the paper on every choice that matters
for surface-robustness.

## Options considered

**A — Harden CAA toward the paper.** Topic-match + length-balance the
vignettes, mean-pool, add neutral-PC projection. Rejected as primary: it
re-implements the story method badly with short prompts, requires
re-authoring all four stimulus sets, and still pools over 13–17 tokens.

**B — Promote the story method to primary (chosen).** It already bakes
in the paper's confound controls, is built and unit-tested, and is the
faithful replication. Costs: GPU-heavier (each model generates its
corpus), never GPU smoke-tested, story quality depends on model
capability (weak dev models), and it needs a pre-registration amendment.

## Decision

Story method = **primary** derivation for H1/H2/H3. CAA = **secondary
baseline**, run for CAA↔story comparison (robustness). **H1 stays a
logistic probe** — only its activation source changes to token-50-mean
story activations (activation-source change, not probe-type change),
keeping the "linear probe accessibility" claim intact.

## What carries over (not wasted)

- The four primary emotions (admiration, joy, loathing, sadness).
- The hand-authored vignettes → now the CAA-baseline stimuli and the
  implicit-emotion read-out set.
- `audit_h1_confounds.py` → now the validation gate on the story
  construction (run its cross-domain control on story activations).
- The deferral of LLM augmentation.

## Consequences / open work

1. **Implementation gap:** `scripts/train_probes.py` reads CAA
   last-token `<emotion>_{train,test}.npz`; extend it to consume the
   story pipeline's pooled per-story activations
   (`activations/<model_key>-story/<emotion>.npz`) for the primary H1.
2. **Topic-matched story stimuli:** generate the four-emotion corpora
   over one shared topic list (`data/public/story_topics.txt`).
3. **GPU smoke-test the story pipeline** (never run) — see below.
4. **Add the numerical-intensity-template validation** (semantic vs
   surface) as a stronger control than cross-domain.

## Story-pipeline GPU smoke-test plan (do first)

Goal: prove `generate → extract_story → derive_story` runs end-to-end on
one dev model before any primary run.

1. Model: `qwen25_05b` (or `gemma2_2b`) on the Mac / cheapest GPU.
2. `generate_emotion_stories.py model=qwen25_05b derivation=story`
   `emotion=admiration,joy,loathing,sadness,neutral` with a tiny budget
   (`derivation.generator.n_stories_per_emotion=5`,
   `derivation.generator.min_story_tokens=10`). Confirm parquet rows and
   that the emotion word is absent.
3. `extract_story_activations.py` same args — confirm
   `activations/<model>-story/<emotion>.npz` shapes `(n_stories, hidden)`
   per layer, and that stories ≤ pool_start_token are dropped (sanity at
   such a small token budget).
4. `derive_story_steering_vectors.py` — confirm per-layer vectors +
   `manifest.yaml`.
5. Run `audit_h1_confounds.py --model-key <...>-story` once a probe path
   exists, to check the story construction de-confounds (shuffle-null
   ≈ 0.5, cross-domain ≥ 0.80).

Only after the smoke-test passes: full four-emotion corpora on the dev
fleet, then the 8B primaries. The H1 confound audit remains Priority 1.
