# Plan: next steps

**Status:** proposed, awaiting approval. Per CLAUDE.md workflow, after
approval start a NEW session that reads only this file.
**Author/date:** drafted 2026-06-12.
**Context inputs:** repo state, `docs/methods.md` (this project's
adapted methodology), `HYPOTHESES_original.md` + `docs/methods_original.md`
(reconstruction of Sofroniew et al. 2026), `HYPOTHESES.md` (locked
pre-registration), `RESEARCH_LOG.md`.

---

## Where we actually are

Scaffolding is largely built and unit-tested; almost nothing is fit on
real data.

- **Built:** CAA + story-method steering primitives (`src/llm_psych/
  steering.py`, 98 tests pass), Hydra configs (6 models, ~20 emotions,
  `derivation: {caa, story}`), frozen behavioral stimuli
  (`blackmail_scenarios.jsonl`, `reward_hacking_scenarios.jsonl`),
  prompt augmentation, story pipeline (generate/extract/derive).
- **Fit on real data:** only **H1 probes on the Qwen 2.5 0.5B dev
  model**. `results/` is empty. No steering vectors, no behavioral
  runs, no primary-model probes.
- **Not yet run:** story pipeline has never been GPU smoke-tested.

### The blocker that gates everything

The dev-model H1 probes report **AUC = 1.0 at every layer, including
layer 12** (`probes/Qwen2.5-0.5B-Instruct/anger_summary.csv`). Perfect
separation on a 0.5B model is a **confound signature**, not a result.
Contributing facts:

- Neutral prompts average **15.9 words** vs **~19–20** for emotion
  prompts → a length tell.
- **3,500 of 3,750** augmented prompts are GPT-paraphrases
  (`source=paraphrase`) → per-emotion stylistic fingerprints a linear
  probe will exploit.
- The original paper hit this same risk and mitigated it by projecting
  out neutral-transcript PCs (`methods_original.md` §1). **The project's
  CAA probe path has no equivalent confound control** — it is a
  deliberately "clean" L2 logistic regression (`methods.md` §Probes).

Until we know the probes track emotion *concepts* and not artifacts,
every downstream number (steering vectors, H2/H3 behavioral effects) is
suspect. This is the top priority.

---

## Priority 1 — H1 confound audit (do first; cheap, dev fleet only)

**Goal:** decide whether H1 probe separability is real or an artifact,
before any cloud spend.

**Tasks:**
1. **Shuffle-label null.** Refit probes with permuted emotion labels
   (same activations). Expected AUC ≈ 0.5. If shuffled AUC stays high,
   the pipeline leaks. Run ≥ 100 permutations for a null distribution.
2. **Length / surface-feature control.** Regress probe score on
   `length_words` and a bag-of-words/char-n-gram baseline. Report how
   much of the separability a trivial surface classifier already buys.
   Re-balance neutral vs emotion prompt length and re-fit.
3. **Cross-domain generalization.** Train on a subset of `category`
   domains (work, health, …), test on **held-out domains**. A concept
   probe should transfer; an artifact probe will collapse.
4. **Paraphrase-source control.** Train on `hand_authored` only, test on
   `paraphrase` (and vice versa). Large gap ⇒ paraphrase fingerprint.
5. **Neutral-PC projection option for the CAA path.** Port the story
   method's `fit_neutral_pcs` / `project_out` (already in
   `steering.py`) as an optional pre-probe step; compare AUC with/
   without. Decide whether the "clean baseline" probe needs it.

**Deliverable:** `results/h1_confound_audit/` with a `report.md` and a
recommendation: (a) probes are sound as-is, (b) need neutral-PC
projection, or (c) stimuli need re-balancing/re-freezing (which would
require a HYPOTHESES.md amendment block, since stimuli are locked).

**Success criterion:** shuffled-label AUC ≈ 0.5 **and** cross-domain
test AUC ≥ 0.80 on the dev model, before trusting any real-AUC claim.

---

## Priority 2 — Reconcile steering design with the paper

Surface these in a HYPOTHESES.md amendment block *before* fitting any
steering behavioral data.

1. **Steering granularity (emotion set).** RESOLVED 2026-06-12: the
   project primary set is fixed to **{admiration, joy, loathing,
   sadness}** — two opposite pairs (admiration ↔ loathing, joy ↔
   sadness). Seed stimuli for admiration/loathing are authored;
   `emotion_prompts.parquet` is regenerated. NOTE the consequence for
   the C7/C8 replication: the paper's most-causal blackmail/reward-
   hacking emotions (desperate, calm) are **not** in the project set, so
   H2/H3 test loathing as the misalignment-relevant negative pole rather
   than desperate. If a closer C7 replication is wanted later, desperate/
   calm can be added back via amendment. Augmentation of the four to the
   500/200 target is **deferred and may be unnecessary** — LLM paraphrase
   is implicated in the AUC=1.0 confound, so run the H1 audit (Priority 1)
   on the 50 seeds first; scale by hand-authoring, not paraphrase, if more
   N is justified.
2. **Steering strength.** `methods.md` fixes α = 1.0 × mean
   residual-norm; the paper steers at **≤ 0.1 × norm** with
   dose-response sweeps from −0.1 to +0.1 (`methods_original.md` §3, §6).
   1.0× is ~10× the paper's strongest setting and risks degenerate
   outputs that masquerade as "no effect." **Replace the fixed α with a
   sweep** and report the curve; this also serves the n≥200 scale-test
   falsifier more cleanly. Requires an amendment (changes the
   operationalization in H2/H3).
3. **CAA vs story-method primacy.** The paper used the **story method**;
   the project pre-registers **CAA** as primary. For the direct
   replication claim, run both and compare per-emotion vector cosine
   similarity + behavioral effect; consider story-method co-primary for
   the C7 contrast. Decision, not yet an amendment.

---

## Priority 3 — Execution order (only after P1 passes)

Keep CLAUDE.md's pilot-vs-scale discipline: any steering effect at
n < 100 is suggestive only and must be re-tested at n ≥ 200.

1. **Story-pipeline smoke test** on Qwen 0.5B (`n_stories_per_emotion=5`,
   `min_story_tokens=10`) on the Mac — verify the three scripts run
   end-to-end. (Open TODO in RESEARCH_LOG.)
2. **Finish dev-fleet H1** (Llama 3.2 1B, Gemma 2 2B) with the audit
   from P1 baked in.
3. **Pin `configs/model/gemma3_4b.yaml` HF revision SHA** (open TODO).
4. **Decide primary third family** — Mistral 7B v0.3 vs OLMo-2 7B (still
   pending in HYPOTHESES.md amendments).
5. **Cloud GPU:** full activation → probe → steering → behavioral
   pipeline on the 8B primaries (Llama 3.1 8B, Qwen 2.5 7B, + third).
   H2 blackmail at n ≥ 200/condition with target/zero/random/orthogonal
   controls; the random-vector control is non-negotiable.

---

## Scope update — 2026-06-12

Four PI decisions amended into `HYPOTHESES.md` (see the three dated
2026-06-12 amendment blocks):

1. **Sycophancy re-added as CO-PRIMARY (H7)**, co-equal with blackmail.
   Paper-faithful stimuli (Sonnet 4.5 system-card sycophancy eval +
   harshness score), steer ≤ 0.1 × norm, loving(→compassionate/blissful)
   /calm vs desperate/angry/afraid, same controls as H2, n ≥ 200. Build
   `src/llm_psych/tasks/sycophancy.py` and freeze the item set + rubric
   before any fit.
2. **Activity preferences / Elo demoted to tertiary / exploratory** —
   out of H3 (reward hacking is now the sole secondary). Reframed as a
   cheap validation (probe–Elo correlation + steering-shifts-Elo) with
   its own FDR family. This is where the team's current Elo / activity-
   analysis to-dos belong.
These do not change Priority 1 (the H1 confound audit still gates
everything). Add to execution order: a sycophancy task pipeline (P3, GPU
phase, n ≥ 200, alongside blackmail).

## Out of scope / do not do

- **Do not** revive the 2014 Christensen / Asch-style two-step
  sycophancy design. Sycophancy is back in scope as of 2026-06-12, but
  **only** via the paper's system-card eval + harshness design; the
  PI's prior two-step pipeline stays out.
- **Do not** edit `HYPOTHESES.md` to fit `HYPOTHESES_original.md`. The
  latter documents the source paper only; the former is the locked
  pre-registration. Any change to locked hypotheses needs a dated,
  justified amendment block.
- **Do not** re-filter or re-freeze stimuli silently — amendment +
  MD5-hash update required (`methods.md` §Stimulus locking).

---

## First action if approved

Start a new session on **Priority 1**. Build
`scripts/audit_h1_confounds.py` (shuffle-label null + cross-domain +
paraphrase-source splits + optional neutral-PC projection), run it on
the existing Qwen 0.5B activations, and write
`results/h1_confound_audit/report.md`. Cheapest step that could
invalidate the most.
