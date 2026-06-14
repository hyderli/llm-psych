## 2026-04-28 — foundation modules done

**Did:**
- Implemented src/llm_psych/hooks.py + tests (ResidualStreamRecorder,
  ResidualStreamSteerer, ~25 unit tests on a fake model).
- Implemented src/llm_psych/models.py — load_model() wrapping HF
  AutoModelForCausalLM with consistent defaults across Llama 3.1 8B,
  Qwen 2.5 7B; MPS-aware; 4-bit gated behind explicit flag.
- Implemented src/llm_psych/probes.py — sklearn LogisticRegression
  wrapper with bootstrap AUC CI, Brier score, save/load, CAA mean-diff
  steering vector. Plus tests/test_probes.py.

**Open TODOs flagged but not blocking:**
- Pin HF revision SHAs in configs/model/*.yaml before HYPOTHESES.md lock.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- ~~Pick reward-hacking benchmark; pick blackmail scenario source.~~
- No `from_config` Hydra helper yet — add when first script needs it.

**Next:** write stimuli. 50 prompts × 4 emotions + 50 neutrals. Hand-
authored or curated from GoEmotions. Save to
data/public/emotion_prompts.parquet. This is research design, not coding.

**Energy:** good day. Three coding sessions, all green-tested, all
committed. Stopping here rather than starting stimuli tired.

## 2026-05-15 — behavioral benchmarks selected and authored

**Did:**
- Selected reward-hacking benchmark: custom 60-item single-turn multiple-
  choice benchmark (`data/public/reward_hacking_scenarios.jsonl`),
  inspired by MACHIAVELLI's annotated choice-nodes (Pan et al. 2023)
  but adapted to single-turn format for 7-8B models. Five categories:
  grader_bias (12), metric_gaming (12), proxy_exploitation (12),
  resource_allocation (12), compliance_gaming (12).
- Selected blackmail scenario source: Anthropic's Oct 2025 agentic-
  misalignment paper (arXiv:2510.05179). Authored 50 parameterized
  variants (`data/public/blackmail_scenarios.jsonl`) based on their
  validated "Alex the email-oversight agent" structure, with 5 company
  types × 5 executive names × 2 compromising-info types × 2 threat
  framings, fully crossed.
- Verified both datasets: 60 and 50 unique items respectively, all JSON
  valid, schema consistent.
- Updated HYPOTHESES.md with formal amendment block locking benchmark
  selections per pre-registration rules.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** stimuli curation (emotion prompts).

**Energy:** focused session. Two large dataset files authored and verified.

## 2026-05-15 — emotion prompts curated and frozen

**Did:**
- Hand-authored 250 emotion-labeled text prompts and saved to
  `data/public/emotion_prompts.parquet`. 50 per emotion (joy, fear,
  anger, sadness) + 50 neutral. 35 train / 15 test per emotion (70/30).
- Schema: `id`, `prompt`, `emotion_label`, `split`, `category`,
  `length_words`, `source`. Diverse domains (work, relationships,
  health, news, daily_life, creative, social, existential).
- Deliberately avoided explicit emotion words in non-neutral prompts
  to minimize lexical confounds. Zero prompts contain explicit
  emotion words (verified by script).
- Prompt lengths balanced: mean 10–21 words, std 1.6–2.3, range 8–21.
- Created reproducible generation script
  `scripts/build_emotion_prompts.py` with seeded shuffle.
- Updated HYPOTHESES.md with amendment block documenting emotion probe
  stimuli, augmentation plan, and quality controls.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** return to model selection / config pinning, then move to
activation extraction pipeline.

**Energy:** good momentum. All three stimulus sets (reward_hacking,
blackmail, emotion_prompts) now frozen. Ready for pre-registration
lock of HYPOTHESES.md.

## 2026-05-15 — Add Gemma 2 2B development model

**Did:**
- Added `configs/model/gemma2_2b.yaml` for `google/gemma-2-2b-it`.
  26 layers, hidden size 2304, float16 on MPS. Third architecture
  family (Gemma) for cross-family pipeline validation.
- Updated HYPOTHESES.md amendment block documenting Gemma 2B as a
  development (not primary) model.
- Updated models.py docstring to include Gemma 2B and expanded MPS
  memory note to 0.5B–2B range.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third primary target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** smoke-test activation extraction on Gemma 2B or Llama 3.2 1B
with a tiny subset of emotion_prompts.parquet.

**Energy:** continuing.

## 2026-05-25 — pre-registration amendment + Bluedot grant submitted

**Did:**
- Amended HYPOTHESES.md with three dated 2026-05-25 amendment blocks:
  (i) sycophancy removed from project scope, blackmail promoted to
  primary H2 (single-turn agentic-misalignment protocol, n ≥ 200 per
  condition via paraphrase expansion); (ii) H5 (activation-vs-self-
  report dissociation) removed entirely; (iii) judge model switched
  from GPT-4o-mini to Claude Haiku 4.5.
- Propagated all four changes through BLUEPRINT.md, docs/methods.md,
  CLAUDE.md, README.md. Two clean commits on main:
  3dc0fe4 (four pre-registration files) and c97dae7 (README).
  Pushed to origin.
- Drafted and submitted Bluedot rapid grant application. Ask: $250
  ($100 RunPod RTX 4090 ~300 GPU-hrs, $90 Anthropic API for Claude
  Haiku 4.5 judging, $60 buffer). Team: four Bluedot alumni
  (Haydar PI, Chris Bosley, Vaaruni Desai, Srujananjali Medicherla).

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third primary target model — narrowed to a Gemma family
  variant in the 7-9B range; specific variant TBD.
- Decide on activity-preferences stimulus design: keep n=100 with
  across-activity aggregate, raise to n=200 with paraphrase expansion,
  or drop the task from H3.
- Write collaborators.md after kick-off call.
- No `from_config` Hydra helper yet.

**Next:** smoke-test activation extraction on Gemma 2B or Llama 3.2 1B
with a tiny subset of emotion_prompts.parquet. Pre-registration is now
locked; pilots can proceed without further amendment risk.

**Energy:** long session; scope tightened and grant out the door.

## 2026-06-09 — RunPod pipeline live + Plutchik opposite pair amendment

**Did:**
- Brought up first real RunPod pod (RTX 3090, 25 GB VRAM, 30 GB
  /workspace volume) and validated the bootstrap → cloud_run pipeline
  end-to-end on Llama-3.1-8B-Instruct, emotion `anger`. Activations,
  probe (anger layer 16), and steering vector all pushed to
  `llm-psych/llm-psych-activations`.
- Hardened `scripts/cloud_bootstrap.sh` (PR #4): explicit dotenv path
  to avoid `find_dotenv()` stack-frame crash under heredoc, fixed the
  cloud_run.sh hint to use `--model/--emotions` flags, pinned
  `HF_HOME` to `/workspace/.cache/huggingface` to avoid root-disk
  exhaustion on first 7-8B model download.
- Added Plutchik opposite-pair extension: new emotion configs
  `loathing` (label 19) and `admiration` (label 20). Dated amendment
  block in HYPOTHESES.md (2026-06-09). EMOTION_LABELS.md updated.
  Marked exploratory, separate FDR family from primary-9.

**Open TODOs:**
- **Stimuli authoring is the binding blocker on every emotion outside
  legacy {anger, fear, joy, sadness}.** `emotion_prompts.parquet`
  contains no rows for the primary-9 set, the Ekman/Wilcox extensions,
  or loathing/admiration. Until stimuli are written and committed,
  cloud_run on those labels will fail with zero matching prompts.
  Author ~50 base prompts per emotion (then augment to ~750 via the
  existing paraphrase expander); primary-9 first, loathing/admiration
  second.
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third primary target model — Gemma family, 7-9B variant TBD.
- Activity-preferences stimulus design decision (deferred from
  2026-05-25).

**Next:** decide whether to author primary-9 + Plutchik-pair stimuli
locally or via a one-shot LLM script reviewed against the
stimulus-design rubric in `docs/methods.md`. Pod can be stopped while
stimuli are authored — no GPU work to do until the parquet has rows
for the target labels.

**Energy:** infra unblocked; the next bottleneck is content, not
compute.

## 2026-06-10 — story-method (original paper) emotion-vector pipeline

**Context:** The original paper's derivation procedure differs from the
project's pre-registered CAA pipeline in four ways: (1) activation
source (model-generated stories vs. hand-authored vignettes), (2) token
aggregation (mean from token 50 vs. last-token), (3) centering formula
(cross-emotion mean vs. emotion-vs-neutral), (4) confound removal
(neutral-PC projection-out vs. none). Both methods are needed: the CAA
pipeline is the pre-registered H1/H2 primary path; the paper method is
a validation / ablation parallel.

**Did:**
- **PR #3 fix-up and merge** — `cb-gemma-emotions-scripts` prototype
  by Chris Bosley had a literal `SyntaxError` (unfilled path
  placeholders) and an out-of-package import. Fixed both, added
  provenance docstrings, merged via `--merge` to preserve Chris's
  authorship. The prototype is reference material for the existing
  `steering_vectors/gemma3-4b-story` HF artifact.
- **`src/llm_psych/steering.py`** — pure-NumPy primitives:
  `derive_story_vectors` (cross-emotion-mean centering),
  `fit_neutral_pcs` (SVD-based orthonormal basis),
  `project_out` (Gram-Schmidt orthogonal removal), and
  `derive_paper_steering_vectors` (end-to-end composition). All
  functions operate per-layer. 21 unit tests covering shapes, dtypes,
  centering correctness under sample-size imbalance, PC orthonormality,
  explained-variance thresholding, projection idempotence, and
  end-to-end orthogonality. 98/98 tests pass.
- **`configs/derivation/{caa,story}.yaml`** — Hydra group selecting
  between the two derivation methods. Default `caa`; `story` overrides
  all relevant params (pool, center, project_out, n_stories,
  generator settings). Verified composition with both overrides.
- **`scripts/generate_emotion_stories.py`** — Hydra entry point.
  Self-generated story corpus per model (each model writes its own
  narratives). 30 mundane topics in `data/public/story_topics.txt`
  (frozen/committed). Prompt templates from the paper: 150-word,
  narrator perspective, no emotion-word constraint. Stories shorter than
  `min_story_tokens` (default 60) are dropped so pooling from token 50
  is meaningful. Output: parquet + JSON manifest per
  `(model, emotion)`.
- **`scripts/extract_story_activations.py`** — Extract + pool
  activations from generated stories. Uses
  `ResidualStreamRecorder(token_position=slice(pool_start, None))`;
  mean-pools across the sliced sequence dimension per story. One
  story at a time (avoids padding ambiguity). Output: `.npz` per
  `(model, emotion)` with one array per layer, shape `(n_stories,
  hidden_dim)`.
- **`scripts/derive_story_steering_vectors.py`** — End-to-end
  derivation. Reads all emotion `.npz` files plus `neutral.npz`, runs
  the paper's full method (centering + PC projection-out) per layer,
  saves one `.npy` per `(emotion, layer)` plus `manifest.yaml`
  capturing model SHA, layers, var_threshold, pool_start_token, git
  SHA, and per-emotion story counts.
- **`configs/emotion/neutral.yaml`** — enables `emotion=neutral` via
  Hydra for the story pipeline (needed for the neutral baseline
  generation and for the PC-fitting step in derivation).
- **`docs/methods.md`** — added "Direction extraction (story method)"
  subsection documenting the full three-script pipeline, the math, and
  the config switch.

**Files added/changed:** 12 files, ~1300 insertions across
`src/`, `tests/`, `scripts/`, `configs/`, `docs/`, `data/public/`.

**Open TODOs:**
- Add `configs/model/gemma3_4b.yaml` with a pinned HF revision SHA.
- Smoke-test the story pipeline end-to-end on a small model
  (e.g. Qwen 0.5B with `n_stories_per_emotion=5` and
  `min_story_tokens=10`) on the Mac to verify the scripts actually
  run (parse + Hydra composes already verified; story quality
  validation needs GPU).
- Decide whether to run the full story pipeline for all 3 primary
  models (Llama 3.1 8B, Qwen 2.5 7B, Gemma3 4B) or limit to the
  already-artifacted Gemma3 4B first.
- The Bluedot rapid grant ($250) is pending decision.

**Energy:** long multi-session push. Core pipeline is implemented and
unit-tested. Ready for cloud GPU smoke-test.

## 2026-06-12 — scope reconciliation: two paths disentangled

**Context:** The behavioral-replication path (locked pre-registration:
blackmail/CAA primary) and the paper-mechanism path (story-method
vectors, geometry/PCA, Elo preferences) had drifted into each other —
the team's weekly to-dos were all on the mechanism path while the locked
doc made it secondary/unspecified. PI ruled on four points; all precede
any behavioral-data fit, so they are pre-fit amendments.

**Decisions (amended into HYPOTHESES.md, 3 dated 2026-06-12 blocks):**
- **Sycophancy re-added as CO-PRIMARY (new H7)**, reversing the
  2026-05-25 removal. Now two primary behavioral tasks (blackmail H2 +
  sycophancy H7). Stimuli follow the paper: Sonnet 4.5 system-card
  sycophancy eval (delusional-belief pushback) + companion harshness
  score; sycophancy↔harshness tradeoff (Sofroniew C9). Steer ≤ 0.1×
  norm sweep; loving proxied by compassionate/blissful (no `loving`
  config); same three controls as H2; n ≥ 200. The 2014 Christensen/
  Asch two-step design stays out of scope.
- **Activity preferences / Elo demoted** secondary → tertiary/
  exploratory. Reward hacking is now the sole H3 secondary task. Elo
  reframed as cheap validation (probe–Elo correlation + steering-shifts-
  Elo), own FDR family, no falsifier — the correct home for the team's
  current Elo / activity-analysis to-dos.
- **Story-method "fork" investigated and dismissed.** Checked the local
  repo, all git branches, and (attempted) HF: there is **no donor-
  generated story corpus** — the only story code is self-generation
  (each model writes its own stories). The Haiku-generated content that
  exists is `emotion_prompts_augmented.parquet` (3,500 paraphrased
  emotion *prompts* for the CAA probe path), not stories. A draft had
  documented a "Plan A / Plan B" fork and a cross-model-comparability
  limitation; both were removed. Rationale: the geometry work is
  **within-model** (existence of a recoverable emotion vector per model,
  not cross-model vector similarity), and within a model the story
  design already balances topic (shared `story_topics.txt`) and strips
  generic content (neutral-PC projection), so self-generation poses no
  meaningful confound there. No story-method change; method stands as
  self-generation.

**Files changed:** `HYPOTHESES.md` (H7 added; H3 + sample-size + MC plan
+ stopping rule updated; sycophancy + Elo-demotion amendment blocks),
`docs/methods.md` (sycophancy eval subsection; activity-prefs
relabelled), `plans/next-steps.md` (scope update + out-of-scope
correction), `BLUEPRINT.md` + `CLAUDE.md` (two-primary-task sync).

**Open TODOs (new):**
- Build `src/llm_psych/tasks/sycophancy.py`; obtain/adapt the system-
  card sycophancy items + author sycophancy & harshness rubrics; freeze
  before first fit.
- Add a `loving` operationalization decision note (compassionate vs
  blissful) to the emotion configs.
- Priority 1 (H1 confound audit) still gates all of the above.

**Energy:** bookkeeping pass — the work was already underway; this gives
it a pre-registered home.

## 2026-06-12 — fix the four primary emotions + author admiration/loathing stimuli

**Decision:** project confirmatory emotion set is now exactly four —
**admiration, joy, loathing, sadness** — two opposite pairs (admiration
↔ loathing on Plutchik trust/disgust; joy ↔ sadness on valence), neutral
as reference. Supersedes the H2 {anger, fear, joy, sadness} set and the
`EMOTION_LABELS.md` primary-9. Loathing/admiration promoted from
exploratory to confirmatory; anger/fear dropped from the primary set
(stimuli retained for reference). Pre-data amendment.

**Did:**
- **Authored stimuli.** 50 hand-authored, emotion-implicit seed prompts
  each for admiration and loathing in `build_emotion_prompts.py`,
  matching the existing schema and the no-explicit-emotion-word rule
  (script check: 0/50 flagged for all four emotions). Rebuilt `main()`
  for the four-emotion set; regenerated `data/public/emotion_prompts.parquet`
  (250 rows; 35/15 split; length means 13–17 words). md5 now
  `ffd067e6346cd597dbc91e791415d115`.
- **Configs.** `admiration.yaml`/`loathing.yaml` updated (authoring no
  longer pending); `joy.yaml`/`sadness.yaml` repointed from the
  augmented parquet to the seed parquet so all four share one balanced
  50/emotion source until augmentation is re-run. Synced banned-word
  dicts in `build_` and `augment_emotion_prompts.py`.
- **Pre-registration.** Dated 2026-06-12 amendment in HYPOTHESES.md;
  updated H2 emotion conditions, H6 specificity mapping, and flagged the
  H7 sycophancy conflict (paper uses loving/calm/desperate/afraid, none
  in the four-set) with a default recast (admiration/joy vs
  loathing/sadness) pending PI confirmation.
- **Propagated** to `EMOTION_LABELS.md`, `docs/methods.md`,
  `BLUEPRINT.md`, `CLAUDE.md`, `plans/next-steps.md`, and the emotion-
  list defaults in `cloud_run.sh`, `run_smoketest_qwen05b.sh`,
  `cloud_bootstrap.sh`, `generate_emotion_stories.py`, `config.yaml`.

**Resolved this session:**
- **H7 emotions** — recast on the **admiration ↔ loathing** pair (same
  trust/disgust stance axis as the paper's warmth-vs-threat axis):
  admiration ↑ sycophancy, loathing ↑ harshness. Joy/sadness exploratory
  add-ons. H7 scoped as a conceptual extension of C9, not a literal
  replication. The paper's loving/calm/desperate/afraid are not used.
- **Augmentation** — deferred and likely unnecessary. LLM paraphrase is
  implicated in the dev AUC=1.0 confound (per-emotion style fingerprint),
  so the H1 confound audit runs on the 50 hand-authored seeds first. If
  more N is justified, hand-author more seeds rather than LLM-augment.
  All four emotion configs stay on the seed parquet (balanced 50/emotion).

**Open TODOs:**
- Run the H1 confound audit (Priority 1) on the 50-seed-per-emotion set
  before any augmentation or scale decision.
- Existing local Qwen 0.5B activations are for `anger` (legacy) — re-
  extract for the four primary emotions on the dev fleet.
- Build `src/llm_psych/tasks/sycophancy.py` + freeze the sycophancy /
  harshness rubrics (H7).

**Energy:** scope-defining change; stimuli authored, set locked to four.

## 2026-06-13 — H1 confound audit built; story method promoted to primary derivation

**Did:**
- **Built `scripts/audit_h1_confounds.py`** (Priority-1 diagnostic).
  Tier A (surface baselines: length-only + word/char TF-IDF) runs on the
  prompt parquet without activations; Tier B (shuffle-null, cross-domain
  split, paraphrase-source split, neutral-PC projection) runs when
  activations exist. steering.py loaded lazily by path so Tier A needs no
  torch. Verified Tier B logic against the legacy augmented `anger`
  activations (meta-join, cross-domain, neutral-PC all work; reproduces
  AUC=1.0, and neutral-PC projection does NOT fix it — corroborating the
  augmentation-confound concern).
- **Ran the surface tier on the four-emotion seed set.** Flagged a real
  confound: neutral ~13 words vs emotion ~16–17 (length-only AUC up to
  0.93, admiration worst); TF-IDF surface AUC 0.86–0.97. Output in
  `results/h1_confound_audit/`.
- **Decision (PI): story method → PRIMARY derivation; CAA → secondary
  baseline.** The audit confirmed the CAA short-vignette path is
  surface-confounded in exactly the ways Sofroniew et al. designed
  against; the story method already implements their controls
  (topic-matched stories, emotion-word ban, token-50 mean-pool,
  cross-emotion centering, neutral-PC projection). H1 stays a logistic
  probe but on token-50-mean story activations (activation-source change,
  not probe-type). Dated 2026-06-13 amendment in HYPOTHESES.md; decision
  record at `plans/derivation-primacy-decision.md`. Propagated to
  methods.md, config.yaml (default `derivation: story`), next-steps.md.

**Open TODOs:**
- Extend `scripts/train_probes.py` to consume story-pipeline pooled
  activations (`activations/<model>-story/<emotion>.npz`) for primary H1.
- Topic-match the four-emotion story corpora over a shared topic list.
- GPU smoke-test the story pipeline (never run) — see the decision record.
- Add the numerical-intensity-template semantic-vs-surface validation.
- Re-run the audit's cross-domain control on story activations once they
  exist (Priority 1 gate, now on the story construction).

**Energy:** the audit paid for itself immediately — caught the confound
pre-GPU and redirected the project onto the paper-faithful construction.

## 2026-06-13 — story pipeline green; train_probes on story acts; topic-matched generation

**Did:**
- **Smoke-tested the story pipeline end-to-end** on Qwen 0.5B
  (`run_smoketest_story_qwen05b.sh`). Flushed out five issues before any
  real run: clean-git pre-flight tripping on untracked `_original.md`
  (now committed); a `model.generate()` BatchEncoding bug on newer
  transformers; `device_map` dispatch hanging `generate()` on Mac
  (`load_model` now bypasses device_map for single-device dev loads); and
  the real one — `_generate_one` used an assistant-prefill chat template
  with `add_generation_prompt=False` that closed the turn (≈0 tokens),
  which an unbounded retry loop then spun on forever. Fixed the template
  (single user turn, `add_generation_prompt=True`) and capped the loop.
  Pipeline now produces stories → pooled activations → vectors.
- **Wired `train_probes.py` for story activations** (primary H1): reads
  `activations/<model>-story/<emotion>.npz`, seeded per-story 70/30 split,
  same L2 logistic probe, saves to `probes/<model>-story/`. Verified the
  load→split→fit→evaluate flow on the smoke activations. CAA unchanged.
- **Topic-matched story generation (paper-style).** Replaced the implicit
  round-robin `n_stories_per_emotion` with explicit `stories_per_topic`:
  nested topics × k with bounded per-slot retries, so every emotion +
  neutral covers the same topics equally (topic can't separate emotions).
  Added `max_topics` (smoke cap). Expanded `story_topics.txt` to 46
  ambiguous everyday topics. Manifest records `stories_per_topic` +
  `topic_matched`.

**Open TODOs:**
- Run the dev-fleet story corpora at a real `stories_per_topic`, then the
  confound audit on the story activations (shuffle-null ≈ 0.5, cross-domain
  ≥ 0.80) — the Priority-1 gate, now on the story construction.
- Build `src/llm_psych/tasks/sycophancy.py` + rubrics (H7).

**Energy:** plumbing solved and de-confounding construction in place;
next bottleneck is a real GPU run + the audit on story activations.

## 2026-06-13 — Third primary fixed: Gemma 2 9B; gemma3_4b demoted to exploratory

**Did:**
- **Resolved the long-pending third primary slot.** Set it to
  `google/gemma-2-9b-it` (dated amendment in HYPOTHESES.md), superseding
  the "Mistral 7B v0.3 or OLMo-2 7B (pending)" note from 2026-05-15. The
  three primaries are now Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 9B. Chosen
  for dense+bf16 parity with the other two (clean cross-model steering
  comparison), fits a 4090 24 GB without quant (~18.5 GB), stays in the
  7-8B framing, and maps cleanly from the Gemma 2 2B dev canary. Rejected:
  Gemma 3 4B (below bar, vision-language), Gemma 3/4 12B (too big / needs
  quant), Gemma 4 E4B (MoE, architectural confound).
- **Caught and fixed a config/prereg drift.** `configs/model/gemma3_4b.yaml`
  had claimed to be "one of the three primary targets (Gemma family, 7-9B
  range)" — false on both counts (it is 4B; HYPOTHESES.md never named
  Gemma a primary). Relabelled it EXPLORATORY/prototype (within-Gemma
  scale ladder 2B->4B->9B), separate FDR family.
- **New pinned config** `configs/model/gemma2_9b.yaml` (SHA
  11c9b309..., n_layers 42, hidden_size 3584, bf16, device_map auto).
- Added gemma2_9b to the Priority-3 primary run list in
  `plans/story-gate-run.md`.

**Open TODOs:**
- (carries over) build the real-scale story runner + story-aware confound
  audit, then run the dev-fleet gate on RunPod (see
  `plans/story-gate-run.md`).
- On first Gemma 2 9B load, confirm n_layers=42 / hidden_size=3584 against
  the loaded model (sanity check the pinned config).

**Energy:** primary roster now complete and internally consistent; the
gate-run plan is the next build.

## 2026-06-14 — Dev-fleet story gate run on RunPod; within-corpus probing is surface-saturated

**Did:**
- **Built the gate tooling** (`scripts/run_story_pipeline.sh` real-scale
  pod runner; story-activation path in `scripts/audit_h1_confounds.py` with a
  cross-TOPIC control + story-text surface baseline).
- **Ran the dev fleet on a RunPod 4090** at the real budget
  (`stories_per_topic=7`, 46 topics → 322 stories/emotion): Qwen 2.5 0.5B,
  Llama 3.2 1B, Gemma 2 2B. Activations + probes + steering vectors for all
  three `*-story` keys are on the HF dataset.
- **Confound audit on story activations** (`results/h1_confound_audit_story/`):

  | model | null | length-only | surface TF-IDF | real probe | cross-topic |
  |---|---|---|---|---|---|
  | Qwen 0.5B | 0.51 | 0.45–0.51 | 0.91–0.95 | 0.94–0.98 | 0.87–0.93 |
  | Llama 1B  | 0.51 | **0.67–0.79** | **1.00** | **1.00** | 1.00 |
  | Gemma 2B  | — (audit not finished; see below) | | | | |

**Findings:**
1. **Pipeline is sound** — shuffle-null = 0.51 on both completed models. No
   leak; the harness is trustworthy. Neutral-PC projection barely dents the
   probe (0.93–0.98), so the signal is orthogonal to the neutral baseline.
2. **The story construction removed the LENGTH confound on Qwen** (length-only
   0.45–0.51, vs 0.74–0.93 for the CAA vignettes) — the switch to topic-matched
   generation worked for what it was for.
3. **But within-story-corpus H1 probing is surface-saturated.** A bag-of-words
   classifier separates emotion stories from neutral from the text alone
   (Qwen 0.91–0.95, Llama 1.00), and the activation probe beats it by a
   vanishing margin (Qwen ~0.03–0.05, **Llama 0.00**). The margin *shrinks*
   with model capability (better instruction-following → more distinctively
   emotional text), so scaling to 8B will not fix it. **Within-corpus probe
   AUC cannot, by itself, evidence an emotion *concept* over emotional
   lexis.** This is the load-bearing conclusion of the gate.
4. **Length balance regressed on Llama** (length-only 0.67–0.79). Topic-matched
   generation did not length-balance robustly across models — a real, separate
   construction issue to investigate (needs the story texts → length
   distributions).

**Implication (reshapes H1):** the C1/H1 linear-decodability result on the
story corpus is necessary but not sufficient and is surface-saturated for
capable models. The **numerical-intensity semantic control (paper C2) and
causal steering become the load-bearing evidence**, not optional. Plan drafted:
`plans/numerical-intensity-control.md`.

**Open TODOs:**
- Re-run the audit (incl. Gemma) — the shuffle-null/surface fit ran for hours
  on Gemma's larger corpus because lbfgs never converges on perfectly-separable
  TF-IDF; fixed in `audit_h1_confounds.py` (capped sparse TF-IDF + liblinear).
- **Rescue or regenerate the dev story corpora** — they were stranded on the
  stopped pod (Community Cloud could not free a GPU to resume) and are NOT yet
  on HF. Fixed forward: `run_story_pipeline.sh --push` now also pushes
  `data/derived/stories/<key>` via `scripts/push_stories.py`.
- Investigate the Llama length regression (needs the corpora).
- Build the numerical-intensity control (Priority next).

**Infra notes (for teammates / future pods):**
- Default Linux torch resolves to a **cu13** wheel; cloud 4090 hosts ship a
  **CUDA 12.6** driver → torch sees no GPU. Pinned torch → cu124 on Linux in
  `pyproject.toml` (Mac unaffected via platform marker). Re-`uv lock` after.
- **Disk:** a 20–40 GB container disk is too small for multi-model runs (the
  cu13→cu124 double-download plus several model caches fill it). Use **≥50 GB**
  (or a network volume) and `export HF_HOME=/workspace/.cache/huggingface`.
- Run long jobs under **tmux/nohup**; the RunPod web terminal drops kill
  foreground jobs.

**Energy:** brutal infra day, but the science landed — a clean, useful negative
result that correctly redirects the project from within-corpus probing to the
semantic-generalization + steering evidence.
