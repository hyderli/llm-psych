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
