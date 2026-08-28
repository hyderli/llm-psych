# Plan: Plutchik-wheel emotion-vector extraction

**Date:** 2026-08-28 (supersedes the hypothesis-driven version of this
file and `plans/emotion-set-expansion-design.md`)
**Scope:** Extract emotion vectors for all 32 Plutchik-wheel cells plus
neutral, on the three primary models. **That is the entire deliverable.**

Out of scope, deliberately: hypotheses, pre-registration amendments,
geometry analyses, steering, behavioral runs, and the ~2,700 rows of
implicit-scenario and intensity-template stimuli that validation would
have required. Nothing here makes a claim, so nothing here needs a
pre-registration. If these vectors are later used for a confirmatory
claim, that claim gets its own amendment at that time.

---

## 1. The blocker: the pipeline would overwrite the locked four-emotion corpora — FIXED 2026-08-28

`scripts/extract_story_activations.py:201` and
`scripts/derive_story_steering_vectors.py:124,163` hardcode the track
suffix:

```python
act_dir = _repo_root / cfg.paths.activations_dir / f"{model_key}-story"
```

and `scripts/generate_emotion_stories.py` writes corpora to
`data/derived/stories/<model_key>/<emotion>.parquet` with no track
component at all. The story pipeline takes the corpus name from
`cfg.emotion.name` and nothing else.

Those paths are occupied. On disk right now:

```
data/derived/stories/Llama-3.1-8B-Instruct/{admiration,joy,loathing,sadness,neutral}.parquet
data/derived/stories/Qwen2.5-7B-Instruct/...
data/derived/stories/gemma-2-9b-it/...
```

Five of the 33 wheel corpora share a name with an existing one: `joy`,
`sadness`, `admiration`, `loathing`, `neutral`. **Running the wheel
pipeline as it stands regenerates `joy.parquet` in place and destroys the
corpus behind the locked four-emotion vectors, the C2 validation numbers,
and the locked layers.** Story generation is stochastic, so an
overwritten corpus is not recoverable by re-running.

### Fix (do this first, it is small)

Add a track component to the three path constructions:

- `configs/config.yaml`: new key `track: story` (default preserves every
  existing path exactly).
- `extract_story_activations.py`, `derive_story_steering_vectors.py`:
  `f"{model_key}-{cfg.track}"` instead of `f"{model_key}-story"`.
- `generate_emotion_stories.py`: write to
  `data/derived/stories/<model_key>/<track>/<emotion>.parquet`, with a
  read-time fallback to the flat path so existing corpora still resolve.

The wheel run then sets `track: story-wheel32` and cannot touch anything
the four-emotion track owns.

### Status: done

- `src/llm_psych/paths.py` — `track_slug`, `story_dir`,
  `story_corpus_path`, `resolve_story_corpus`. The default track
  reproduces the pre-track layout byte-for-byte, so no existing artefact
  moves.
- `configs/config.yaml` — `track: story`.
- The three pipeline scripts use the helpers; `derive` also records
  `track` in `manifest.yaml`.
- `tests/test_track_paths.py` — 6 tests. The load-bearing one is
  `test_wheel_track_never_falls_back_to_another_tracks_corpus`: only the
  default track falls back to the historical flat layout, so a missing
  wheel corpus raises instead of silently reading four-emotion stories
  and shifting every wheel vector through the grand mean.

**Still hardcoded elsewhere** (out of scope for extraction, but they will
need the track before they can be pointed at wheel artefacts):
`run_story_pipeline.sh:129`, `validate_logit_lens.py:128`,
`validate_intensity_semantic.py:134`, `train_probes.py:205`,
`decompose_emotion_vectors.py:886`, `plot_intensity_projection.py:63`,
`h8_prep.sh`, `cloud_decompose.sh`. `run_wheel.sh` (step 3) passes
`track=story-wheel32` straight to the three pipeline scripts and does not
depend on any of them.

---

## 2. The 33 corpora

24 ring cells = 8 axes x 3 rings. Labels `100 + 10*axis + ring`
(ring 1=high, 2=middle, 3=low), so no collision with the used 1-20.

| axis | idx | high | middle | low | opposite |
|---|---|---|---|---|---|
| joy | 1 | ecstasy (111) | joy (112) | serenity (113) | sadness |
| trust | 2 | admiration (121) | trust (122) | acceptance (123) | disgust |
| fear | 3 | terror (131) | fear (132) | apprehension (133) | anger |
| surprise | 4 | amazement (141) | surprise (142) | distraction (143) | anticipation |
| sadness | 5 | grief (151) | sadness (152) | pensiveness (153) | joy |
| disgust | 6 | loathing (161) | disgust (162) | boredom (163) | trust |
| anger | 7 | rage (171) | anger (172) | annoyance (173) | fear |
| anticipation | 8 | vigilance (181) | anticipation (182) | interest (183) | surprise |

8 dyads, derived on equal footing with the 24:
love (191, joy+trust), submission (192, trust+fear), awe (193,
fear+surprise), disapproval (194, surprise+sadness), remorse (195,
sadness+disgust), contempt (196, disgust+anger), aggressiveness (197,
anger+anticipation), optimism (198, anticipation+joy).

Plus `neutral` (label 0) as the reference corpus for the PC projection.
**33 corpora per model, 32 vectors per model.**

---

## 3. Configs

Generated, not hand-written, from a single `configs/wheel.yaml` spec by
`scripts/build_wheel_configs.py`.

They go in the **existing** `configs/emotion/` Hydra group with a
`wheel_` filename prefix — `configs/emotion/wheel_ecstasy.yaml`,
selected as `emotion=wheel_ecstasy`. A separate `configs/emotion_wheel/`
group would need Hydra defaults-list plumbing in every script for no
gain. The prefix resolves the ten filename collisions (nine cell names
plus `neutral`) without touching any existing config.

Inside the file, `name:` is the bare cell name (`ecstasy`), because
`cfg.emotion.name` is what the pipeline uses for corpus and activation
filenames — and after §1 those live under the wheel track, so bare names
are safe and keep the artefacts readable.

Each config carries `label`, `axis`, `ring`, `opposite`, and for dyads
`components`, so downstream code reads wheel structure from config
rather than a hard-coded table.

Note: the Plutchik dyad `contempt` (196) is a different construct from
the legacy Ekman `contempt` config (label 16) and inherits nothing from
it.

Tests: all 33 labels unique and disjoint from the existing 1-20; every
generated file is `wheel_`-prefixed; the 24 ring cells cover 8 axes x 3
rings exactly; every dyad's components are two distinct axes; no
generated config overwrites an existing path.

---

## 4. Run

Per model, per cell, the existing three scripts unchanged apart from §1:

1. `generate_emotion_stories.py` — 46 topics x 7 stories = **322 stories
   per cell**, 10,626 per model across 33 corpora. Emotion word and
   direct synonyms banned; topic list frozen and shared, so topic cannot
   separate cells.
2. `extract_story_activations.py` — residual stream at every candidate
   layer, mean-pooled over token positions >= 50.
3. `derive_story_steering_vectors.py` — vectors at every layer.

Then push to HF under the wheel namespace.

### Centering

`derive_story_steering_vectors.py` composes
`v_e = project_out(mean_e - grand_mean_over_emotions, neutral_PCs)`.
`project_out` is linear (`V - VBᵀB`), and the `.npz` files hold one
pooled vector per story per layer, so **every centering variant is a
cheap offline recomputation on the Mac.** Derive three and keep all:

| tag | vector | note |
|---|---|---|
| `xc` | `mean_e - grand_mean_over_emotions` | default; the paper construction |
| `nr` | `mean_e - mean_neutral` | norms interpretable against a fixed reference |
| `resid` | `xc` on text-residualized activations | per `plans/residualization-admiration.md` |

`xc` is the one to treat as canonical, since it is what the four-emotion
track used. Note that switching between `xc` and `nr` is close to a rigid
translation of the whole vector cloud: it changes norms and cosines but
leaves pairwise differences almost unchanged.

**These vectors are not the four-emotion vectors.** The wheel corpora are
regenerated (stochastic sampling at temperature 0.7), the neutral PCs are
refit on a new neutral corpus, and the grand mean is taken over 32
emotions rather than 4. `cos(v_wheel_joy, v_four_joy)` is worth computing
as a cheap indicator of how sensitive the construction is to its own
sampling.

### Footgun

`_discover_emotions` globs `*.npz` and excludes `neutral.npz`, so **the
emotion set — and therefore the grand mean — is whatever files sit in the
activation directory.** Removing one cell's `.npz` silently changes every
other vector. Rule: the wheel activation directory always holds exactly
33 files; a cell to exclude is excluded at analysis time by name, never
by deleting the file. `run_wheel.sh` asserts the count before calling
derive.

### Minimum sanity check

Not validation, just a check that the vectors are not garbage: run
`validate_logit_lens.py` at ~2/3 depth on each cell and read the top
tokens. It needs no authored stimuli and takes minutes. Cells whose lens
read-out is incoherent are worth knowing about before anything is built
on them.

---

## 5. Cost

| item | estimate |
|---|---|
| generation + extraction, per model | ~3-5 GPU-h |
| three primaries | ~9-15 GPU-h |
| at $0.34/h Community Cloud | **~$3-5**, ~$15 with 3x contingency |

Comfortably inside the $150 cap. Storage is the real constraint: 33
corpora vs the current 5 is **~6.6x** the activation volume per model —
pod disk >= 100 GB, and keep the free-the-HF-cache-between-models step
from `run_primaries.sh`.

**Caveat:** the per-cell story-pipeline wall-clock has never been
recorded (the runbook's cost table covers the old CAA prompt path).
Instrument the first cell of the first model and revise this table before
booking the full run.

---

## 6. Order

1. ~~§1 path namespacing + test~~ — **done 2026-08-28.**
2. ~~`configs/wheel.yaml` + `scripts/build_wheel_configs.py` + tests -> 33
   configs~~ — **done 2026-08-28** (commit 50dd487).
3. `scripts/run_wheel.sh` — the three scripts per cell, the 33-file
   assertion, HF push, free cache between models.
4. Smoke test: 2 cells on Qwen 2.5 0.5B on the Mac, `max_topics=3`.
5. Llama 3.1 8B full, then Qwen 2.5 7B, then Gemma 2 9B.
6. Three centering parameterizations offline; logit-lens sanity pass.
