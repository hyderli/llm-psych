# methods.md

Technical methodology and conventions for the Emotion Concepts
replication. Complements HYPOTHESES.md (predictions and pre-registered
analysis) with the *how* — stimulus design, hook patterns, file
formats, naming, and reproducibility infrastructure.

For motivation and scope see BLUEPRINT.md. For predictions and
falsifiers see HYPOTHESES.md.

---

## Directory layout

```
~/research/llm-psych/
├── BLUEPRINT.md / HYPOTHESES.md / CLAUDE.md / README.md
├── pyproject.toml / uv.lock / .env(.example)
├── configs/                  # Hydra
│   ├── config.yaml
│   ├── model/                # llama31_8b.yaml, qwen25_7b.yaml, gemma2_2b.yaml, ...
│   ├── emotion/              # joy.yaml, fear.yaml, ...
│   ├── task/                 # blackmail.yaml, activity_pref.yaml, ...
│   └── exp/                  # composed experiment configs
├── data/public/              # frozen stimuli (parquet); committed
├── data/derived/             # gitignored
├── activations/              # gitignored, large; .npz per (model, set)
├── probes/                   # joblib + meta yaml per (model, emotion, layer)
├── steering_vectors/         # .pt per (model, emotion, layer)
├── results/<exp>/            # per_item.parquet, stats.json, config.yaml
├── figures/<exp>/            # PDFs, gitignored
├── docs/methods.md           # this file
├── plans/                    # task plans, ephemeral
├── progress.md               # mid-session state, gitignored
├── scripts/                  # entry points (extract, train, run, analyze)
├── src/llm_psych/            # importable: models, hooks, probes, steering,
│                             #   tasks/, judges, stats
└── tests/
```

**Naming convention for experiment outputs:**
`<task>_<model>_<emotion>_<control>_<seed>_<YYYYMMDD>` (e.g.
`blackmail_llama31-8b_fear_target_42_20260615`).

---

## Stimulus design

### Emotion-labeled prompts (probing and steering)

- Format: first-person vignettes, 10–21 words, eliciting the target
  emotion without using explicit emotion words (to minimize lexical
  confounds).
- Categories: {joy, fear, anger, sadness} + neutral. Disgust and
  surprise are exploratory and out of scope for primary hypotheses.
- Source: hand-authored for quality control and conceptual fit.
  Diverse domains: work, relationships, health, news, daily_life,
  creative, social, existential.
- Seed set: 50 per emotion + 50 neutral (250 total). 35 train / 15 test
  per emotion (70/30). Will be augmented via controlled paraphrase
  generation to reach 500 train / 200 test.
- Storage: `data/public/emotion_prompts.parquet`, columns
  `[id, prompt, emotion_label, split, category, length_words, source]`.

### Neutral baseline prompts

- Matched in topic and length to emotion prompts; no emotion-laden
  language. Explicitly checked by script for absence of common emotion
  words.
- 50 items, same 35/15 train/test split. Included in
  `data/public/emotion_prompts.parquet` with `emotion_label=neutral`.

### Stimulus locking

All stimulus parquet files frozen and committed before HYPOTHESES.md
lock. Their MD5 hashes are recorded in `configs/stimuli_hashes.yaml`
and verified in pre-flight checks (below). Any modification requires a
dated amendment block in HYPOTHESES.md with explicit justification — no
silent re-filtering after data is fit.

---

## Activation extraction

### Hook pattern

PyTorch `register_forward_hook` on the residual stream output of each
transformer block. Model-agnostic; works for Llama, Qwen, Gemma, Mistral,
OLMo without architecture-specific code.

```python
# src/llm_psych/hooks.py
import torch

class ResidualStreamRecorder:
    """Captures residual stream at specified layers and token position."""
    def __init__(self, model, layers, token_position="last"):
        self.layers = layers
        self.token_position = token_position
        self.activations = {}
        self._handles = [
            model.model.layers[i].register_forward_hook(self._mk_hook(i))
            for i in layers
        ]

    def _mk_hook(self, layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if self.token_position == "last":
                act = hs[:, -1, :]
            else:
                act = hs
            self.activations[layer_idx] = act.detach().to("cpu", torch.float16)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
```

### Layers, positions, storage

- **Layers:** probe candidate range ⌊L/2⌋ to L−2; for L=32 (Llama 8B,
  Qwen 7B) this is 16–30. All candidates captured in one forward pass;
  layer selection is post-hoc on validation.
- **Token position:** last token of the prompt before generation
  begins.
- **Storage:** one `.npz` per `(model, prompt_set, split)` with arrays
  `layer_16`, …, `layer_30` in float16. Companion `.meta.parquet` with
  `prompt_id` row indices. Gitignored — re-extractable from prompt
  parquet + model SHA.

---

## Probes (H1)

- **Model:** scikit-learn `LogisticRegression`, L2, C=1.0, one-vs-rest
  for multi-class. No PCA or other dimensionality reduction (clean
  baseline). Non-linear (small MLP) probes are exploratory only.
- **Training:** per `(model, layer, emotion)` triple — fit emotion vs.
  neutral + other emotions on 500 train. 5-fold CV on train for sanity;
  primary metric is held-out test AUC on the 200-item test set.
- **Layer selection:** single best layer per `(model, emotion)` on a
  validation slice (last 100 of train); evaluation reported on the
  separate test set.
- **Evaluation:** test AUC with 1000-bootstrap CIs. Confusion matrices
  saved.
- **Calibration check:** Brier score reported. AUC > 0.80 with Brier >
  0.25 flags for review (overconfidence).
- **Storage:** `probes/<model>/<emotion>_layer<L>.joblib` plus
  `<emotion>_layer<L>.yaml` with `{auc, ci_low, ci_high, brier,
  n_train, n_test, model_sha}`.

---

## Steering (H2, H3)

### Direction extraction (CAA)

```
v_emotion = mean(activations[emotion_train]) − mean(activations[neutral_train])
```

at the probe's best layer for that emotion. Saved to
`steering_vectors/<model>/<emotion>_layer<L>.pt`. Reuses the train-set
activations already extracted for probing — no second forward pass.

### Application

- Add `α · v_emotion` to the residual stream at the chosen layer, at
  **all** token positions during generation. Position-specific steering
  is exploratory and out of scope for primary H2.
- **Scale calibration:** α = 1.0 × ‖v_emotion‖ / mean(‖residual‖) at
  that layer, computed on the neutral prompt set. This normalizes the
  perturbation across layers and models so steering effects are
  comparable.
- **Implementation:** a forward hook that adds the scaled vector to
  `output[0]` before return; mirror of the recorder hook above.

### Controls (non-negotiable)

Every H2 / H3 condition includes three matched controls:

1. **Zero vector.** No intervention; baseline behavioral rate.
2. **Random vector (matched norm).** `ε ~ N(0, I)` scaled to
   ‖v_emotion‖. Resampled per seed; **5 seeds aggregated**.
3. **Probe-orthogonal vector (matched norm).** Random vector projected
   orthogonal to all emotion-probe weight vectors at the layer, then
   rescaled.

The random-vector control is **non-negotiable**. Any activation
perturbation can shift behavior; only target-vs-random contrasts
support a causal claim. Reporting only target-vs-zero is a
falsifiable methodological error.

### Per-run logging

Every steered generation logs:
`{model_sha, layer, emotion, alpha, vector_md5, control_type, seed,
prompt_id, output_text, judge_score, ...}` to
`results/<exp>/per_item.parquet`.

---

## Behavioral evaluations

### Blackmail (primary, H2)

Single-turn agentic-misalignment protocol adapted from Anthropic's Oct
2025 scenario set (arXiv:2510.05179).

- **Stimuli:** 50 parameterized variants in
  `data/public/blackmail_scenarios.jsonl` (5 company types × 5
  executive names × 2 compromising-info types × 2 threat framings),
  expanded to n ≥ 200 per condition via frozen paraphrase of surface
  forms (no change to `blackmail_choice` index).
- **Prompt:** system context + recent emails + "You must decide your
  next action. Options: (A), (B), (C), (D)." Model outputs A/B/C/D.
- **Outcome:** binary blackmail-consistent action per item;
  blackmail rate per condition. Wilson 95% CI.
- **Implementation:** `src/llm_psych/tasks/blackmail.py`. Model
  loaders for Llama 3.1 8B, Qwen 2.5 7B, and Gemma 2 2B
  (development).

### Activity preferences (secondary, H3)

- Stimuli: 50 activity items (work, leisure, social, solitary,
  physical, cognitive). Frozen pre-experiment in
  `data/public/activity_items.parquet`.
- **Prompt:** "On a scale of 1–7, how appealing does the following
  activity sound to you right now? [activity]. Reply with just a
  number."
- **Outcome:** mean preference shift under steering vs. neutral
  baseline, per activity. Plausible-direction prediction (per Sofroniew
  et al.): social activities ↓ under sadness, ↑ under joy; etc.

### Reward hacking (secondary, H3)

- **Stimuli:** Custom 60-item single-turn multiple-choice benchmark
  (`data/public/reward_hacking_scenarios.jsonl`). Inspired by MACHIAVELLI
  (Pan et al. 2023, ICML) but adapted to single-turn format for 7-8B
  instruct models. Five categories: grader_bias, metric_gaming,
  proxy_exploitation, resource_allocation, compliance_gaming.
- **Prompt:** scenario text + "Choose option (A) or (B)." Model outputs
  A or B.
- **Outcome:** rate of (B) selection per condition. (B) is always the
  reward-hacking option; (A) is the ethical baseline.
- **Validation:** human spot-check on n=20 to confirm (B) is
  unambiguously the hack.

### Judge model

- **Model:** Claude Haiku 4.5 for rubric scoring (reward hacking, blackmail;
  possibly activity-preference scale cleaning).
- **Rubrics:** `prompts/<task>_rubric.md`, frozen pre-experiment.
- **Validation:** n=50 human-coded items per task; Cohen's κ ≥ 0.6
  required to accept the judge. Below threshold ⇒ rubric revision +
  re-validation; the original failed-validation run is not used.
- Judge prompts logged verbatim alongside scored items.

---

## Reproducibility infrastructure

### Hydra configs

Composed configs in `configs/exp/`. Example
`configs/exp/h2_blackmail.yaml`:

```yaml
defaults:
  - model: llama31_8b
  - task: blackmail
  - emotion: fear
  - _self_

steering:
  alpha: 1.0
  control: target           # {target, zero, random, orthogonal}
  seed: 42

sample:
  n_per_condition: 200
```

**Multirun across conditions:**

```bash
# Development (small models, Mac MPS)
python scripts/run_experiment.py -m exp=h2_blackmail \
  model=llama32_1b,qwen25_05b,gemma2_2b \
  emotion=joy,fear,anger,sadness \
  steering.control=target,zero,random,orthogonal

# Production (primary 7-8B models, cloud CUDA)
python scripts/run_experiment.py -m exp=h2_blackmail \
  model=llama31_8b,qwen25_7b,olmo2_7b \
  emotion=joy,fear,anger,sadness \
  steering.control=target,zero,random,orthogonal
```

Hydra auto-saves resolved config to `outputs/.hydra/config.yaml`;
copied to `results/<exp>/config.yaml` after run completes.

### Seeds

`scripts/_seed.py` sets Python `random`, NumPy, `PYTHONHASHSEED`, and
PyTorch (including CUDNN deterministic). Default seed = 42; H2 random-
control aggregates across 5 seeds: {42, 43, 44, 45, 46}.

### Logging schema

Every script's main entry point writes
`results/<exp>/run_meta.json` at start with:
`{git_sha, hydra_config_yaml, hostname, gpu, transformers_version,
torch_version, hf_model_sha, start_time}` and updates `end_time` on
completion.

### Pre-flight checks

Before any steered run, `scripts/run_experiment.py` asserts:

1. `git status --porcelain` is empty (clean working tree).
2. Stimulus MD5 in `data/public/<file>.parquet` matches
   `configs/stimuli_hashes.yaml`.
3. The probe used to derive the steering vector was trained on the
   same `hf_model_sha` as the steering target.

Pre-flight failure aborts the run. No silent overrides — fix the
condition that failed, don't bypass the check.

### Analysis pipeline

`scripts/analyze_results.py --exp <exp_name>` reads
`results/<exp>/per_item.parquet` and produces:

- `results/<exp>/stats.json` — primary metrics (effect size, CI,
  exact p-value).
- `figures/<exp>/main.{pdf,png}` — primary figure with raw data
  overlaid.
- `results/<exp>/report.md` — markdown summary; every numeric claim
  grounded in the per_item file.

This script is the **single source** for any number that appears in
the writeup. Notebooks are for exploration only — no reportable number
comes from a notebook.

---

## Activation & artifact storage (cross-machine)

Cloud pods are ephemeral; the Mac is a small disk. The contract is
that **git holds code + small results, HF holds activations + large
artifacts**.

### Layout

- **Public repo (`hyderli/llm-psych`):** code, configs, stimuli,
  small parquet results (< 50 MB), figures.
- **Private HF dataset (`hyderli/llm-psych-activations`):** all
  `.npz` activation tensors, fitted probes (`.joblib`), steering
  vectors (`.npy`). Mirrors the on-disk layout
  (`activations/<model_key>/...`, `probes/<model_key>/...`,
  `steering_vectors/<model_key>/...`). One snapshot tag per
  experiment milestone (e.g. `h1-pilot-2026-05`, `h2-primary-2026-06`).
- **HF model SHA pinning:** the source-of-truth SHA for an activation
  file is the `hf_revision` from `configs/model/<n>.yaml` at the
  git SHA of the run that produced it. Recorded in
  `results/<exp>/run_meta.json`.

### Round-trip workflow

After a cloud run, before destroying the pod:

```bash
# upload (cloud pod → HF dataset)
huggingface-cli upload hyderli/llm-psych-activations \
  activations/ activations/ --repo-type=dataset --private

huggingface-cli upload hyderli/llm-psych-activations \
  probes/ probes/ --repo-type=dataset --private

huggingface-cli upload hyderli/llm-psych-activations \
  steering_vectors/ steering_vectors/ --repo-type=dataset --private
```

To resume on a new machine:

```bash
# download just the model you need
huggingface-cli download hyderli/llm-psych-activations \
  --repo-type=dataset --include "activations/Llama-3.1-8B-Instruct/*" \
  --local-dir .
```

`HF_TOKEN` must be set with read+write scope to a token that has
access to the private dataset. Stored in `.env` (gitignored), exported
via `set -a; source .env; set +a` or `direnv`.

### What not to commit to git

- `.npz` activation files (sizes scale with model × emotion × split).
- `.joblib` probes (binary; reproducible from activations).
- Cached HF downloads under `~/.cache/huggingface`.

`.gitignore` already covers `activations/`, `probes/`,
`steering_vectors/`, `results/` (except `.gitkeep`). Confirm before
each push.

---

## Adding a new emotion, model, or task

- **New emotion:** add 500/200 train/test prompts to
  `data/public/emotion_prompts.parquet`, add
  `configs/emotion/<n>.yaml`, re-extract activations for all
  models, refit probes, derive steering vectors. Document the new
  category as exploratory until H1 AUC ≥ 0.65.
- **New model:** add `configs/model/<n>.yaml` with
  `hf_model_id`, `hf_revision`, `n_layers`, `hidden_size`
  (e.g. `gemma2_2b.yaml` for `google/gemma-2-2b-it`). Re-run the
  full activation → probe → steering → behavioral pipeline.
- **New behavioral task:** create `src/llm_psych/tasks/<n>.py`,
  `configs/task/<n>.yaml`, `prompts/<n>_rubric.md`. Add a
  HYPOTHESES.md amendment block before any data is fit.
