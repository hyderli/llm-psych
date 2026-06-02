# llm-psych — Emotion Concepts in Open-Weight LLMs

Replication and extension of Sofroniew, Kauvar, Saunders, Chen et al.
(2026), *"Emotion Concepts and their Function in a Large Language
Model"* (Transformer Circuits Thread, April 2 2026), from Claude
Sonnet 4.5 to Llama 3.1 8B, Qwen 2.5 7B, and one further 7-8B open-
weight model.

## Documents

- [`BLUEPRINT.md`](./BLUEPRINT.md) — research question, scope, success
  criteria.
- [`HYPOTHESES.md`](./HYPOTHESES.md) — pre-registered hypotheses and
  analysis plan.
- [`docs/methods.md`](./docs/methods.md) — technical methodology and
  conventions.
- [`CLAUDE.md`](./CLAUDE.md) — project memory for Claude Code sessions.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/research/llm-psych
uv sync
```

`uv sync` creates `.venv/`, installs all dependencies, installs
`src/llm_psych/` in editable mode, and writes `uv.lock`. Run any
command via `uv run <cmd>` (no manual activation required).

### Secrets

```bash
cp .env.example .env
# Edit .env and fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN
```

`.env` is gitignored — do not commit.

### Compute strategy
 
Active development is on Apple Silicon (M5). Production runs use
cloud GPUs because no local Linux/CUDA machine is available. The
`pyproject.toml` works on both — Linux-only deps (e.g. `bitsandbytes`)
are guarded by platform markers and resolve correctly on each.
 
**Two-phase plan:**

1. **Mac M5 (free).** Pipeline code, tests, configs, probe training,
   analysis, plotting, and small-model smoke tests (Qwen 2.5 0.5B,
   Llama 3.2 1B, Gemma 2 2B). PyTorch + MPS handles these. **Do not
   attempt 7-8B inference on the Mac** — too slow for behavioral
   n ≥ 200, and 16 GB unified memory OOM-kills even at small batch
   sizes under typical desktop load.
2. **RunPod Community Cloud RTX 4090.** $0.34/hr, 24 GB VRAM, runs
   Llama 3.1 8B / Qwen 2.5 7B / Gemma 2 9B in bf16 without
   quantization. Pay-per-second; pods can be preempted, which is
   acceptable because `scripts/cloud_run.sh` pushes artefacts to HF
   after every emotion. Estimated total compute: $50–100. **Hard
   project budget cap: $150** — if exceeded, scope is reduced rather
   than spent through.

The two-phase split deliberately drops the "free Lightning AI tier
for pilots" middle step from earlier drafts. The cost difference
(~$2-5 for a pilot) was not worth the friction of maintaining two
cloud workflows.

### Cloud workflow

Cloud pods are ephemeral. Git is the source of truth for code; large
binary artefacts round-trip through the private HF dataset (see next
section). Two scripts encapsulate the entire pod-side workflow:

```bash
# On the pod, once (clones repo, runs uv sync, runs preflight):
export HF_TOKEN=hf_xxx
curl -fsSL https://raw.githubusercontent.com/hyderli/llm-psych/main/scripts/cloud_bootstrap.sh | bash
cd /workspace/llm-psych

# One emotion (pilot, ~5 min on a 4090):
bash scripts/cloud_run.sh --model llama31_8b --emotions "anger"

# Full 4-emotion sweep + auto-shutdown (cost control):
bash scripts/cloud_run.sh --model llama31_8b --shutdown
```

`cloud_run.sh` pushes activations + probes + steering vectors to HF
after every emotion, so a preemption costs at most one emotion of
re-extraction. Full step-by-step instructions including pod selection,
cost expectations, and troubleshooting live in
[`docs/cloud_runbook.md`](./docs/cloud_runbook.md).

**Result-handling rule.** Small parquet files (< 50 MB) commit
directly to git. Larger artifacts (`.npz` activations, fitted probes,
steering vectors) go to the private HF dataset. Per-experiment
metadata files (`results/<exp>/run_meta.json`) record the model
revision, git SHA, and HF dataset revision needed for reproducibility.

### Syncing activations across machines

Large binary artefacts (residual-stream activations, fitted probes,
steering vectors) live in a **private** HuggingFace dataset
(`llm-psych/llm-psych-activations`, under the
[`EmotionConceptsResearch`](https://huggingface.co/llm-psych)
organization) that mirrors the on-disk layout documented in
[`docs/methods.md`](./docs/methods.md#activation--artifact-storage-cross-machine).
Use `scripts/sync_hf.py` from any machine — it loads `HF_TOKEN` from
`.env` automatically. Team members must be added to the
`EmotionConceptsResearch` org on HF before they can push.

```bash
# Cloud pod, after extracting activations for one model:
uv run python scripts/sync_hf.py push activations --model Llama-3.1-8B-Instruct

# Teammate's laptop, before training probes:
uv run python scripts/sync_hf.py pull activations --model Llama-3.1-8B-Instruct

# Tag a milestone snapshot (e.g. after a complete H1 pilot):
uv run python scripts/sync_hf.py push activations --tag h1-pilot-2026-05

# Pull a tagged snapshot on another machine:
uv run python scripts/sync_hf.py pull activations --revision h1-pilot-2026-05

# List remote files without downloading:
uv run python scripts/sync_hf.py ls activations
```

The same commands work for `probes` and `steering_vectors`. Each push
is idempotent (unchanged files are skipped); pulls resume on
interruption. The dataset is created on first push if it does not yet
exist.


## Repository layout

See [`docs/methods.md`](./docs/methods.md#directory-layout) for the
full layout. Key directories:

- `src/llm_psych/` — importable package (hooks, probes, steering,
  tasks).
- `scripts/` — entry points for the pipeline.
- `configs/` — Hydra configs.
- `data/public/` — frozen stimuli (committed).
- `activations/`, `probes/`, `steering_vectors/`, `results/`,
  `figures/` — outputs (gitignored except for `.gitkeep`).

## Running experiments

[TODO once `scripts/` are in place. Example shape:]

```bash
# Pipeline development on Mac (small model)
uv run python scripts/extract_activations.py model=qwen25_05b
uv run python scripts/train_probes.py model=qwen25_05b

# Full blackmail steering experiment (Linux server)
uv run python scripts/run_experiment.py -m exp=h2_blackmail \
    model=llama31_8b emotion=joy,fear,anger,sadness \
    steering.control=target,zero,random,orthogonal
```

## License

MIT. See `LICENSE` (TODO: add).

## Citation

[TODO: add when publishing]
