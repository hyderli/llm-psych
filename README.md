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
 
**Three-phase plan:**
 
1. **Mac M5 (now, free).** Pipeline code, tests, configs, analysis on
   small test models (Qwen 2.5 0.5B, Llama 3.2 1B, Gemma 2 2B). PyTorch
   with MPS handles these comfortably. **Do not attempt 7-8B inference
   on the Mac** — too slow for behavioral n ≥ 200.
2. **Lightning AI Studios free tier (pilots, free).** ~15 GPU
   credits/month on T4 / L4-class GPUs with persistent storage.
   Used to validate the pipeline end-to-end on real 7-8B models
   before any paid run. Browser VS Code or SSH from the Mac.
3. **RunPod Community Cloud RTX 4090 (production, paid).** $0.34/hr,
   24 GB VRAM, runs Llama 3.1 8B in bf16 without quantization.
   Pay-per-second; pods can be preempted (acceptable for batch
   experimental runs that checkpoint between conditions). Estimated
   total project compute: $50–100. **Hard project budget cap: $150**
   — if exceeded, scope is reduced rather than spent through.
### Cloud workflow (Phase 2 / 3)
 
Cloud machines are ephemeral. Git is the source of truth; data and
results round-trip through GitHub or a HuggingFace Dataset / S3
bucket.
 
```bash
# On the cloud pod (after spin-up):
git clone git@github.com:hyderli/llm-psych.git
cd llm-psych
uv sync                                  # installs CUDA torch + bitsandbytes
echo "HF_TOKEN=$HF_TOKEN" > .env         # set via pod environment vars
 
# Run experiment
uv run python scripts/run_experiment.py exp=h2_blackmail \
    model=llama31_8b emotion=fear steering.control=target
 
# Push results back before destroying the pod
git add results/ figures/
git commit -m "H2 blackmail: llama31_8b × fear × target"
git push
```
 
**Result-handling rule.** Small parquet files (< 50 MB) commit
directly. Larger artifacts go to a HF Dataset or S3 bucket; the path
is recorded in `results/<exp>/manifest.json` so reruns are
reproducible.

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
