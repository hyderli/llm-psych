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
   small test models (Qwen 2.5 0.5B, Llama 3.2 1B). PyTorch with MPS
   handles these comfortably. **Do not attempt 7-8B inference on the
   Mac** — too slow for behavioral n ≥ 200.
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
uv run python scripts/run_experiment.py exp=h2_sycophancy \
    model=llama31_8b emotion=anger steering.control=target
 
# Push results back before destroying the pod
git add results/ figures/
git commit -m "H2 sycophancy: llama31_8b × anger × target"
git push
```
 
**Result-handling rule.** Small parquet files (< 50 MB) commit
directly. Larger artifacts go to a HF Dataset or S3 bucket; the path
is recorded in `results/<exp>/manifest.json` so reruns are
reproducible.
 

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

# Full sycophancy steering experiment (Linux server)
uv run python scripts/run_experiment.py -m exp=h2_sycophancy \
    model=llama31_8b emotion=joy,fear,anger,sadness \
    steering.control=target,zero,random,orthogonal
```

## License

MIT. See `LICENSE` (TODO: add).

## Citation

[TODO: add when publishing]
