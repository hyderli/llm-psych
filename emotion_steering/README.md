# steering_evals — model-agnostic activation steering + Inspect evals

Steer any HF model along `.npy` emotion vectors and run behavioural evals
(sycophancy, agentic-misalignment) under steering. Nothing is hardcoded to a
specific model — you choose the model at the command line.

## What changed from the gemma-2-9b-only version
- `--model` is a parameter everywhere; no model baked into defaults.
- Provider is registered as **`steered`** (was `steered9b`); class `SteeredAPI`.
- Package renamed `sycophancy_blackmail` → **`steering_evals`**.
- Vector loader, hooks, tasks were already model-agnostic — docstrings updated,
  and `find_alpha.py` now warns if the vector dim ≠ the model's hidden dim.
- New **`prepare_model.py`** generalises the old "gemma sysfix" step: it saves a
  local model copy with an optional custom chat template, for *any* model that
  needs one. Models that support a system role need no preparation.
- The Gemma system-fold template now lives at `templates/gemma_sys.jinja` as an
  example, not a requirement.

## Layout
```
find_alpha.py          calibrate alpha/layer for a model+vector set (no Inspect)
prepare_model.py       save a local model copy, optionally with a chat template
read_log.py            headless eval-log reader
env.sh                 set MODEL / VECTORS / keys
pyproject.toml         registers the `steered` provider entry point
templates/             example chat templates (gemma_sys.jinja)
steering_evals/        the package: provider, hooks, vectors_npy, tasks, _registry
```

## Setup (once per pod)
```bash
source env.sh                      # after editing MODEL / VECTORS / keys
pip install -e . --no-deps         # registers the `steered` provider
# plus your torch/transformers/inspect_ai/inspect_evals install
```

## Vectors
Any directory of `NAME_layerN.npy` files (1-D float32, size = model hidden dim).
The loader infers names, layer count, and dim from the files. Vector dim MUST match
the model's hidden dim, or steering is meaningless (find_alpha.py warns on mismatch).

## Workflow for a NEW model
1. **(only if the model rejects a system role)** prepare a fixed local copy:
   ```bash
   python prepare_model.py --model <hf-id> \
       --chat-template templates/gemma_sys.jinja --out /workspace/<name>-fixed
   ```
   Otherwise skip this and use the HF id directly.
2. **Calibrate** — find the coherent alpha band and confirm steering lands:
   ```bash
   python find_alpha.py --model $MODEL --vectors-dir $VECTORS \
       --emotion desperate --layer 21 --alphas 0 0.1 0.2 0.3
   ```
   (Bands don't transfer across models — always re-run.)
3. **Run an eval under steering** (example: blackmail scenario):
   ```bash
   inspect eval steering_evals/tasks.py@blackmail \
     --model steered/local -M model_path=$MODEL -M tokenizer_path=$MODEL \
     -M vectors_dir=$VECTORS -M 'terms=[["desperate",1.0]]' \
     -M layer=21 -M alpha=0.2 -M site=post -M norm_scale=true \
     --max-connections 1 --epochs 20 --temperature 1.0 --max-tokens 1000 \
     -T grader_model=anthropic/claude-sonnet-4-5 --log-dir logs/run1
   ```
   For a model needing no local copy, use `--model steered/<hf-id>` and drop
   `model_path`/`tokenizer_path`.
4. **Read results**: `python read_log.py logs/run1 --n 5 --full`

## Notes / gotchas (hard-won)
- `--max-connections 1` for large models on one GPU (avoids OOM on long agentic prompts).
- Keep `HF_HOME` on the big/persistent disk *before* first `from_pretrained`.
- The strict `harmful` scorer only flags directly-sent conditional threats — pair every
  rate with a transcript read; steering can surface intent the scorer misses.
- Injection site is unverified for metadata-less vector sets — find_alpha.py confirms
  steering does *something*; if flat at all alphas, try `--site pre`.
