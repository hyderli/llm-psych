#!/usr/bin/env bash
# Story-method pipeline smoke test on Qwen 2.5 0.5B (Mac / MPS).
#
# Proves the PAPER-METHOD (story) derivation runs end-to-end before any
# primary run:  generate -> extract_story -> derive_story.  This pipeline
# has never been GPU/MPS smoke-tested (see plans/derivation-primacy-
# decision.md and RESEARCH_LOG).  It is intentionally tiny (5 stories per
# emotion, short token budget) — the goal is "does it run and produce
# correctly-shaped artifacts?", NOT a usable result.
#
# Usage:  bash scripts/run_smoketest_story_qwen05b.sh
# Run inside tmux so the laptop can sleep.  Local only — no HF push; push
# happens only after this passes and a real run is done.
#
# Override the tiny budget via env vars, e.g.  N_STORIES=8 bash ...

set -euo pipefail

# Let any MPS-unsupported op fall back to CPU instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODEL="qwen25_05b"
MODEL_KEY="Qwen2.5-0.5B-Instruct"

# --- tiny smoke budget (override via env) ---
# Topic-matched: stories_per_topic x max_topics stories per emotion.
# Defaults give 1 x 5 = 5 stories per emotion (same size as before).
STORIES_PER_TOPIC="${STORIES_PER_TOPIC:-1}"
MAX_TOPICS="${MAX_TOPICS:-5}"        # cap topics for a fast smoke run
MIN_TOK="${MIN_TOK:-10}"             # drop stories shorter than this
MAX_NEW="${MAX_NEW:-80}"             # generation length cap
# CRITICAL: pooling starts at pool_start_token (default 50).  Tiny smoke
# stories are far shorter than 50 tokens, so without lowering this EVERY
# story is dropped at extraction and you get empty .npz files.  Keep it
# below MIN_TOK so each surviving story contributes >=1 pooled position.
POOL_START="${POOL_START:-5}"

# Device. Default CPU + float32: proven reliable for this 0.5B smoke test
# (~1.5s/generation), and avoids MPS generate() warmup/flakiness. For the
# faster Apple-GPU path once you trust it:
#     DEVICE_MAP=mps DTYPE=float16 bash scripts/run_smoketest_story_qwen05b.sh
DEVICE_MAP="${DEVICE_MAP:-cpu}"
DTYPE="${DTYPE:-float32}"

EMOS=(admiration joy loathing sadness neutral)

LOG_DIR="outputs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/smoketest_story_qwen05b_$(date +%Y%m%d_%H%M%S).log"

OVERRIDES=(
  "model=${MODEL}"
  "model.device_map=${DEVICE_MAP}"
  "model.torch_dtype=${DTYPE}"
  "derivation=story"
  "derivation.stories_per_topic=${STORIES_PER_TOPIC}"
  "derivation.max_topics=${MAX_TOPICS}"
  "derivation.generator.min_story_tokens=${MIN_TOK}"
  "derivation.generator.max_new_tokens=${MAX_NEW}"
  "derivation.pool_start_token=${POOL_START}"
)

echo "== START $(date) ==" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Budget: stories_per_topic=${STORIES_PER_TOPIC} max_topics=${MAX_TOPICS} min_tok=${MIN_TOK} max_new=${MAX_NEW} pool_start=${POOL_START}" | tee -a "$LOG"

# --- Step 1: generate stories (per emotion + neutral) ---
for e in "${EMOS[@]}"; do
  echo "== generate $e ==" | tee -a "$LOG"
  uv run python scripts/generate_emotion_stories.py \
    "${OVERRIDES[@]}" "emotion=${e}" \
    2>&1 | tee -a "$LOG"
done

# --- Step 2: extract pooled story activations (per emotion + neutral) ---
for e in "${EMOS[@]}"; do
  echo "== extract_story $e ==" | tee -a "$LOG"
  uv run python scripts/extract_story_activations.py \
    "${OVERRIDES[@]}" "emotion=${e}" \
    2>&1 | tee -a "$LOG"
done

# --- Step 3: derive story steering vectors (once; discovers emotions) ---
echo "== derive_story_steering_vectors ==" | tee -a "$LOG"
uv run python scripts/derive_story_steering_vectors.py \
  "${OVERRIDES[@]}" \
  2>&1 | tee -a "$LOG"

# --- Step 4: H1 probes on story activations (per emotion vs neutral) ---
# neutral is the negative class, not a probe target.
for e in admiration joy loathing sadness; do
  echo "== train_probes (story) $e ==" | tee -a "$LOG"
  uv run python scripts/train_probes.py \
    "${OVERRIDES[@]}" "emotion=${e}" \
    2>&1 | tee -a "$LOG"
done

# --- Verification: list and count the artifacts produced ---
echo "== VERIFY artifacts ==" | tee -a "$LOG"
{
  echo "-- stories (data/derived/stories/${MODEL_KEY}/) --"
  ls -la "data/derived/stories/${MODEL_KEY}/" 2>&1 || echo "MISSING"
  echo "-- activations (activations/${MODEL_KEY}-story/) --"
  ls -la "activations/${MODEL_KEY}-story/" 2>&1 || echo "MISSING"
  echo "-- steering vectors (steering_vectors/${MODEL_KEY}-story/) --"
  ls -la "steering_vectors/${MODEL_KEY}-story/" 2>&1 || echo "MISSING"
  echo "-- H1 probes (probes/${MODEL_KEY}-story/) --"
  ls "probes/${MODEL_KEY}-story/"*_summary.csv 2>&1 || echo "MISSING"
  echo "-- npz row counts (should be >0 per emotion) --"
  uv run python - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob("activations/Qwen2.5-0.5B-Instruct-story/*.npz")):
    z = np.load(f)
    layers = [k for k in z.files if k.startswith("layer_")]
    n = z[layers[0]].shape[0] if layers else 0
    print(f"  {os.path.basename(f)}: {len(layers)} layers, {n} stories, dim={z[layers[0]].shape[1] if layers else '?'}")
PY
} 2>&1 | tee -a "$LOG"

echo "== DONE $(date) ==" | tee -a "$LOG"
echo "PASS if: every .npz has >0 stories, steering_vectors/ has *_layer*.npy + manifest.yaml, and probes/${MODEL_KEY}-story/ has <emotion>_summary.csv. Full generate->extract->derive->H1-probe path runs. Next: real dev-fleet run + confound audit on story activations." | tee -a "$LOG"
