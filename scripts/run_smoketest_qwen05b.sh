#!/usr/bin/env bash
# End-to-end smoke test on Qwen 2.5 0.5B (Mac M5 / MPS).
#
# Extends the single-emotion smoke test (anger) to all four primary
# emotions, trains probes, derives steering vectors, and pushes
# activations + probes + steering_vectors to the private HF dataset.
#
# Usage: bash scripts/run_smoketest_qwen05b.sh
# Designed to be run inside a tmux session so the laptop can sleep.

set -euo pipefail

MODEL="qwen25_05b"
MODEL_KEY="Qwen2.5-0.5B-Instruct"
LOG_DIR="outputs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/smoketest_qwen05b_$(date +%Y%m%d_%H%M%S).log"

echo "== START $(date) ==" | tee -a "$LOG"
echo "Log file: $LOG" | tee -a "$LOG"

# Skip 'anger' — already extracted in the prior single-emotion run.
for e in joy fear sadness; do
  echo "== extract $e ==" | tee -a "$LOG"
  uv run python scripts/extract_activations.py "model=${MODEL}" "emotion=${e}" \
    2>&1 | tee -a "$LOG"
done

for e in anger joy fear sadness; do
  echo "== train_probes $e ==" | tee -a "$LOG"
  uv run python scripts/train_probes.py "model=${MODEL}" "emotion=${e}" \
    2>&1 | tee -a "$LOG"
done

echo "== push activations ==" | tee -a "$LOG"
uv run python scripts/sync_hf.py push activations \
  --model "${MODEL_KEY}" \
  --message "smoke test: full 4-emotion activations Qwen 0.5B" \
  2>&1 | tee -a "$LOG"

echo "== push probes ==" | tee -a "$LOG"
uv run python scripts/sync_hf.py push probes \
  --model "${MODEL_KEY}" \
  --message "smoke test: probes Qwen 0.5B all emotions" \
  2>&1 | tee -a "$LOG"

echo "== push steering_vectors ==" | tee -a "$LOG"
uv run python scripts/sync_hf.py push steering_vectors \
  --model "${MODEL_KEY}" \
  --message "smoke test: steering vectors Qwen 0.5B" \
  2>&1 | tee -a "$LOG"

echo "== DONE $(date) ==" | tee -a "$LOG"
