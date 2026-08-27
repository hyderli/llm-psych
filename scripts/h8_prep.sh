#!/usr/bin/env bash
# One-shot H8 prep run on a CPU pod.
#
# Re-decomposes the story-method emotion vectors with k=64, computes
# digit-span projections, and correlates J-space fractions with the C2
# validation sweep metrics. Pushes all outputs to the private HF dataset.
#
# Run inside tmux on a 32 GB RAM CPU pod:
#
#     export HF_TOKEN=hf_...
#     bash scripts/h8_prep.sh

set -euo pipefail

REPO_ID="llm-psych/llm-psych-activations"
DECOMP_DIR="results/workspace_decomposition_k64"
LOG_DIR="outputs"
PYTHON=".venv-cpu/bin/python"

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/h8_prep_${TS}.log"

log()  { printf '\033[1;34m[h8-prep]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

# Verify HF_TOKEN is available for gated models and dataset writes.
if [[ -z "${HF_TOKEN:-}" ]]; then
    printf 'ERROR: HF_TOKEN is not set\n' >&2
    exit 1
fi

log "Python: $PYTHON"
log "log: $LOG"

pull_validation() {
    # Download C2 sweep tables needed for the correlation readout.
    $PYTHON - <<'PY' | tee -a "$LOG"
from huggingface_hub import snapshot_download

repo_id = "llm-psych/llm-psych-activations"
for pattern in ["results/vector_validation/*", "vector_validation/*"]:
    try:
        snapshot_download(repo_id, repo_type="dataset", allow_patterns=pattern, local_dir=".")
        print(f"pulled {pattern}")
        break
    except Exception as exc:
        print(f"{pattern} failed: {exc}")
PY
}

pull_steering_vectors() {
    local model_key="$1"
    $PYTHON - "$model_key" <<'PY' | tee -a "$LOG"
import sys
from huggingface_hub import snapshot_download

model_key = sys.argv[1]
snapshot_download(
    "llm-psych/llm-psych-activations",
    repo_type="dataset",
    allow_patterns=f"steering_vectors/{model_key}-story/*",
    local_dir=".",
)
print(f"pulled steering_vectors/{model_key}-story")
PY
}

push_results() {
    local folder="$1"
    local path_in_repo="$2"
    $PYTHON - "$folder" "$path_in_repo" <<'PY' | tee -a "$LOG"
import sys
from pathlib import Path
from huggingface_hub import HfApi

folder = Path(sys.argv[1])
path_in_repo = sys.argv[2]
if not folder.is_dir():
    raise SystemExit(f"missing folder: {folder}")
files = [p for p in folder.rglob("*") if p.is_file()]
if not files:
    raise SystemExit(f"no files to push in {folder}")
HfApi().upload_folder(
    repo_id="llm-psych/llm-psych-activations",
    repo_type="dataset",
    folder_path=str(folder),
    path_in_repo=path_in_repo,
    commit_message=f"h8_prep: {path_in_repo}",
)
print(f"pushed {folder} -> llm-psych/llm-psych-activations/{path_in_repo}")
PY
}

run_model() {
    local model="$1"
    local model_key
    model_key=$(awk '/^hf_model_id:/{print $2}' "configs/model/${model}.yaml")
    model_key="${model_key##*/}"

    section "pull steering vectors (${model_key})"
    pull_steering_vectors "$model_key"

    section "decompose k=64 + digit projection (${model_key})"
    $PYTHON scripts/decompose_emotion_vectors.py \
        --model-config "configs/model/${model}.yaml" \
        --lens-source neuronpedia \
        --emotions admiration joy loathing sadness \
        --k 64 --n-candidates 512 \
        --digit-projection \
        --output-dir "$DECOMP_DIR" 2>&1 | tee -a "$LOG"

    section "push k=64 results (${model_key})"
    push_results "$DECOMP_DIR/${model_key}-story" "results/workspace_decomposition_k64/${model_key}-story"
}

# --------------------------------------------------------------------------
# Pull validation tables first (small, needed for correlation)
# --------------------------------------------------------------------------
section "pull C2 validation sweep tables"
pull_validation

# --------------------------------------------------------------------------
# Run all three primaries
# --------------------------------------------------------------------------
for model in llama31_8b qwen25_7b gemma2_9b; do
    run_model "$model"
done

# --------------------------------------------------------------------------
# Correlate J-space fraction with C2 metrics
# --------------------------------------------------------------------------
section "correlate J-space fraction with C2 metrics"
$PYTHON scripts/h8_prep_correlate.py --decomp-dir "$DECOMP_DIR" 2>&1 | tee -a "$LOG"

section "push correlation outputs"
push_results "$DECOMP_DIR" "results/workspace_decomposition_k64"

log "H8 prep complete. Outputs in $DECOMP_DIR and on HF."
