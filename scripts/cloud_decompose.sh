#!/usr/bin/env bash
# Run the J-space decomposition (plans/j-space-decomposition.md, phase 1)
# for one or more primary models on a cloud pod, and push the results to
# the private HF dataset.
#
# CPU-ONLY: no GPU is needed. The decomposition uses the weights-light
# loader (lm_head + final norm shards only) plus the pre-fitted
# Neuronpedia J-lens; everything else is linear algebra. A cheap CPU
# instance with fast HF egress finishes all three primaries in well
# under an hour; expected cost is cents, not dollars.
#
# Designed to survive preemption: each model pulls its own inputs,
# decomposes, and pushes its results before the next model starts, so a
# preemption costs at most one model of work. Re-running with the same
# args overwrites (does not append).
#
# Usage::
#
#     bash scripts/cloud_decompose.sh                       # all three primaries
#     bash scripts/cloud_decompose.sh --models "llama31_8b" # one model
#     bash scripts/cloud_decompose.sh --shutdown            # stop pod when done
#
# Required env::
#
#     HF_TOKEN — read access to the gated model repos (Llama, Gemma) and
#                read+write to llm-psych/llm-psych-activations
#                (loaded automatically from .env if cloud_bootstrap.sh was used)
#
# Exit codes
# ----------
# 0   all models decomposed and pushed
# 1   user error (bad args)
# 2   pre-flight failed (missing model config)
# 3   one or more models failed (see log; successful models are already pushed)

set -euo pipefail

# Load .env if present so a fresh tmux pane inherits HF_TOKEN.
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a
    source .env
    set +a
fi

# --------------------------------------------------------------------------
# Python interpreter: prefer the minimal CPU venv created by
# cloud_bootstrap_cpu_minimal.sh, fall back to uv's venv, then to uv run.
# --------------------------------------------------------------------------

PYTHON_CMD="uv run python"
if [[ -x ".venv-cpu/bin/python" ]]; then
    PYTHON_CMD=".venv-cpu/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
fi

# --------------------------------------------------------------------------
# Defaults (canonical phase-1 parameters — keep in sync with
# plans/j-space-decomposition.md and the PR #13 test plan)
# --------------------------------------------------------------------------

MODELS="llama31_8b qwen25_7b gemma2_9b"
EMOTIONS="admiration joy loathing sadness"
K=16
N_CANDIDATES=512
DO_SHUTDOWN=0
LOG_DIR="outputs"
DATASET_REPO="llm-psych/llm-psych-activations"

usage() {
    cat <<'EOF' >&2
Usage: cloud_decompose.sh [options]

Options:
  --models "<list>"       Space-separated model configs (default: "llama31_8b qwen25_7b gemma2_9b")
  --emotions "<list>"     Space-separated emotions (default: "admiration joy loathing sadness")
  --k <N>                 Max J-lens atoms (default: 16)
  --n-candidates <N>      Candidate pool size (default: 512)
  --shutdown              Stop the RunPod pod on exit (even on failure)
  -h, --help              Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)       MODELS="$2"; shift 2 ;;
        --emotions)     EMOTIONS="$2"; shift 2 ;;
        --k)            K="$2"; shift 2 ;;
        --n-candidates) N_CANDIDATES="$2"; shift 2 ;;
        --shutdown)     DO_SHUTDOWN=1; shift ;;
        -h|--help)      usage 0 ;;
        *)              printf 'Unknown arg: %s\n' "$1" >&2; usage 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/cloud_decompose_${TS}.log"

log() { printf '\033[1;35m[cloud_decompose]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

# Guaranteed shutdown (cost control), even if a model fails.
shutdown_pod() {
    if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
        if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
            log "Stopping RunPod pod $RUNPOD_POD_ID via runpodctl…"
            runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed — stop the pod manually."
        else
            log "Auto-shutdown requested but RUNPOD_POD_ID / runpodctl not available."
        fi
    fi
}
trap shutdown_pod EXIT

log "models=$MODELS  emotions=$EMOTIONS  k=$K  n_candidates=$N_CANDIDATES"
log "log file: $LOG"

# --------------------------------------------------------------------------
# Pre-flight: all model configs must exist before any work starts
# --------------------------------------------------------------------------

declare -A MODEL_KEYS
for model in $MODELS; do
    cfg="configs/model/${model}.yaml"
    if [[ ! -f "$cfg" ]]; then
        printf 'ERROR: model config not found: %s\n' "$cfg" >&2
        exit 2
    fi
    hf_id=$(awk '/^hf_model_id:/{print $2}' "$cfg")
    MODEL_KEYS[$model]="${hf_id##*/}"
done

# --------------------------------------------------------------------------
# Per-model: pull vectors -> decompose -> push results
# --------------------------------------------------------------------------

FAILED=""

pull_vectors() {
    # Download steering_vectors/<model_key>-story from the private dataset.
    local model_key="$1"
    $PYTHON_CMD - "$model_key" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_key = sys.argv[1]
repo_id = "llm-psych/llm-psych-activations"
pattern = f"steering_vectors/{model_key}-story/*"
snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[pattern], local_dir=str(Path.cwd()))
print(f"pulled steering_vectors/{model_key}-story from {repo_id}")
PY
}

push_results() {
    # Upload results/workspace_decomposition/<model_key>-story to the
    # private dataset, mirroring the on-disk layout (methods.md convention).
    local model_key="$1"
    $PYTHON_CMD - "$model_key" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi

model_key = sys.argv[1]
folder = Path.cwd() / "results" / "workspace_decomposition" / f"{model_key}-story"
if not folder.is_dir():
    raise SystemExit(f"missing results folder: {folder}")
HfApi().upload_folder(
    repo_id="llm-psych/llm-psych-activations",
    repo_type="dataset",
    folder_path=str(folder),
    path_in_repo=f"results/workspace_decomposition/{model_key}-story",
    commit_message=f"cloud_decompose: {model_key} J-space decomposition",
)
print(f"pushed {folder} -> llm-psych/llm-psych-activations")
PY
}

for model in $MODELS; do
    model_key="${MODEL_KEYS[$model]}"
    section "model ${model} (${model_key})"

    # Failure isolation: one model failing must not block the others.
    if (
        set -euo pipefail

        section "pull steering_vectors (${model_key})"
        pull_vectors "$model_key" 2>&1 | tee -a "$LOG"

        section "decompose (${model_key})"
        # shellcheck disable=SC2086
        $PYTHON_CMD scripts/decompose_emotion_vectors.py \
            --model-config "configs/model/${model}.yaml" \
            --lens-source neuronpedia \
            --emotions $EMOTIONS \
            --k "$K" --n-candidates "$N_CANDIDATES" 2>&1 | tee -a "$LOG"

        section "push results (${model_key})"
        push_results "$model_key" 2>&1 | tee -a "$LOG"
    ); then
        log "OK: ${model_key}"
    else
        log "FAILED: ${model_key} (continuing with remaining models)"
        FAILED="${FAILED} ${model}"
    fi
done

if [[ -n "$FAILED" ]]; then
    log "FAILED models:${FAILED}"
    exit 3
fi

log "All models decomposed and pushed. DONE"
