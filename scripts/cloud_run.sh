#!/usr/bin/env bash
# Run the full extraction + probe-training + HF-push pipeline for one
# model and a list of emotions, designed to survive preemption.
#
# After each emotion, the script immediately pushes that emotion's
# artefacts to HF before moving on to the next. If the pod is
# preempted mid-run, re-running with the same args picks up where it
# left off (existing artefacts are overwritten, not appended).
#
# Usage::
#
#     bash scripts/cloud_run.sh \
#         --model llama31_8b \
#         --emotions "anger joy fear sadness"
#
#     # Optional: power off the pod when finished (cost control)
#     bash scripts/cloud_run.sh --model llama31_8b --shutdown
#
# Required env::
#
#     HF_TOKEN                — read+write to llm-psych/llm-psych-activations
#     (loaded automatically from .env if cloud_bootstrap.sh was used)
#
# Exit codes
# ----------
# 0   all stages completed successfully
# 1   user error (bad args)
# 2   pre-flight failed (dirty git tree, missing model config)
# 3   pipeline stage failed (see log)

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

MODEL_CONFIG=""                      # e.g. "llama31_8b" (matches configs/model/<NAME>.yaml)
EMOTIONS="anger joy fear sadness"    # space-separated list
EXTRACT_BS=16                        # safe for 24 GB VRAM on 7-8B bf16
DO_SHUTDOWN=0
LOG_DIR="outputs"

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------

usage() {
    cat <<'EOF' >&2
Usage: cloud_run.sh --model <config> [options]

Required:
  --model <name>         Hydra model config (e.g. llama31_8b)

Options:
  --emotions "<list>"    Space-separated list (default: "anger joy fear sadness")
  --batch-size <N>       Per-batch prompts during extraction (default: 16)
  --shutdown             Run `runpodctl stop pod $RUNPOD_POD_ID` on success
  -h, --help             Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL_CONFIG="$2"; shift 2 ;;
        --emotions)    EMOTIONS="$2"; shift 2 ;;
        --batch-size)  EXTRACT_BS="$2"; shift 2 ;;
        --shutdown)    DO_SHUTDOWN=1; shift ;;
        -h|--help)     usage 0 ;;
        *)             printf 'Unknown arg: %s\n' "$1" >&2; usage 1 ;;
    esac
done

if [[ -z "$MODEL_CONFIG" ]]; then
    printf 'ERROR: --model is required.\n' >&2
    usage 1
fi

# Resolve the model key (the filesystem-safe id used in paths). This must
# match what extract_activations.py derives at runtime, which is the
# basename of hf_model_id. We extract it from the Hydra config file.
MODEL_CONFIG_FILE="configs/model/${MODEL_CONFIG}.yaml"
if [[ ! -f "$MODEL_CONFIG_FILE" ]]; then
    printf 'ERROR: model config not found: %s\n' "$MODEL_CONFIG_FILE" >&2
    exit 2
fi
HF_MODEL_ID=$(awk '/^hf_model_id:/{print $2}' "$MODEL_CONFIG_FILE")
MODEL_KEY="${HF_MODEL_ID##*/}"
if [[ -z "$MODEL_KEY" ]]; then
    printf 'ERROR: could not parse hf_model_id from %s\n' "$MODEL_CONFIG_FILE" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/cloud_run_${MODEL_KEY}_${TS}.log"

log() { printf '\033[1;34m[cloud_run]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

log "model_config=$MODEL_CONFIG  model_key=$MODEL_KEY"
log "emotions=$EMOTIONS  batch_size=$EXTRACT_BS"
log "log file: $LOG"

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

run_step() {
    # Run a uv command, tee output to the log, fail loudly on non-zero.
    local label="$1"; shift
    section "$label"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        log "STAGE FAILED: $label"
        exit 3
    fi
}

for emotion in $EMOTIONS; do
    run_step "extract ${emotion} (model=${MODEL_CONFIG})" \
        uv run python scripts/extract_activations.py \
        "model=${MODEL_CONFIG}" "emotion=${emotion}" \
        "extract.batch_size=${EXTRACT_BS}"

    run_step "train_probes ${emotion}" \
        uv run python scripts/train_probes.py \
        "model=${MODEL_CONFIG}" "emotion=${emotion}"

    # Push after every emotion so preemption costs at most one emotion of work.
    run_step "push activations (${emotion} just finished)" \
        uv run python scripts/sync_hf.py push activations \
        --model "${MODEL_KEY}" \
        --message "cloud_run: ${MODEL_KEY} activations through ${emotion}"

    run_step "push probes (${emotion})" \
        uv run python scripts/sync_hf.py push probes \
        --model "${MODEL_KEY}" \
        --message "cloud_run: ${MODEL_KEY} probes through ${emotion}"

    run_step "push steering_vectors (${emotion})" \
        uv run python scripts/sync_hf.py push steering_vectors \
        --model "${MODEL_KEY}" \
        --message "cloud_run: ${MODEL_KEY} steering vectors through ${emotion}"
done

log "All emotions complete for ${MODEL_KEY}."

# --------------------------------------------------------------------------
# Optional auto-shutdown (cost control)
# --------------------------------------------------------------------------

if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
    section "auto-shutdown"
    if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
        log "Stopping RunPod pod $RUNPOD_POD_ID via runpodctl…"
        runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed (artefacts are safe on HF)."
    else
        log "Auto-shutdown requested but RUNPOD_POD_ID / runpodctl not available."
        log "Stop the pod manually from the RunPod console to avoid charges."
    fi
fi

log "DONE"
