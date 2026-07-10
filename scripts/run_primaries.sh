#!/usr/bin/env bash
# End-to-end PRIMARY run on a pod.
#
# For each primary model: derive story vectors (+push), run the full C2
# validation suite (logit-lens + implicit + intensity, headline & sweep),
# then free that model's weights so the next one fits. Finally push the
# validation reports and (optionally) stop the pod.
#
# Prereqs:
#   * a pod with a >=60 GB container disk (three 7-9B model caches + venv +
#     activations) OR a network volume; export HF_HOME onto the volume.
#   * cloud_bootstrap.sh already run (uv env + .env with HF_TOKEN).
#   * run inside tmux or via nohup — the story generation is multi-hour.
#
# Usage (on the pod):
#   tmux new -s primaries
#   nohup bash scripts/run_primaries.sh --shutdown > outputs/primaries.out 2>&1 &
#   tail -f outputs/primaries.out
#
# Options:
#   --models "<list>"   space-separated Hydra model configs
#                       (default: "llama31_8b qwen25_7b gemma2_9b")
#   --shutdown          runpodctl stop pod $RUNPOD_POD_ID at the very end

set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on CUDA

MODELS="llama31_8b qwen25_7b gemma2_9b"
DO_SHUTDOWN=0
DTYPE=bfloat16
DEVICE=cuda
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
DATASET="llm-psych/llm-psych-activations"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)   MODELS="$2"; shift 2 ;;
        --shutdown) DO_SHUTDOWN=1; shift ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 1 ;;
    esac
done

log() { printf '\033[1;35m[primaries]\033[0m %s\n' "$*" >&2; }

# Guaranteed shutdown: registered on EXIT so the pod stops (when --shutdown)
# even if a stage aborts the script — an unattended overnight failure must
# not leave the pod billing until someone notices.
shutdown_pod() {
    if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
        if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
            log "stopping pod $RUNPOD_POD_ID"
            runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed; stop manually."
        else
            log "shutdown requested but runpodctl/RUNPOD_POD_ID unavailable; stop the pod manually."
        fi
    fi
}
trap shutdown_pod EXIT

run_validators() {
    local m="$1"
    # diagnostics — never abort the whole run if one fails
    uv run python scripts/validate_logit_lens.py        --model "$m" --device "$DEVICE" --dtype "$DTYPE"          || log "WARN logit_lens $m failed"
    uv run python scripts/validate_implicit_scenarios.py --model "$m" --device "$DEVICE" --dtype "$DTYPE"          || log "WARN implicit $m failed"
    uv run python scripts/validate_implicit_scenarios.py --model "$m" --device "$DEVICE" --dtype "$DTYPE" --sweep  || log "WARN implicit-sweep $m failed"
    uv run python scripts/validate_intensity_semantic.py --model "$m" --device "$DEVICE" --dtype "$DTYPE"          || log "WARN intensity $m failed"
    uv run python scripts/validate_intensity_semantic.py --model "$m" --device "$DEVICE" --dtype "$DTYPE" --sweep  || log "WARN intensity-sweep $m failed"
}

free_model_cache() {
    # Remove this model's HF weights once vectors are made + pushed, so the
    # next 7-9B model fits on a modest disk. HF cache dir name: models--<org>--<name>.
    local cfg="configs/model/$1.yaml"
    local hf_id cache
    hf_id=$(awk '/^hf_model_id:/{print $2}' "$cfg")
    cache="$HF_HOME/hub/models--$(printf '%s' "$hf_id" | sed 's#/#--#g')"
    if [[ -d "$cache" ]]; then
        log "freeing cache $cache"
        rm -rf "$cache"
    fi
}

FAILED_MODELS=()

for m in $MODELS; do
    log "================  $m  ================"
    log "story pipeline (generate -> extract -> derive -> probes -> push)"
    # Failure isolation: one model failing must not kill the remaining
    # primaries (each is an independent multi-hour investment).
    if ! bash scripts/run_story_pipeline.sh --model "$m" --push; then
        log "ERROR: story pipeline failed for $m — skipping its validators, continuing with remaining models"
        FAILED_MODELS+=("$m")
        free_model_cache "$m"   # partial cache would starve the next model of disk
        continue
    fi
    log "C2 validation suite"
    run_validators "$m"
    free_model_cache "$m"
    log "$m done"
done

log "pushing C2 validation reports to HF dataset"
uv run python - <<PY || log "WARN report push failed (reports are still on the pod under results/vector_validation/)"
from huggingface_hub import upload_folder
upload_folder(repo_id="${DATASET}", repo_type="dataset",
              folder_path="results/vector_validation",
              path_in_repo="vector_validation",
              commit_message="C2 validation reports (primaries)")
print("pushed vector_validation/")
PY

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    log "COMPLETE WITH FAILURES: ${FAILED_MODELS[*]} — their artefacts are NOT on HF; rerun with --models \"${FAILED_MODELS[*]}\""
    exit 1   # EXIT trap still stops the pod if --shutdown was given
fi
log "ALL PRIMARIES COMPLETE"
# (shutdown handled by the EXIT trap)
