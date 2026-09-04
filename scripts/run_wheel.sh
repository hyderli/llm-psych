#!/usr/bin/env bash
# Plutchik-wheel extraction: generate, extract, derive for all 33 cells.
#
# This is the wheel-track counterpart of run_story_pipeline.sh. It runs the
# same three scripts (generate -> extract -> derive) but at track=story-wheel32
# over the 33 wheel emotion configs, and adds the 33-file assertion before
# derive to protect against a partial set corrupting the grand mean.
#
# Designed for a single-GPU pod (>= 100 GB disk). Frees the HF model cache
# between models to stay within disk budget. After each model completes,
# pushes activations, steering vectors, story corpora and (if already
# generated) logit-lens validation reports to HF.
#
# Usage::
#
#     # All three primaries (pod, tmux, overnight):
#     bash scripts/run_wheel.sh --push --shutdown
#
#     # One model:
#     bash scripts/run_wheel.sh --models "llama31_8b" --push
#
#     # Smoke test (2 cells + neutral, Mac):
#     bash scripts/run_wheel.sh --models "qwen25_05b" --cells "wheel_ecstasy wheel_grief" \
#         --max-topics 3 --device mps --dtype float16
#
#     When --cells is set, the track is automatically changed to
#     story-wheel32-smoke so a partial run cannot write plausible-looking
#     garbage into the real wheel namespace. wheel_neutral is injected
#     automatically (derive needs neutral.npz for the PC projection).
#
# Required env::
#
#     HF_TOKEN  — read+write to llm-psych/llm-psych-activations (only with --push)
#
# Exit codes
# ----------
# 0   all models complete
# 1   user error (bad args)
# 2   pre-flight failure (missing config)
# 3   pipeline stage failed (see log)
# 4   33-file assertion failed before derive

set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Load .env if present so a fresh tmux pane or nohup'd shell inherits HF_TOKEN.
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a
    source .env
    set +a
fi

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

TRACK="story-wheel32"
MODELS="llama31_8b qwen25_7b gemma2_9b"
CELLS=""                        # empty = all 33 wheel configs
DEVICE_MAP="auto"
DTYPE="bfloat16"
STORIES_PER_TOPIC=""            # empty => story.yaml default (7)
MAX_TOPICS=""                   # empty => story.yaml default (null = all 46)
DO_PUSH=0
DO_SHUTDOWN=0
LOG_DIR="outputs"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
DATASET="llm-psych/llm-psych-activations"

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------

usage() {
    cat <<'EOF' >&2
Usage: run_wheel.sh [options]

Options:
  --models "<list>"         Space-separated Hydra model configs (default: all 3 primaries)
  --cells "<list>"          Space-separated wheel_* emotion configs to run (default: all 33)
  --device <map>            device_map (default: auto = single CUDA GPU)
  --dtype <dtype>           torch_dtype (default: bfloat16; float16 on MPS)
  --stories-per-topic <N>   Override derivation.stories_per_topic
  --max-topics <N>          Cap topic list (for smoke tests)
  --push                    Push activations + vectors to HF after each model
  --shutdown                Stop the RunPod pod on exit (even on failure)
  -h, --help                Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)            MODELS="$2"; shift 2 ;;
        --cells)             CELLS="$2"; shift 2 ;;
        --device)            DEVICE_MAP="$2"; shift 2 ;;
        --dtype)             DTYPE="$2"; shift 2 ;;
        --stories-per-topic) STORIES_PER_TOPIC="$2"; shift 2 ;;
        --max-topics)        MAX_TOPICS="$2"; shift 2 ;;
        --push)              DO_PUSH=1; shift ;;
        --shutdown)          DO_SHUTDOWN=1; shift ;;
        -h|--help)           usage 0 ;;
        *)                   printf 'Unknown arg: %s\n' "$1" >&2; usage 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Logging / shutdown trap
# --------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/run_wheel_${TS}.log"

log()     { printf '\033[1;36m[run_wheel]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

shutdown_pod() {
    if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
        if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
            log "stopping pod $RUNPOD_POD_ID"
            runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed; stop manually."
        else
            log "shutdown requested but runpodctl/RUNPOD_POD_ID unavailable; stop pod manually."
        fi
    fi
}
trap shutdown_pod EXIT

# --------------------------------------------------------------------------
# Resolve cells
# --------------------------------------------------------------------------

if [[ -z "$CELLS" ]]; then
    # All 33 wheel configs (sorted for deterministic order).
    # Use a while-read loop instead of mapfile for Bash 3.2 (macOS).
    CELL_ARRAY=()
    while IFS= read -r f; do
        CELL_ARRAY+=("$f")
    done < <(find configs/emotion -name 'wheel_*.yaml' -exec basename {} .yaml \; | sort)
    IS_SUBSET=0
else
    read -ra CELL_ARRAY <<< "$CELLS"
    IS_SUBSET=1

    # Auto-inject wheel_neutral: derive needs neutral.npz for the PC
    # projection. Forgetting it is a silent FileNotFoundError.
    has_neutral=0
    for c in "${CELL_ARRAY[@]}"; do
        [[ "$c" == "wheel_neutral" ]] && has_neutral=1
    done
    if [[ "$has_neutral" -eq 0 ]]; then
        CELL_ARRAY+=("wheel_neutral")
        log "auto-injected wheel_neutral (required by derive)"
    fi

    # Subset runs use a smoke track so partial vectors cannot masquerade
    # as real wheel output in steering_vectors/<model>-story-wheel32/.
    TRACK="story-wheel32-smoke"
    log "subset mode: track overridden to ${TRACK}"
fi

EXPECTED_FULL=33
N_CELLS=${#CELL_ARRAY[@]}

if [[ "$N_CELLS" -eq 0 ]]; then
    printf 'ERROR: no wheel configs found.\n' >&2
    exit 2
fi


log "track=${TRACK}  models=${MODELS}  cells=${N_CELLS}/${EXPECTED_FULL}"
log "device=${DEVICE_MAP}  dtype=${DTYPE}  push=${DO_PUSH}  shutdown=${DO_SHUTDOWN}"
log "budget: stories_per_topic=${STORIES_PER_TOPIC:-<default 7>}  max_topics=${MAX_TOPICS:-<default all>}"
log "log: ${LOG}"

# --------------------------------------------------------------------------
# Shared Hydra overrides
# --------------------------------------------------------------------------

build_overrides() {
    local model="$1"
    local -a ov=(
        "model=${model}"
        "model.device_map=${DEVICE_MAP}"
        "model.torch_dtype=${DTYPE}"
        "derivation=story"
        "track=${TRACK}"
    )
    [[ -n "$STORIES_PER_TOPIC" ]] && ov+=("derivation.stories_per_topic=${STORIES_PER_TOPIC}")
    [[ -n "$MAX_TOPICS" ]]        && ov+=("derivation.max_topics=${MAX_TOPICS}")
    printf '%s\n' "${ov[@]}"
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

free_model_cache() {
    local cfg="configs/model/$1.yaml"
    local hf_id cache
    hf_id=$(awk '/^hf_model_id:/{print $2}' "$cfg")
    cache="$HF_HOME/hub/models--$(printf '%s' "$hf_id" | sed 's#/#--#g')"
    if [[ -d "$cache" ]]; then
        log "freeing cache $cache"
        rm -rf "$cache"
    fi
}

push_artefacts() {
    local model_key="$1"
    local slug="${model_key}-${TRACK}"
    section "push ${slug}"
    uv run python - "$model_key" "$TRACK" "$DATASET" <<'PY' || log "WARN: push failed for $slug"
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from huggingface_hub import HfApi  # noqa: E402

model_key, track, repo = sys.argv[1], sys.argv[2], sys.argv[3]
slug = f"{model_key}-{track}"
api = HfApi()

# (local dir, path on the dataset). Stories are the irreplaceable artefact:
# generation is seeded *sampling*, so a corpus lost with the pod cannot be
# regenerated. Their repo path mirrors data/derived/ so pull_stories.py
# resolves them. Validation reports go to vector_validation/<slug>, where
# h8_prep.sh expects them.
targets = [
    (Path("activations") / slug, f"activations/{slug}"),
    (Path("steering_vectors") / slug, f"steering_vectors/{slug}"),
    (Path("data/derived/stories") / model_key / track, f"stories/{model_key}/{track}"),
    (Path("results/vector_validation") / slug, f"vector_validation/{slug}"),
]

failed = []
for folder, path_in_repo in targets:
    if not folder.is_dir():
        print(f"skip {folder} (not found)")
        continue
    try:
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=f"run_wheel: {slug} -> {path_in_repo}",
        )
    except Exception as exc:
        print(f"FAILED {folder} -> {path_in_repo}: {exc}", file=sys.stderr)
        failed.append(path_in_repo)
        continue
    print(f"pushed {folder} -> {path_in_repo}")

if failed:
    print(f"PUSH INCOMPLETE: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
PY
}

# --------------------------------------------------------------------------
# Main loop: per model
# --------------------------------------------------------------------------

FAILED_MODELS=()

for model in $MODELS; do
    cfg="configs/model/${model}.yaml"
    if [[ ! -f "$cfg" ]]; then
        printf 'ERROR: model config not found: %s\n' "$cfg" >&2
        exit 2
    fi
    hf_id=$(awk '/^hf_model_id:/{print $2}' "$cfg")
    model_key="${hf_id##*/}"
    slug="${model_key}-${TRACK}"

    section "MODEL: ${model} (${model_key})"
    log "cells: ${N_CELLS}"

    OVERRIDES=()
    while IFS= read -r o; do
        OVERRIDES+=("$o")
    done < <(build_overrides "$model")

    # ------------------------------------------------------------------
    # Step 1: generate stories for each cell
    # ------------------------------------------------------------------
    section "generate (${model_key})"
    gen_failed=0
    for cell in "${CELL_ARRAY[@]}"; do
        log "  generate ${cell}"
        if ! uv run python scripts/generate_emotion_stories.py \
                "${OVERRIDES[@]}" "emotion=${cell}" 2>&1 | tee -a "$LOG"; then
            log "  FAILED: generate ${cell}"
            gen_failed=1
        fi
    done
    if [[ "$gen_failed" -eq 1 ]]; then
        log "ERROR: generation had failures for ${model_key} — skipping"
        FAILED_MODELS+=("$model")
        free_model_cache "$model"
        continue
    fi

    # ------------------------------------------------------------------
    # Step 2: extract activations for each cell
    # ------------------------------------------------------------------
    section "extract (${model_key})"
    ext_failed=0
    for cell in "${CELL_ARRAY[@]}"; do
        log "  extract ${cell}"
        if ! uv run python scripts/extract_story_activations.py \
                "${OVERRIDES[@]}" "emotion=${cell}" 2>&1 | tee -a "$LOG"; then
            log "  FAILED: extract ${cell}"
            ext_failed=1
        fi
    done
    if [[ "$ext_failed" -eq 1 ]]; then
        log "ERROR: extraction had failures for ${model_key} — skipping derive"
        FAILED_MODELS+=("$model")
        free_model_cache "$model"
        continue
    fi

    # ------------------------------------------------------------------
    # Step 3: assert 33 npz files, then derive
    # ------------------------------------------------------------------
    act_dir="activations/${slug}"
    n_npz=$(find "$act_dir" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$N_CELLS" -eq "$EXPECTED_FULL" && "$n_npz" -ne "$EXPECTED_FULL" ]]; then
        log "ASSERTION FAILED: expected ${EXPECTED_FULL} .npz in ${act_dir}, found ${n_npz}"
        log "A partial set would corrupt the grand mean. Aborting derive for ${model_key}."
        FAILED_MODELS+=("$model")
        free_model_cache "$model"
        exit 4
    fi

    section "derive (${model_key})"
    if ! uv run python scripts/derive_story_steering_vectors.py \
            "${OVERRIDES[@]}" 2>&1 | tee -a "$LOG"; then
        log "ERROR: derive failed for ${model_key}"
        FAILED_MODELS+=("$model")
        free_model_cache "$model"
        continue
    fi

    # ------------------------------------------------------------------
    # Step 4: push to HF (optional)
    # ------------------------------------------------------------------
    if [[ "$DO_PUSH" -eq 1 ]]; then
        push_artefacts "$model_key"
    fi

    # ------------------------------------------------------------------
    # Step 5: free model cache for the next model
    # ------------------------------------------------------------------
    free_model_cache "$model"
    log "${model_key} DONE"
done

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    log "COMPLETE WITH FAILURES: ${FAILED_MODELS[*]}"
    log "Rerun with: --models \"${FAILED_MODELS[*]}\""
    exit 3
fi

log "ALL MODELS COMPLETE (${MODELS})"
log "Artefacts: activations/<model>-${TRACK}/, steering_vectors/<model>-${TRACK}/,"
log "           data/derived/stories/<model>/${TRACK}/, results/vector_validation/<model>-${TRACK}/"
