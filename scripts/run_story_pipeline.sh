#!/usr/bin/env bash
# Real-scale story-method pipeline for one model, pod-ready.
#
# This is the production counterpart of
# ``scripts/run_smoketest_story_qwen05b.sh``. The smoke script proves the
# generate -> extract_story -> derive_story -> H1-probe path *runs* at a
# tiny budget on the Mac; this one runs it at the REAL budget
# (``configs/derivation/story.yaml`` defaults: stories_per_topic=7,
# max_topics=null, pool_start_token=50, min_story_tokens=60) on any model,
# pushes artefacts to HF, and can power the pod off when done.
#
# Used for the Priority-1 gate (dev-fleet story corpora + confound audit
# on real story activations) and, once the gate passes, the primary
# story runs. See ``plans/story-gate-run.md`` and ``docs/cloud_runbook.md``.
#
# NOTE: kept as a separate file from the smoke script on purpose — the
# smoke script is verified-working (RESEARCH_LOG 2026-06-13) and its tiny
# fixed budget (pool_start=5, min_tok=10) exists only so 10-token smoke
# stories survive pooling. Do NOT carry those smoke overrides here.
#
# Usage::
#
#     # Dev-fleet gate run (pod), pushing artefacts to HF:
#     bash scripts/run_story_pipeline.sh --model qwen25_05b --push
#     bash scripts/run_story_pipeline.sh --model llama32_1b --push
#     bash scripts/run_story_pipeline.sh --model gemma2_2b  --push --shutdown
#
#     # A primary, once the gate passes:
#     bash scripts/run_story_pipeline.sh --model gemma2_9b --push --shutdown
#
# Required env::
#
#     HF_TOKEN   — read+write to llm-psych/llm-psych-activations (only
#                  needed with --push; loaded from .env if cloud_bootstrap
#                  was used).
#
# Exit codes
# ----------
# 0   all stages completed successfully
# 1   user error (bad args)
# 2   pre-flight failed (missing model config)
# 3   pipeline stage failed (see log)

set -euo pipefail

# Let any device-unsupported op fall back gracefully (matters on MPS; a
# harmless no-op on CUDA/CPU).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

MODEL_CONFIG=""                 # e.g. "qwen25_05b" (configs/model/<NAME>.yaml)
# "auto" is the repo's documented production device_map (src/llm_psych/
# models.py): on a single CUDA GPU it places the whole model on the GPU via
# accelerate. Do NOT use "cpu"/"mps" here — those take load_model's
# single-device .to() path, which keeps the model off the GPU. On the Mac,
# use the smoke script instead.
DEVICE_MAP="auto"               # pod default (single CUDA GPU)
DTYPE="bfloat16"                # pod default; "float16" on MPS
STORIES_PER_TOPIC=""            # empty => use story.yaml default (7)
MAX_TOPICS=""                   # empty => use story.yaml default (null = all)
DO_PUSH=0
DO_SHUTDOWN=0
LOG_DIR="outputs"

# Emotions generated/extracted (neutral is the reference class, generated
# but not a probe target).
GEN_EMOS=(admiration joy loathing sadness neutral)
PROBE_EMOS=(admiration joy loathing sadness)

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------

usage() {
    cat <<'EOF' >&2
Usage: run_story_pipeline.sh --model <config> [options]

Required:
  --model <name>            Hydra model config (e.g. qwen25_05b, gemma2_9b)

Options:
  --device <map>            device_map (default: auto = single CUDA GPU)
  --dtype <dtype>           torch_dtype (default: bfloat16; float16 on MPS)
  --stories-per-topic <N>   override derivation.stories_per_topic (default: 7)
  --max-topics <N>          cap the topic list (default: all topics)
  --push                    sync activations/probes/steering_vectors to HF
  --shutdown                runpodctl stop pod $RUNPOD_POD_ID on success
  -h, --help                Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL_CONFIG="$2"; shift 2 ;;
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

if [[ -z "$MODEL_CONFIG" ]]; then
    printf 'ERROR: --model is required.\n' >&2
    usage 1
fi

# Resolve the filesystem-safe model key (basename of hf_model_id), matching
# what the story scripts derive at runtime. Same approach as cloud_run.sh.
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
# Story artefacts live under the "-story" suffixed key.
STORY_KEY="${MODEL_KEY}-story"

# --------------------------------------------------------------------------
# Hydra overrides shared by every step
# --------------------------------------------------------------------------

OVERRIDES=(
    "model=${MODEL_CONFIG}"
    "model.device_map=${DEVICE_MAP}"
    "model.torch_dtype=${DTYPE}"
    "derivation=story"
)
# Only override the budget knobs when explicitly requested; otherwise the
# real defaults in configs/derivation/story.yaml apply (stories_per_topic=7,
# max_topics=null, pool_start_token=50, min_story_tokens=60).
[[ -n "$STORIES_PER_TOPIC" ]] && OVERRIDES+=("derivation.stories_per_topic=${STORIES_PER_TOPIC}")
[[ -n "$MAX_TOPICS" ]]        && OVERRIDES+=("derivation.max_topics=${MAX_TOPICS}")

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/run_story_${MODEL_KEY}_${TS}.log"

log()     { printf '\033[1;34m[run_story]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

log "model_config=${MODEL_CONFIG}  model_key=${MODEL_KEY}  story_key=${STORY_KEY}"
log "device=${DEVICE_MAP}  dtype=${DTYPE}  push=${DO_PUSH}  shutdown=${DO_SHUTDOWN}"
log "budget: stories_per_topic=${STORIES_PER_TOPIC:-<default 7>}  max_topics=${MAX_TOPICS:-<default all>}"
log "log file: ${LOG}"

run_step() {
    local label="$1"; shift
    section "$label"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        log "STAGE FAILED: ${label}"
        exit 3
    fi
}

# --------------------------------------------------------------------------
# Step 1: generate stories (per emotion + neutral)
# --------------------------------------------------------------------------

for e in "${GEN_EMOS[@]}"; do
    run_step "generate ${e}" \
        uv run python scripts/generate_emotion_stories.py \
        "${OVERRIDES[@]}" "emotion=${e}"
done

# --------------------------------------------------------------------------
# Step 2: extract pooled story activations (per emotion + neutral)
# --------------------------------------------------------------------------

for e in "${GEN_EMOS[@]}"; do
    run_step "extract_story ${e}" \
        uv run python scripts/extract_story_activations.py \
        "${OVERRIDES[@]}" "emotion=${e}"
done

# --------------------------------------------------------------------------
# Step 3: derive story steering vectors (once; discovers emotions)
# --------------------------------------------------------------------------

run_step "derive_story_steering_vectors" \
    uv run python scripts/derive_story_steering_vectors.py "${OVERRIDES[@]}"

# --------------------------------------------------------------------------
# Step 4: H1 probes on story activations (per emotion vs neutral)
# --------------------------------------------------------------------------

for e in "${PROBE_EMOS[@]}"; do
    run_step "train_probes (story) ${e}" \
        uv run python scripts/train_probes.py "${OVERRIDES[@]}" "emotion=${e}"
done

log "Story pipeline complete for ${STORY_KEY}."

# --------------------------------------------------------------------------
# Step 5: push artefacts to HF (optional)
# --------------------------------------------------------------------------

if [[ "$DO_PUSH" -eq 1 ]]; then
    for kind in activations probes steering_vectors; do
        run_step "push ${kind} (${STORY_KEY})" \
            uv run python scripts/sync_hf.py push "${kind}" \
            --model "${STORY_KEY}" \
            --message "run_story: ${STORY_KEY} ${kind}"
    done
    log "Pushed activations/probes/steering_vectors for ${STORY_KEY} to HF."
else
    log "Skipping HF push (--push not set). Artefacts are local only."
fi

# --------------------------------------------------------------------------
# Step 6: optional auto-shutdown (cost control)
# --------------------------------------------------------------------------

if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
    section "auto-shutdown"
    if [[ "$DO_PUSH" -ne 1 ]]; then
        log "WARNING: --shutdown without --push — local artefacts will be lost when the pod stops."
    fi
    if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
        log "Stopping RunPod pod ${RUNPOD_POD_ID} via runpodctl..."
        runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed (artefacts are safe on HF if --push ran)."
    else
        log "Auto-shutdown requested but RUNPOD_POD_ID / runpodctl not available."
        log "Stop the pod manually from the RunPod console to avoid charges."
    fi
fi

log "DONE"
log "Next: audit on story activations ->"
log "  uv run python scripts/audit_h1_confounds.py --story --model-key ${STORY_KEY} --emotions admiration joy loathing sadness"
