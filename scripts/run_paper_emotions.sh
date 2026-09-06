#!/usr/bin/env bash
# Add the Sofroniew-paper emotion cells to the wheel set and re-derive.
#
# Sofroniew et al. (2026) steer `desperate` and `calm` in the blackmail and
# reward-hacking experiments, and both plus `loving`/`happy`/`angry`/`afraid`
# in sycophancy. All but desperate and calm already have a Plutchik cell (see
# configs/wheel_paper.yaml for the full cross-reference). This script
# generates the missing cells and produces a new track containing the wheel
# cells *and* them.
#
# Why a new track. Vectors are cross-emotion centered: the grand mean is taken
# over every corpus in the set, so adding a cell changes every vector. The
# story-wheel32 vectors are frozen (the wheel report's numbers and the pending
# J-space decomposition refer to them), so this writes a new track instead of
# re-deriving in place.
#
# What is and is not recomputed. Activations depend only on their own corpus,
# never on the emotion set, so the 33 story-wheel32 activation files are
# reused as-is (linked in, pulled from HF if absent). The only GPU work is
# generate + extract for the new cells — roughly 2/33 of a wheel run. The
# derivation then runs over the whole assembled set, which is linear algebra
# on stored activations. Story corpora are never regenerated: generation is
# seeded *sampling*, so regenerating would silently produce different text.
#
# Usage::
#
#     # All three primaries (pod, tmux):
#     bash scripts/run_paper_emotions.sh --push --shutdown
#
#     # One model:
#     bash scripts/run_paper_emotions.sh --models "llama31_8b" --push
#
#     # Smoke test (few topics, Mac). Track is auto-suffixed -smoke so a
#     # short run cannot masquerade as the real thing.
#     bash scripts/run_paper_emotions.sh --models "qwen25_05b" \
#         --max-topics 3 --device mps --dtype float16
#
# Required env::
#
#     HF_TOKEN  — read+write to llm-psych/llm-psych-activations
#                 (needed to pull base activations, and with --push)
#
# Exit codes
# ----------
# 0   all models complete
# 1   user error (bad args)
# 2   pre-flight failure (missing config / no enabled cells)
# 3   pipeline stage failed (see log)
# 4   corpus-count assertion failed before derive

set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a
    source .env
    set +a
fi

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

MODELS="llama31_8b qwen25_7b gemma2_9b"
DEVICE_MAP="auto"
DTYPE="bfloat16"
STORIES_PER_TOPIC=""            # empty => story.yaml default (7); must match wheel32
MAX_TOPICS=""                   # empty => default (all 46); anything else = smoke
DO_PUSH=0
DO_SHUTDOWN=0
LOG_DIR="outputs"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
DATASET="llm-psych/llm-psych-activations"

usage() {
    cat <<'EOF' >&2
Usage: run_paper_emotions.sh [options]

Options:
  --models "<list>"         Space-separated Hydra model configs (default: all 3 primaries)
  --device <map>            device_map (default: auto)
  --dtype <dtype>           torch_dtype (default: bfloat16; use float16 on MPS)
  --stories-per-topic <N>   Override derivation.stories_per_topic (must match the base track)
  --max-topics <N>          Cap topic list; implies smoke mode (track gets -smoke suffix)
  --push                    Push activations + vectors + stories to HF after each model
  --shutdown                Stop the RunPod pod on exit (even on failure)
  -h, --help                Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)            MODELS="$2"; shift 2 ;;
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

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/run_paper_emotions_${TS}.log"

log()     { printf '\033[1;33m[run_paper]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
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
# Pre-flight: regenerate paper configs, resolve cells and track from the spec
# --------------------------------------------------------------------------

if [[ ! -f configs/wheel_paper.yaml ]]; then
    printf 'ERROR: configs/wheel_paper.yaml not found.\n' >&2
    exit 2
fi

log "regenerating paper emotion configs from configs/wheel_paper.yaml"
uv run python scripts/build_paper_emotion_configs.py 2>&1 | tee -a "$LOG"

CELL_ARRAY=()
while IFS= read -r c; do
    [[ -n "$c" ]] && CELL_ARRAY+=("$c")
done < <(uv run python scripts/build_paper_emotion_configs.py --list-enabled)

if [[ ${#CELL_ARRAY[@]} -eq 0 ]]; then
    printf 'ERROR: no enabled cells in configs/wheel_paper.yaml.\n' >&2
    exit 2
fi

TRACK=$(uv run python scripts/build_paper_emotion_configs.py --print-track)
BASE_TRACK=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/wheel_paper.yaml'))['base_track'])")
BASE_CORPORA=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/wheel_paper.yaml'))['base_corpora'])")

N_NEW=${#CELL_ARRAY[@]}
EXPECTED_TOTAL=$((BASE_CORPORA + N_NEW))

# Smoke mode: a truncated topic list must never write into the real namespace.
if [[ -n "$MAX_TOPICS" ]]; then
    TRACK="${TRACK}-smoke"
    log "smoke mode (--max-topics ${MAX_TOPICS}): track overridden to ${TRACK}"
fi

log "track=${TRACK}  base_track=${BASE_TRACK}  models=${MODELS}"
log "new cells (${N_NEW}): ${CELL_ARRAY[*]}"
log "expected corpora before derive: ${BASE_CORPORA} base + ${N_NEW} new = ${EXPECTED_TOTAL}"
log "device=${DEVICE_MAP}  dtype=${DTYPE}  push=${DO_PUSH}  shutdown=${DO_SHUTDOWN}"
log "log: ${LOG}"

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

# --------------------------------------------------------------------------
# Assemble the new track's activation directory from the base track.
#
# Hardlinks where the filesystem allows it (these files are large and the
# pod's disk budget is tight), copies otherwise. Never overwrites a file the
# extraction step just wrote.
# --------------------------------------------------------------------------

assemble_activations() {
    local model_key="$1"
    uv run python - "$model_key" "$TRACK" "$BASE_TRACK" "$DATASET" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

model_key, track, base_track, repo = sys.argv[1:5]
base_dir = Path("activations") / f"{model_key}-{base_track}"
dest_dir = Path("activations") / f"{model_key}-{track}"
dest_dir.mkdir(parents=True, exist_ok=True)

if not base_dir.is_dir() or not list(base_dir.glob("*.npz")):
    from huggingface_hub import snapshot_download

    pattern = f"activations/{model_key}-{base_track}/*"
    print(f"base activations absent locally; pulling {pattern} from {repo}")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(Path.cwd()),
    )

base_files = sorted(base_dir.glob("*.npz"))
if not base_files:
    raise SystemExit(f"no base activations found in {base_dir}")

linked = copied = skipped = 0
for src in base_files:
    dst = dest_dir / src.name
    if dst.exists():
        # Written by this run's extraction step, or a previous assembly.
        skipped += 1
        continue
    try:
        os.link(src, dst)
        linked += 1
    except OSError:
        shutil.copy2(src, dst)
        copied += 1

print(
    f"assembled {dest_dir}: {linked} linked, {copied} copied, "
    f"{skipped} already present ({len(list(dest_dir.glob('*.npz')))} total)"
)
PY
}

push_artefacts() {
    # push_artefacts <model_key> <new_cell_name>...
    local model_key="$1"; shift
    local slug="${model_key}-${TRACK}"
    section "push ${slug}"
    uv run python - "$model_key" "$TRACK" "$DATASET" "$@" <<'PY' || log "WARN: push failed for $slug"
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from huggingface_hub import HfApi  # noqa: E402

model_key, track, repo = sys.argv[1], sys.argv[2], sys.argv[3]
new_cells = sys.argv[4:]
slug = f"{model_key}-{track}"
api = HfApi()

# Stories are the irreplaceable artefact (seeded sampling); push them first.
# Activations are skipped: all but the new cells are hardlinks to the base
# track, which is already on the dataset. Push the new cells' files only.
targets = [
    (Path("data/derived/stories") / model_key / track, f"stories/{model_key}/{track}"),
    (Path("steering_vectors") / slug, f"steering_vectors/{slug}"),
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
            commit_message=f"run_paper_emotions: {slug} -> {path_in_repo}",
        )
    except Exception as exc:
        print(f"FAILED {folder} -> {path_in_repo}: {exc}", file=sys.stderr)
        failed.append(path_in_repo)
        continue
    print(f"pushed {folder} -> {path_in_repo}")

# New cells' activations, one file each (the rest are links to the base track).
act_dir = Path("activations") / slug
for name in new_cells:
    f = act_dir / f"{name}.npz"
    if not f.is_file():
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"activations/{slug}/{f.name}",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"run_paper_emotions: {slug} activations/{f.name}",
        )
        print(f"pushed {f}")
    except Exception as exc:
        print(f"FAILED {f}: {exc}", file=sys.stderr)
        failed.append(str(f))

if failed:
    print(f"PUSH INCOMPLETE: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
PY
}

# --------------------------------------------------------------------------
# Main loop
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

    OVERRIDES=()
    while IFS= read -r o; do
        OVERRIDES+=("$o")
    done < <(build_overrides "$model")

    # --- Step 1: generate stories for the new cells only ------------------
    # Neutral is NOT injected here (unlike run_wheel.sh): the derivation's
    # neutral corpus comes from the base track via assembly below.
    section "generate (${model_key})"
    stage_failed=0
    for cell in "${CELL_ARRAY[@]}"; do
        log "  generate ${cell}"
        if ! uv run python scripts/generate_emotion_stories.py \
                "${OVERRIDES[@]}" "emotion=${cell}" 2>&1 | tee -a "$LOG"; then
            log "  FAILED: generate ${cell}"
            stage_failed=1
        fi
    done
    if [[ "$stage_failed" -eq 1 ]]; then
        log "ERROR: generation failed for ${model_key} — skipping"
        FAILED_MODELS+=("$model"); free_model_cache "$model"; continue
    fi

    # --- Step 2: extract activations for the new cells --------------------
    section "extract (${model_key})"
    for cell in "${CELL_ARRAY[@]}"; do
        log "  extract ${cell}"
        if ! uv run python scripts/extract_story_activations.py \
                "${OVERRIDES[@]}" "emotion=${cell}" 2>&1 | tee -a "$LOG"; then
            log "  FAILED: extract ${cell}"
            stage_failed=1
        fi
    done
    if [[ "$stage_failed" -eq 1 ]]; then
        log "ERROR: extraction failed for ${model_key} — skipping derive"
        FAILED_MODELS+=("$model"); free_model_cache "$model"; continue
    fi

    # --- Step 3: assemble base activations into the new track -------------
    section "assemble (${model_key})"
    if ! assemble_activations "$model_key" 2>&1 | tee -a "$LOG"; then
        log "ERROR: assembly failed for ${model_key}"
        FAILED_MODELS+=("$model"); free_model_cache "$model"; continue
    fi

    # --- Step 4: assert the full corpus set, then derive ------------------
    act_dir="activations/${slug}"
    n_npz=$(find "$act_dir" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$n_npz" -ne "$EXPECTED_TOTAL" ]]; then
        log "ASSERTION FAILED: expected ${EXPECTED_TOTAL} .npz in ${act_dir}, found ${n_npz}"
        log "A partial set would corrupt the grand mean. Aborting."
        FAILED_MODELS+=("$model"); free_model_cache "$model"
        exit 4
    fi
    log "corpus check OK: ${n_npz}/${EXPECTED_TOTAL} in ${act_dir}"

    section "derive (${model_key})"
    if ! uv run python scripts/derive_story_steering_vectors.py \
            "${OVERRIDES[@]}" 2>&1 | tee -a "$LOG"; then
        log "ERROR: derive failed for ${model_key}"
        FAILED_MODELS+=("$model"); free_model_cache "$model"; continue
    fi

    # --- Step 5: push -----------------------------------------------------
    if [[ "$DO_PUSH" -eq 1 ]]; then
        # Strip the config prefix: cell names on disk are the bare emotion names.
        push_artefacts "$model_key" "${CELL_ARRAY[@]#paper_}"
    fi

    free_model_cache "$model"
    log "${model_key} DONE"
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    log "COMPLETE WITH FAILURES: ${FAILED_MODELS[*]}"
    log "Rerun with: --models \"${FAILED_MODELS[*]}\""
    exit 3
fi

log "ALL MODELS COMPLETE (${MODELS})"
log "Artefacts: activations/<model>-${TRACK}/, steering_vectors/<model>-${TRACK}/,"
log "           data/derived/stories/<model>/${TRACK}/"
log "NOTE: every vector in ${TRACK} is centered over ${EXPECTED_TOTAL}-1 emotions and is"
log "      NOT numerically comparable to the same emotion in ${BASE_TRACK}."
