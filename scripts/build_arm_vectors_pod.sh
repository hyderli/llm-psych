#!/usr/bin/env bash
# Build H8 arm vectors for a steering mixture on a cloud/RunPod CPU instance,
# verify them, and push the results to the private HF dataset.
#
# CPU-only: the J-space pursuit is linear algebra on the unembedding + lens.
# A pod with >= 32 GB RAM is recommended; peak usage comes from the float32
# unembedding and temporary pursuit tensors.
#
# Usage::
#
#     export HF_TOKEN=hf_xxx
#     export LAYER=22
#     bash scripts/build_arm_vectors_pod.sh
#
# Optional env / overrides::
#
#     MODEL=gemma2_9b                    # model config basename (default: gemma2_9b)
#     TRACK=story-wheel32                # extraction track (default: story-wheel32)
#     MIX="contempt=1 aggressiveness=1"  # mixture terms (default: contempt + aggressiveness)
#     TAG=ca_unit                        # arm filename prefix (default: ca_unit)
#     ALPHA=1                            # saved vector norm scalar (default: 1)
#     K=64                               # pursuit atom budget (default: 64)
#     N_CANDIDATES=2048                  # candidate pool size (default: 2048)
#
# Required env::
#
#     HF_TOKEN — read access to gated model repos and read+write to
#                llm-psych/llm-psych-activations
#
# Exit codes
# ----------
# 0   arms built, verified, and pushed
# 1   user error (missing env / bad args)
# 2   pre-flight failed (repo / config missing)
# 3   download or build failed
# 4   verification failed
# 5   HF upload failed

set -euo pipefail

# Load .env if present so a fresh tmux pane inherits HF_TOKEN.
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a
    source .env
    set +a
fi

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

MODEL="${MODEL:-gemma2_9b}"
TRACK="${TRACK:-story-wheel32}"
MIX="${MIX:-contempt=1 aggressiveness=1}"
TAG="${TAG:-ca_unit}"
ALPHA="${ALPHA:-1}"
K="${K:-64}"
N_CANDIDATES="${N_CANDIDATES:-2048}"

DO_SHUTDOWN=0
LOG_DIR="outputs"
DATASET_REPO="llm-psych/llm-psych-activations"

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------

usage() {
    cat <<'EOF' >&2
Usage: build_arm_vectors_pod.sh [options]

Options:
  --model <name>        Model config basename (default: gemma2_9b)
  --track <name>        Extraction track (default: story-wheel32)
  --layer <int>         Decoder layer to inject at (required)
  --mix "n=c ..."       Mixture terms (default: "contempt=1 aggressiveness=1")
  --tag <str>           Arm filename prefix (default: ca_unit)
  --alpha <float>       Saved vector norm scalar (default: 1)
  --k <int>             Max pursuit atoms (default: 64)
  --n-candidates <int>  Candidate pool size (default: 2048)
  --shutdown            Stop the RunPod pod on exit (even on failure)
  -h, --help            Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --track)          TRACK="$2"; shift 2 ;;
        --layer)          LAYER="$2"; shift 2 ;;
        --mix)            MIX="$2"; shift 2 ;;
        --tag)            TAG="$2"; shift 2 ;;
        --alpha)          ALPHA="$2"; shift 2 ;;
        --k)              K="$2"; shift 2 ;;
        --n-candidates)   N_CANDIDATES="$2"; shift 2 ;;
        --shutdown)       DO_SHUTDOWN=1; shift ;;
        -h|--help)        usage 0 ;;
        *)                printf 'Unknown arg: %s\n' "$1" >&2; usage 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Required env
# --------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
    printf 'ERROR: HF_TOKEN is not set. Export it before running:\n    export HF_TOKEN=hf_xxx\n' >&2
    exit 1
fi

if [[ -z "${LAYER:-}" ]]; then
    printf 'ERROR: --layer / LAYER is required.\n' >&2
    exit 1
fi

if ! [[ "$LAYER" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: LAYER must be an integer, got: %s\n' "$LAYER" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/build_arm_vectors_${TAG}_L${LAYER}_${TS}.log"

log()     { printf '\033[1;36m[build_arm]\033[0m %s\n' "$*" | tee -a "$LOG" >&2; }
section() { printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

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

log "model=$MODEL  track=$TRACK  layer=$LAYER  tag=$TAG  alpha=$ALPHA  k=$K  n_candidates=$N_CANDIDATES"
log "log file: $LOG"

# --------------------------------------------------------------------------
# Python interpreter
# --------------------------------------------------------------------------

PYTHON_CMD="uv run --locked python"
if [[ -x ".venv-cpu/bin/python" ]]; then
    PYTHON_CMD=".venv-cpu/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
fi
log "python command: $PYTHON_CMD"

# --------------------------------------------------------------------------
# Pre-flight: config must exist
# --------------------------------------------------------------------------

MODEL_CFG="configs/model/${MODEL}.yaml"
if [[ ! -f "$MODEL_CFG" ]]; then
    log "ERROR: model config not found: $MODEL_CFG"
    exit 2
fi

HF_MODEL_ID=$(awk '/^hf_model_id:/{print $2}' "$MODEL_CFG")
HF_REVISION=$(awk '/^hf_revision:/{print $2}' "$MODEL_CFG")
MODEL_KEY="${HF_MODEL_ID##*/}"
log "HF model: $HF_MODEL_ID (revision: ${HF_REVISION:-null})"

# --------------------------------------------------------------------------
# Download only the source vectors needed for the mixture
# --------------------------------------------------------------------------

section "download source vectors"

$PYTHON_CMD - "$MODEL_KEY" "$TRACK" "$LAYER" "$MIX" <<'PY' | tee -a "$LOG"
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

model_key, track, layer, mix = sys.argv[1:]
layer = int(layer)
repo = "llm-psych/llm-psych-activations"
folder = f"steering_vectors/{model_key}-{track}"

emotions = []
for term in mix.split():
    name, _, _ = term.partition("=")
    if not name:
        raise SystemExit(f"bad mix term: {term!r}")
    emotions.append(name)

api = HfApi()
revision = api.repo_info(repo, repo_type="dataset").sha
print(f"dataset revision: {revision}", flush=True)

for emotion in emotions:
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=f"{folder}/{emotion}_layer{layer}.npy",
        local_dir=".",
    )
    print(path, flush=True)

os.makedirs("results/jspace_gate", exist_ok=True)
with open(f"results/jspace_gate/.download_revision_{model_key}_{track}_L{layer}", "w") as fh:
    fh.write(revision)
PY

# --------------------------------------------------------------------------
# Build the arms
# --------------------------------------------------------------------------

section "build arm vectors"

$PYTHON_CMD scripts/build_arm_vectors.py \
    --model-config "$MODEL_CFG" \
    --track "$TRACK" \
    --layer "$LAYER" \
    --alpha "$ALPHA" \
    --mix $MIX \
    --unit-normalise \
    --tag "$TAG" \
    --k "$K" \
    --n-candidates "$N_CANDIDATES" \
    --no-ladder \
    2>&1 | tee -a "$LOG"

# --------------------------------------------------------------------------
# Verify outputs
# --------------------------------------------------------------------------

section "verify arm vectors"

$PYTHON_CMD - "$MODEL_KEY" "$TRACK" "$LAYER" "$TAG" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

import numpy as np

model_key, track, layer, tag = sys.argv[1:]
layer = int(layer)
report_path = Path(f"results/jspace_gate/arms_{model_key}_{track}_L{layer}_{tag}.json")
report = json.loads(report_path.read_text())

assert report["layer"] == layer, f"report layer mismatch: {report['layer']} != {layer}"
assert report["lens"]["layer_used"] == layer, f"lens layer mismatch: {report['lens']['layer_used']}"
assert report["unit_normalised"] is True, "unit_normalise was not applied"
assert report["target_norm"] > 0, "target_norm must be positive"

folder = Path(f"steering_vectors/{model_key}-{track}")
expected = [f"{tag}_{sign}_{arm}_layer{layer}.npy"
            for sign in ("pos", "neg")
            for arm in ("full", "jspace", "resid", "randatom")]
first_shape = None
for name in expected:
    path = folder / name
    vector = np.load(path, allow_pickle=False)
    assert vector.ndim == 1, f"{path}: expected 1-D vector, got {vector.ndim}"
    if first_shape is None:
        first_shape = vector.shape[0]
    assert vector.shape[0] == first_shape, f"{path}: shape {vector.shape} != {first_shape}"
    assert np.isfinite(vector).all(), f"{path}: non-finite values"
    assert np.isclose(np.linalg.norm(vector), report["target_norm"], rtol=1e-4), f"{path}: norm mismatch"
    print(f"OK: {path}", flush=True)

print(json.dumps(report, indent=2), flush=True)
print("All eight arm files passed verification.", flush=True)
PY

# --------------------------------------------------------------------------
# Record provenance and upload
# --------------------------------------------------------------------------

section "upload to HF dataset"

$PYTHON_CMD - "$MODEL" "$MODEL_KEY" "$TRACK" "$LAYER" "$TAG" "$ALPHA" "$K" "$N_CANDIDATES" "$MIX" <<'PY' | tee -a "$LOG"
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

model, model_key, track, layer, tag, alpha, k, n_candidates, mix = sys.argv[1:]
layer = int(layer)
alpha = float(alpha)
k = int(k)
n_candidates = int(n_candidates)
repo = "llm-psych/llm-psych-activations"
folder = f"steering_vectors/{model_key}-{track}"
report_path = Path(f"results/jspace_gate/arms_{model_key}_{track}_L{layer}_{tag}.json")
rev_path = Path(f"results/jspace_gate/.download_revision_{model_key}_{track}_L{layer}")

provenance_path = report_path.with_suffix(".provenance.json")
cfg = yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text())

api = HfApi()
lens_repo = "neuronpedia/jacobian-lens"
try:
    lens_revision = api.repo_info(lens_repo).sha
except Exception:
    lens_revision = None

provenance = {
    "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "model_id": cfg["hf_model_id"],
    "model_revision": cfg.get("hf_revision"),
    "dataset_revision": rev_path.read_text().strip() if rev_path.exists() else None,
    "lens_repo": lens_repo,
    "lens_revision": lens_revision,
    "model_key": model_key,
    "track": track,
    "layer": layer,
    "tag": tag,
    "alpha": alpha,
    "k": k,
    "n_candidates": n_candidates,
    "mix": mix,
    "unit_normalised": True,
}
provenance_path.write_text(json.dumps(provenance, indent=2))

files = [
    f"{folder}/{tag}_{sign}_{arm}_layer{layer}.npy"
    for sign in ("pos", "neg")
    for arm in ("full", "jspace", "resid", "randatom")
]
files.append(str(report_path))
files.append(str(provenance_path))

for path in files:
    if not Path(path).is_file():
        raise SystemExit(f"missing file: {path}")

api = HfApi()
existing = set(api.list_repo_files(repo, repo_type="dataset"))
collisions = existing.intersection(files)
if collisions:
    raise SystemExit(f"STOP: these HF paths already exist: {sorted(collisions)}")

commit = api.upload_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path=str(Path.cwd()),
    allow_patterns=files,
    commit_message=f"Build {tag} arms for {model_key} {track} layer {layer}",
)

remote = set(api.list_repo_files(repo, repo_type="dataset", revision=commit.oid))
assert set(files).issubset(remote), "uploaded file listing is incomplete"
print(f"Verified {len(files)} files at HF dataset revision: {commit.oid}", flush=True)
PY

log "done"
