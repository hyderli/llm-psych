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
SIGNS="${SIGNS:-pos neg}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
# The ladder is OFF by default: it costs N_LADDER_DRAWS x len(LADDER_DEG)+1
# extra emits and most runs do not need it. --ladder turns it on. NOTE that
# enabling it changes the rng state consumed by gauss/shuffle, which are drawn
# after it -- so a ladder build must use a different --tag or it will overwrite
# control arms with different draws.
LADDER="${LADDER:-0}"
# The orthogonal plane (J19) is also OFF by default. It is drawn AFTER
# gauss/shuffle, so enabling it does NOT disturb any existing arm's draws --
# unlike --ladder, which is drawn before them.
PLANE="${PLANE:-0}"

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
  --signs "pos neg"     Which signs to decompose (default: both).
                        Use "pos" to skip the negative side, whose
                        faratom fit collapses to the zero vector.
  --allow-existing      Skip arm files already on HF instead of aborting.
  --ladder              Also build the angle-ladder arms (lad10..lad85, ladstar).
                        Off by default. Enabling it shifts the rng state that
                        gauss/shuffle consume, so use a distinct --tag.
  --plane               Also build the J19 orthogonal-plane arms: at each angle
                        in PLANE_DEG, pin<deg> (perpendicular drawn inside the
                        picked-atom span) and pout<deg> (perpendicular drawn
                        from its orthogonal complement). Same cos with v, span
                        content differing by up to 1289x. Drawn after
                        gauss/shuffle, so it disturbs no existing arm.
                        For an INCREMENTAL build that adds new arms (faratom,
                        the _jw sweep) beside ones already uploaded. Default
                        is still to abort, so a plain re-run cannot overwrite.
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
        --signs)          SIGNS="$2"; shift 2 ;;
        --allow-existing) ALLOW_EXISTING=1; shift ;;
        --ladder)         LADDER=1; shift ;;
        --plane)          PLANE=1; shift ;;
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

log "model=$MODEL  track=$TRACK  layer=$LAYER  tag=$TAG  alpha=$ALPHA  k=$K  n_candidates=$N_CANDIDATES  signs=$SIGNS"
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

LADDER_FLAG="--no-ladder"
[ "$LADDER" -eq 1 ] && LADDER_FLAG=""
log "ladder: $([ "$LADDER" -eq 1 ] && echo ON || echo off)"

PLANE_FLAG=""
[ "$PLANE" -eq 1 ] && PLANE_FLAG="--plane"
log "plane:  $([ "$PLANE" -eq 1 ] && echo ON || echo off)"

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
    --signs $SIGNS \
    $LADDER_FLAG \
    $PLANE_FLAG \
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
# Glob rather than hardcode. The builder also emits _faratom and the _jw/_jwr/
# _jwf sweep arms; the old hardcoded list of 8 meant those were built on every
# run, never verified, never uploaded, and lost with the pod (2026-09-27).
expected = sorted(q.name for q in folder.glob(f"{tag}_*_layer{layer}.npy"))
signs = [s for s in ("pos", "neg") if any(f"{tag}_{s}_" in n for n in expected)]
for sign in signs:
    for arm in ("full", "jspace", "resid"):
        need = f"{tag}_{sign}_{arm}_layer{layer}.npy"
        assert need in expected, f"missing required arm: {need}"
print(f"verifying {len(expected)} arm files across signs {signs}", flush=True)
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

# The plane arms make a geometric CLAIM -- same angle, different span content --
# and a construction that silently fails that claim is the N_LADDER_DRAWS bug
# all over again (2026-10-08: ladstar's realised cos was 0.504, not 0.319, and
# every statement about it had to be withdrawn). So check the claim here, before
# a single GPU-hour is spent on arms that do not mean what their names say.
for sign in signs:
    plane = report.get(sign, {}).get("plane")
    if not plane:
        continue
    frac_j = report[sign]["frac_jspace"]
    print(f"\nplane check ({sign}): frac_jspace = {frac_j:.4f}, "
          f"span dim = {report[sign]['plane_span_dim']}", flush=True)
    for name, d in sorted(plane.items(), key=lambda kv: (kv[0][1], kv[1]["nominal_deg"])):
        th = np.radians(d["nominal_deg"])
        want = (np.cos(th) ** 2 * frac_j
                + (np.sin(th) ** 2 if name.startswith("pin") else 0.0))
        assert abs(d["realised_deg"] - d["nominal_deg"]) < 0.05, (
            f"{name}: realised {d['realised_deg']:.3f} deg != nominal "
            f"{d['nominal_deg']:.1f} deg -- the construction is wrong")
        assert abs(d["frac_span"] - want) < 2e-3, (
            f"{name}: frac_span {d['frac_span']:.4f} != predicted {want:.4f}")
        print(f"  OK {name:>8}  {d['realised_deg']:6.2f} deg   "
              f"frac_span {d['frac_span']:.4f} (predicted {want:.4f})", flush=True)
    for deg in sorted({d["nominal_deg"] for d in plane.values()}):
        i = plane.get(f"pin{deg:g}"); o = plane.get(f"pout{deg:g}")
        if i and o:
            print(f"  {deg:g} deg: span-energy contrast "
                  f"{i['frac_span'] / o['frac_span']:.0f}x at identical angle", flush=True)

print(json.dumps(report, indent=2), flush=True)
print(f"All {len(expected)} arm files passed verification.", flush=True)
PY

# --------------------------------------------------------------------------
# Record provenance and upload
# --------------------------------------------------------------------------

section "upload to HF dataset"

$PYTHON_CMD - "$MODEL" "$MODEL_KEY" "$TRACK" "$LAYER" "$TAG" "$ALPHA" "$K" "$N_CANDIDATES" "$MIX" "$ALLOW_EXISTING" <<'PY' | tee -a "$LOG"
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

model, model_key, track, layer, tag, alpha, k, n_candidates, mix, allow_existing = sys.argv[1:]
allow_existing = allow_existing == "1"
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

files = sorted(
    str(q) for q in Path(folder).glob(f"{tag}_*_layer{layer}.npy"))
if not files:
    raise SystemExit(f"no arm files found under {folder} for tag {tag}")
print(f"uploading {len(files)} arm files", flush=True)
files.append(str(report_path))
files.append(str(provenance_path))

for path in files:
    if not Path(path).is_file():
        raise SystemExit(f"missing file: {path}")

api = HfApi()
existing = set(api.list_repo_files(repo, repo_type="dataset"))
collisions = existing.intersection(files)
if collisions and not allow_existing:
    raise SystemExit(
        f"STOP: these HF paths already exist: {sorted(collisions)}\n"
        "Re-run with --allow-existing to upload only the new arms, or change --tag.")
if collisions:
    print(f"--allow-existing: skipping {len(collisions)} file(s) already on HF:",
          flush=True)
    for c in sorted(collisions):
        print(f"    kept remote: {c}", flush=True)
    files = [f for f in files if f not in collisions]
    if not files:
        raise SystemExit("nothing new to upload; every arm file is already on HF.")

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
