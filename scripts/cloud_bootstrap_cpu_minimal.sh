#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod CPU-only pod running the J-space
# decomposition. Avoids the multi-gigabyte CUDA torch wheel, so it fits on
# 5 GB container disks.
#
# Usage (paste into a fresh CPU pod's terminal)::
#
#     export HF_TOKEN=hf_xxx            # required: write access to the
#                                       #           private HF dataset
#     bash scripts/cloud_bootstrap_cpu_minimal.sh
#
# Required env::
#
#     HF_TOKEN — read access to gated model repos and read+write to
#                llm-psych/llm-psych-activations
#
# Exit codes
# ----------
# 0   bootstrap complete; pod is ready
# 1   missing HF_TOKEN
# 2   git/repo error
# 3   Python env creation failed
# 4   dependency installation failed
# 5   HF auth / dataset check failed

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

REPO_URL="${REPO_URL:-https://github.com/hyderli/llm-psych.git}"
REPO_DIR="${REPO_DIR:-/workspace/llm-psych}"
GIT_REF="${GIT_REF:-main}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${REPO_DIR}/.venv-cpu"

log()  { printf '\033[1;34m[bootstrap-cpu]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[bootstrap-cpu WARN]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[bootstrap-cpu FAIL]\033[0m %s\n' "$*" >&2; exit "${2:-1}"; }

# --------------------------------------------------------------------------
# 1. Required env vars
# --------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
    fail "HF_TOKEN is not set. Export it before bootstrap:\n    export HF_TOKEN=hf_xxx" 1
fi

# --------------------------------------------------------------------------
# 2. Install uv
# --------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# --------------------------------------------------------------------------
# 3. Clone or update repo
# --------------------------------------------------------------------------

if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning $REPO_URL into $REPO_DIR…"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR" || fail "git clone failed" 2
else
    log "Repo exists; fetching latest $GIT_REF…"
    git -C "$REPO_DIR" fetch origin "$GIT_REF" || warn "git fetch failed; continuing."
fi

git -C "$REPO_DIR" checkout "$GIT_REF" || fail "git checkout $GIT_REF failed" 2
git -C "$REPO_DIR" pull --ff-only origin "$GIT_REF" || warn "git pull failed; continuing with current checkout."

cd "$REPO_DIR"
log "Repo at $(git rev-parse --short HEAD) on branch $(git rev-parse --abbrev-ref HEAD)"

# --------------------------------------------------------------------------
# 4. Create minimal CPU Python environment
# --------------------------------------------------------------------------

log "Creating CPU venv at $VENV_DIR…"
uv venv "$VENV_DIR" --python "$PYTHON_VERSION" || fail "venv creation failed" 3
PYTHON="$VENV_DIR/bin/python"

log "Installing CPU-only torch + runtime deps…"
uv pip install --python "$PYTHON" \
    torch --index-url https://download.pytorch.org/whl/cpu \
    || fail "torch CPU install failed" 4

uv pip install --python "$PYTHON" \
    transformers \
    safetensors \
    accelerate \
    huggingface_hub \
    scipy \
    numpy \
    python-dotenv \
    pyyaml \
    httpx \
    tqdm \
    || fail "dependency install failed" 4

log "Python env ready: $PYTHON"

# --------------------------------------------------------------------------
# 5. Write .env
# --------------------------------------------------------------------------

log "Writing .env…"
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env
chmod 600 .env

# --------------------------------------------------------------------------
# 6. HF dataset accessibility check
# --------------------------------------------------------------------------

log "Checking HF dataset accessibility…"
$PYTHON - <<'PY' || fail "HF dataset check failed" 5
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

sys.path.insert(0, str(Path.cwd() / "src"))
from llm_psych.hf_sync import DEFAULT_DATASET_REPO_ID, list_remote

files = list_remote(repo_id=DEFAULT_DATASET_REPO_ID)
print(f"HF dataset {DEFAULT_DATASET_REPO_ID}: {len(files)} files visible")
PY

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------

log "Bootstrap complete. Pod is ready."
log "Next: bash scripts/cloud_decompose.sh --shutdown"
